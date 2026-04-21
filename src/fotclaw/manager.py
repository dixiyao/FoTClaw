from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fot.fot_client import LocalReasoningClient
from fot.fot_server import GlobalReasoningServer, choose_most_common_model
from fotclaw.config import (
    DEFAULT_SETTINGS,
    DEFAULT_RUNTIME_STATE,
    Layout,
    ensure_layout,
    load_config,
    save_config,
)
from fotclaw.models import AgentRecord
from fotclaw.openclaw_adapter import (
    OpenClawError,
    build_augmented_message,
    delete_agent as delete_openclaw_agent,
    ensure_agent_exists,
    extract_transcript_text,
    get_agent_workspace,
    list_openclaw_agents,
    parse_openclaw_agent_args,
    render_insight_markdown,
    resolve_default_model,
    run_openclaw_prompt,
    write_workspace_insights,
)
from fotclaw.store import agent_dir, list_agents as list_store_agents, load_agent, next_trace_index, remove_agent, save_agent


TERMINAL_STATUSES = {"finished", "broken", "stopped"}
POSTPROCESS_TERMINAL_STATUSES = {"idle", "finished", "broken", "skipped"}
AGGREGATION_AGENT_NAME = "fotaggregation"
AGGREGATE_RECORD_ID = "agt-aggregate"
LEGACY_AGGREGATION_AGENT_PREFIX = "fotclaw-aggregate-"


def create_background_agent(
    home: str | Path | None,
    raw_agent_args: list[str],
    *,
    preferred_name: str | None = None,
    allow_existing_named: bool = False,
) -> AgentRecord:
    layout = ensure_layout(home)
    config = load_config(layout)
    parsed = parse_openclaw_agent_args(raw_agent_args)
    model = parsed["model"] or config.get("default_model") or resolve_default_model()
    if not model:
        raise OpenClawError(
            "No OpenClaw model could be resolved. Pass `--model ...` in the create command "
            "or set FOTCLAW_DEFAULT_MODEL."
        )

    agent_name = sanitize_agent_name(preferred_name) if preferred_name else None
    agent_id = f"agt-{agent_name}" if agent_name else _generate_agent_id()
    existing = load_agent(layout, agent_id)
    if existing is not None and not allow_existing_named:
        raise OpenClawError(f"Agent `{agent_id}` already exists.")
    openclaw_agent_name = f"fotclaw-{agent_id}"
    workspace = agent_dir(layout, agent_id) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    if existing is not None:
        record = existing
        record.model = model
        record.original_message = parsed["message"]
        record.runtime_args = parsed["runtime_args"]
        record.raw_agent_args = raw_agent_args
        record.status = "starting"
        record.started_at = None
        record.finished_at = None
        record.trace_path = None
        record.transcript_path = None
        record.postprocess_status = "idle"
        record.postprocess_error = None
        record.postprocess_pid = None
        record.postprocess_pgid = None
        record.postprocess_started_at = None
        record.postprocess_finished_at = None
        record.exit_code = None
        record.error = None
        record.timed_out = False
        record.stop_requested = False
        record.usage = {}
        record.auto_aggregated = False
        record.session_id = f"fotclaw_{agent_id}"
        record.workspace = str(workspace)
        record.stdout_path = str(agent_dir(layout, agent_id) / "stdout.log")
        record.stderr_path = str(agent_dir(layout, agent_id) / "stderr.log")
        Path(record.stdout_path).write_text("", encoding="utf-8")
        Path(record.stderr_path).write_text("", encoding="utf-8")
    else:
        record = AgentRecord(
            id=agent_id,
            name=agent_name,
            openclaw_agent_name=openclaw_agent_name,
            model=model,
            original_message=parsed["message"],
            runtime_args=parsed["runtime_args"],
            raw_agent_args=raw_agent_args,
            status="starting",
            created_at=time.time(),
            updated_at=time.time(),
            session_id=f"fotclaw_{agent_id}",
            workspace=str(workspace),
            stdout_path=str(agent_dir(layout, agent_id) / "stdout.log"),
            stderr_path=str(agent_dir(layout, agent_id) / "stderr.log"),
        )
    save_agent(layout, record)

    log_path = agent_dir(layout, agent_id) / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    supervisor_env = dict(os.environ)
    src_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = supervisor_env.get("PYTHONPATH")
    supervisor_env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_root)
    )

    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fotclaw.supervisor",
                "--home",
                str(layout.home),
                "--bootstrap",
                agent_id,
            ],
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            cwd=str(layout.home),
            env=supervisor_env,
        )

    record.supervisor_pid = proc.pid
    record.supervisor_pgid = proc.pid
    save_agent(layout, record)
    return record


def ensure_named_agent(home: str | Path | None, name: str, raw_agent_args: list[str] | None = None) -> tuple[AgentRecord, str]:
    layout = ensure_layout(home)
    normalized_name = sanitize_agent_name(name)
    agent_id = f"agt-{normalized_name}"
    existing = load_agent(layout, agent_id)

    if existing is not None:
        existing = reconcile_agent(layout, existing)
        if not raw_agent_args:
            return existing, "existing"
        if existing.status in {"starting", "running"}:
            return existing, "running"
        record = create_background_agent(
            home,
            raw_agent_args,
            preferred_name=normalized_name,
            allow_existing_named=True,
        )
        return record, "restarted"

    if not raw_agent_args:
        config = load_config(layout)
        model = config.get("default_model") or resolve_default_model()
        if not model:
            raise OpenClawError("No default model is configured for named agent creation.")
        openclaw_agent_name = f"fotclaw-{agent_id}"
        workspace = agent_dir(layout, agent_id) / "workspace"
        with _agent_creation_lock(layout):
            ensure_agent_exists(
                openclaw_agent_name,
                model,
                workspace,
                openclaw_path=config.get("openclaw_path"),
            )
        actual_workspace = get_agent_workspace(openclaw_agent_name, openclaw_path=config.get("openclaw_path"))
        record = AgentRecord(
            id=agent_id,
            name=normalized_name,
            openclaw_agent_name=openclaw_agent_name,
            model=model,
            original_message="",
            status="created",
            created_at=time.time(),
            updated_at=time.time(),
            session_id=f"fotclaw_{agent_id}",
            workspace=str(actual_workspace),
            stdout_path=str(agent_dir(layout, agent_id) / "stdout.log"),
            stderr_path=str(agent_dir(layout, agent_id) / "stderr.log"),
        )
        save_agent(layout, record)
        return record, "created"

    record = create_background_agent(home, raw_agent_args, preferred_name=normalized_name)
    return record, "started"


def run_agent_bootstrap(home: str | Path | None, agent_id: str) -> AgentRecord:
    layout = ensure_layout(home)
    config = load_config(layout)
    record = load_agent(layout, agent_id)
    if record is None:
        raise OpenClawError(f"Unknown agent: {agent_id}")

    workspace = agent_dir(layout, agent_id) / "workspace"
    with _agent_creation_lock(layout):
        ensure_agent_exists(
            record.openclaw_agent_name,
            record.model,
            workspace,
            openclaw_path=config.get("openclaw_path"),
        )
    actual_workspace = get_agent_workspace(record.openclaw_agent_name, openclaw_path=config.get("openclaw_path"))
    record.workspace = str(actual_workspace)
    record.status = "running"
    record.error = None
    save_agent(layout, record)
    return run_agent_supervisor(home, agent_id)


def run_agent_supervisor(home: str | Path | None, agent_id: str) -> AgentRecord:
    layout = ensure_layout(home)
    config = load_config(layout)
    record = load_agent(layout, agent_id)
    if record is None:
        raise OpenClawError(f"Unknown agent: {agent_id}")

    workspace = Path(record.workspace)
    write_workspace_insights(workspace, layout.insight_markdown)
    record.augmented_message = build_augmented_message(record.original_message, layout.insight_markdown.exists())
    record.status = "running"
    record.started_at = record.started_at or time.time()
    save_agent(layout, record)

    interrupted = {"value": False}

    def _handle_stop(signum: int, _frame: Any) -> None:
        interrupted["value"] = True
        record.stop_requested = True
        record.status = "stopped"
        record.error = f"Stopped by signal {signum}"
        save_agent(layout, record)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    try:
        result = run_openclaw_prompt(
            agent_name=record.openclaw_agent_name,
            prompt=record.augmented_message,
            workspace=workspace,
            session_id=record.session_id,
            runtime_args=record.runtime_args,
            timeout_seconds=float(os.environ.get("FOTCLAW_AGENT_TIMEOUT_SECONDS", "7200")),
            openclaw_path=config.get("openclaw_path"),
        )
    except BaseException as exc:
        if interrupted["value"]:
            raise
        record.status = "broken"
        record.finished_at = time.time()
        record.error = str(exc)
        save_agent(layout, record)
        return record

    Path(record.stdout_path).write_text(result["stdout"], encoding="utf-8")
    Path(record.stderr_path).write_text(result["stderr"], encoding="utf-8")
    record.transcript_path = result.get("transcript_path")
    record.usage = result.get("usage", {})
    record.exit_code = result.get("exit_code")
    record.timed_out = bool(result.get("timed_out"))

    if record.stop_requested:
        record.status = "stopped"
    elif result["status"] == "success":
        record.status = "finished"
    else:
        record.status = "broken"
        record.error = result.get("stderr") or result["status"]

    record.finished_at = time.time()
    record.postprocess_status = "pending" if result.get("transcript") else "skipped"
    record.postprocess_error = None
    save_agent(layout, record)
    if result.get("transcript") and record.status in {"finished", "broken"} and not record.stop_requested:
        _start_postprocess_worker(layout, agent_id)
    return record


def run_agent_postprocessor(home: str | Path | None, agent_id: str) -> AgentRecord:
    layout = ensure_layout(home)
    config = load_config(layout)
    record = load_agent(layout, agent_id)
    if record is None:
        raise OpenClawError(f"Unknown agent: {agent_id}")
    if record.stop_requested:
        record.postprocess_status = "skipped"
        record.postprocess_finished_at = time.time()
        save_agent(layout, record)
        return record

    transcript_entries = _load_transcript_entries(record.transcript_path)
    transcript_text = extract_transcript_text(transcript_entries)
    if not transcript_text:
        record.postprocess_status = "skipped"
        record.postprocess_error = "Transcript missing or empty."
        record.postprocess_finished_at = time.time()
        save_agent(layout, record)
        return record

    record.postprocess_status = "running"
    record.postprocess_started_at = record.postprocess_started_at or time.time()
    record.postprocess_error = None
    save_agent(layout, record)

    try:
        reasoning = _extract_reasoning_trace(
            layout=layout,
            config=config,
            record=record,
            transcript_text=transcript_text,
        )
        trace_path = _save_trace(layout, record, reasoning)
        record.trace_path = str(trace_path)
        record.postprocess_status = "finished"
        record.auto_aggregated = _maybe_auto_aggregate(layout, config)
    except BaseException as exc:
        record.postprocess_status = "broken"
        record.postprocess_error = str(exc)

    record.postprocess_finished_at = time.time()
    save_agent(layout, record)
    return record


def _extract_reasoning_trace(
    *,
    layout: Layout,
    config: dict[str, Any],
    record: AgentRecord,
    transcript_text: str,
) -> dict[str, Any]:
    reader_class = _load_local_reasoning_client_class(config)
    reader = reader_class(
        agent_name=record.openclaw_agent_name,
        workspace=record.workspace,
        openclaw_path=config.get("openclaw_path"),
        output_dir=str(layout.traces_dir),
        timeout_seconds=float(os.environ.get("FOTCLAW_TRACE_TIMEOUT_SECONDS", "7200")),
    )
    return reader.extract_from_trace(problem=record.original_message, solution=transcript_text)


def _save_trace(layout: Layout, record: AgentRecord, reasoning: dict[str, Any]) -> Path:
    index = next_trace_index(layout)
    trace_path = layout.traces_dir / f"problem_{index:06d}.json"
    payload = {
        "problem": reasoning.get("problem"),
        "agent_id": record.id,
        "openclaw_agent_name": record.openclaw_agent_name,
        "model": record.model,
        "status": record.status,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
        "solution": reasoning.get("solution"),
        "reflection": reasoning.get("reflection"),
        "skills_extracted": reasoning.get("skills_extracted", {}),
        "insight_book": reasoning.get("insight_book", {}),
        "usage": reasoning.get("usage", {}),
        "transcript_path": record.transcript_path,
    }
    trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return trace_path


def _maybe_auto_aggregate(layout: Layout, config: dict[str, Any]) -> bool:
    if not config.get("auto_aggregate_enabled", True):
        return False
    min_interval = int(config.get("auto_aggregate_min_interval_seconds", 0) or 0)
    last_aggregate_at = config.get("last_aggregate_at")
    if min_interval > 0 and last_aggregate_at:
        try:
            last_dt = datetime.fromisoformat(str(last_aggregate_at))
            last_ts = last_dt.timestamp()
            if (time.time() - last_ts) < min_interval:
                return False
        except ValueError:
            pass
    trace_files = sorted(layout.traces_dir.glob("problem_*.json"))
    if len(trace_files) < int(config.get("auto_aggregate_trace_threshold", 25)):
        return False
    if len(trace_files) <= int(config.get("last_aggregated_trace_count", 0)):
        return False
    try:
        _run_aggregate_job(layout.home)
    except OpenClawError as exc:
        if "already running" in str(exc):
            return False
        raise
    return True


def _run_aggregate_job(home: str | Path | None) -> dict[str, Any]:
    layout = ensure_layout(home)
    config = load_config(layout)
    with _aggregation_lock(layout):
        trace_files = sorted(layout.traces_dir.glob("problem_*.json"))
        if not trace_files:
            last_aggregate_at = config.get("last_aggregate_at")
            if last_aggregate_at:
                raise OpenClawError(
                    "No new reasoning traces are available to aggregate. "
                    f"The last successful aggregation was at {last_aggregate_at} and consumed the prior trace batch."
                )
            raise OpenClawError("No reasoning traces are available to aggregate.")

        model = config.get("aggregation_model") or config.get("default_model") or _resolve_aggregation_model(trace_files)
        if not model:
            raise OpenClawError("Could not determine an aggregation model.")

        workspace = layout.aggregate_dir / "workspace"
        with _agent_creation_lock(layout):
            ensure_agent_exists(
                AGGREGATION_AGENT_NAME,
                model,
                workspace,
                openclaw_path=config.get("openclaw_path"),
            )
        write_workspace_insights(workspace, layout.insight_markdown)

        server_class = _load_global_reasoning_server_class(config)
        server = server_class(
            agent_name=AGGREGATION_AGENT_NAME,
            workspace=workspace,
            openclaw_path=config.get("openclaw_path"),
            input_dirs=[str(layout.traces_dir)],
            num_insights=None,
            timeout_seconds=float(os.environ.get("FOTCLAW_AGGREGATE_TIMEOUT_SECONDS", "7200")),
        )
        result = server.aggregate_and_build_encyclopedia(
            json_files=[str(path) for path in trace_files],
            output_dir=str(layout.aggregate_dir),
            existing_encyclopedia_path=str(layout.insight_json),
        )
        json_path, markdown_path = server.save_results(result, output_dir=str(layout.aggregate_dir))
        Path(json_path).replace(layout.insight_json)
        Path(markdown_path).replace(layout.insight_markdown)
        (layout.aggregate_dir / "aggregation_result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        deleted_trace_count = 0
        for trace_path in trace_files:
            if trace_path.exists():
                trace_path.unlink(missing_ok=True)
                deleted_trace_count += 1

        config["last_aggregate_at"] = _utc_now_iso()
        config["last_aggregate_model"] = model
        config["last_aggregated_trace_count"] = 0
        save_config(layout, config)

        return {
            "model": model,
            "trace_count": len(trace_files),
            "deleted_trace_count": deleted_trace_count,
            "insight_json": str(layout.insight_json),
            "insight_markdown": str(layout.insight_markdown),
        }


def start_background_aggregate(home: str | Path | None) -> AgentRecord:
    layout = ensure_layout(home)
    config = load_config(layout)
    trace_files = sorted(layout.traces_dir.glob("problem_*.json"))
    if not trace_files:
        raise OpenClawError("No reasoning traces are available to aggregate.")

    model = config.get("aggregation_model") or config.get("default_model") or _resolve_aggregation_model(trace_files)
    if not model:
        raise OpenClawError("Could not determine an aggregation model.")

    existing = load_agent(layout, AGGREGATE_RECORD_ID)
    if existing is not None:
        existing = reconcile_agent(layout, existing)
        if existing.status in {"starting", "running"}:
            return existing
        record = existing
        record.model = model
        record.status = "starting"
        record.started_at = None
        record.finished_at = None
        record.error = None
        record.exit_code = None
        record.timed_out = False
        record.stop_requested = False
        record.trace_path = None
        record.transcript_path = None
        record.postprocess_status = "idle"
        record.postprocess_error = None
        record.postprocess_started_at = None
        record.postprocess_finished_at = None
        record.postprocess_pid = None
        record.postprocess_pgid = None
        record.stdout_path = str(layout.aggregate_dir / "stdout.log")
        record.stderr_path = str(layout.aggregate_dir / "stderr.log")
        Path(record.stdout_path).write_text("", encoding="utf-8")
        Path(record.stderr_path).write_text("", encoding="utf-8")
    else:
        record = AgentRecord(
            id=AGGREGATE_RECORD_ID,
            name="aggregate",
            openclaw_agent_name=AGGREGATION_AGENT_NAME,
            model=model,
            original_message="Aggregate reasoning traces into the persistent insight library.",
            status="starting",
            created_at=time.time(),
            updated_at=time.time(),
            session_id="fotclaw_aggregate",
            workspace=str(layout.aggregate_dir / "workspace"),
            stdout_path=str(layout.aggregate_dir / "stdout.log"),
            stderr_path=str(layout.aggregate_dir / "stderr.log"),
        )
    save_agent(layout, record)

    log_path = layout.aggregate_dir / "supervisor.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_env = dict(os.environ)
    src_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = worker_env.get("PYTHONPATH")
    worker_env["PYTHONPATH"] = f"{src_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_root)
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fotclaw.supervisor",
                "--home",
                str(layout.home),
                "--aggregate",
                AGGREGATE_RECORD_ID,
            ],
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            cwd=str(layout.home),
            env=worker_env,
        )
    record.supervisor_pid = proc.pid
    record.supervisor_pgid = proc.pid
    save_agent(layout, record)
    return record


def run_aggregate_supervisor(home: str | Path | None, agent_id: str = AGGREGATE_RECORD_ID) -> AgentRecord:
    layout = ensure_layout(home)
    record = load_agent(layout, agent_id)
    if record is None:
        raise OpenClawError(f"Unknown agent: {agent_id}")

    record.status = "running"
    record.started_at = record.started_at or time.time()
    record.error = None
    save_agent(layout, record)

    try:
        result = _run_aggregate_job(layout.home)
        Path(record.stdout_path).write_text(
            json.dumps(
                {
                    "last_aggregate_at": load_config(layout).get("last_aggregate_at"),
                    "insight_markdown": result["insight_markdown"],
                    "insight_json": result["insight_json"],
                    "deleted_trace_count": result["deleted_trace_count"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        Path(record.stderr_path).write_text("", encoding="utf-8")
        record.status = "finished"
    except BaseException as exc:
        Path(record.stderr_path).write_text(str(exc), encoding="utf-8")
        record.status = "broken"
        record.error = str(exc)

    record.finished_at = time.time()
    record.postprocess_status = "skipped"
    record.postprocess_finished_at = record.finished_at
    save_agent(layout, record)
    return record


def list_agents(home: str | Path | None) -> list[AgentRecord]:
    layout = ensure_layout(home)
    agents = list_store_agents(layout)
    refreshed: list[AgentRecord] = []
    for record in agents:
        record = _normalize_record_paths(layout, record)
        refreshed.append(reconcile_agent(layout, record))
    return refreshed


def reconcile_agent(layout: Layout, record: AgentRecord) -> AgentRecord:
    if record.status not in TERMINAL_STATUSES:
        pid = record.supervisor_pid
        if not pid:
            record.status = "broken"
            record.error = record.error or "Supervisor PID missing."
            record.finished_at = record.finished_at or time.time()
            save_agent(layout, record)
            return record
        if _pid_exists(pid):
            return record
        _kill_process_group(record.supervisor_pgid or pid)
        record.status = "stopped" if record.stop_requested else "broken"
        record.error = record.error or "Supervisor exited unexpectedly."
        record.finished_at = record.finished_at or time.time()
        save_agent(layout, record)
        return record

    if record.postprocess_status not in POSTPROCESS_TERMINAL_STATUSES:
        pid = record.postprocess_pid
        if pid and _pid_exists(pid):
            return record
        if pid:
            _kill_process_group(record.postprocess_pgid or pid)
        if record.postprocess_status in {"pending", "running"}:
            record.postprocess_status = "broken"
            record.postprocess_error = record.postprocess_error or "Postprocess worker exited unexpectedly."
            record.postprocess_finished_at = record.postprocess_finished_at or time.time()
            save_agent(layout, record)
    return record


def stop_agent(home: str | Path | None, agent_id: str) -> AgentRecord:
    layout = ensure_layout(home)
    record = load_agent(layout, agent_id)
    if record is None:
        raise OpenClawError(f"Unknown agent: {agent_id}")
    record.stop_requested = True
    save_agent(layout, record)
    if record.supervisor_pgid:
        _kill_process_group(record.supervisor_pgid)
    elif record.supervisor_pid:
        _kill_process(record.supervisor_pid)
    if record.postprocess_pgid:
        _kill_process_group(record.postprocess_pgid)
    elif record.postprocess_pid:
        _kill_process(record.postprocess_pid)
    return reconcile_agent(layout, record)


def delete_agent(home: str | Path | None, *, agent_id: str | None = None, name: str | None = None) -> dict[str, Any]:
    layout = ensure_layout(home)
    config = load_config(layout)
    if not agent_id and not name:
        raise OpenClawError("Provide either an agent id or a name.")

    resolved_id = agent_id
    if name:
        resolved_id = f"agt-{sanitize_agent_name(name)}"
    if not resolved_id:
        raise OpenClawError("Could not resolve an agent id.")

    record = load_agent(layout, resolved_id)
    openclaw_name = record.openclaw_agent_name if record else f"fotclaw-{resolved_id}"
    stopped = False
    local_removed = False
    openclaw_removed = False

    if record is not None:
        refreshed = reconcile_agent(layout, record)
        if refreshed.status not in TERMINAL_STATUSES or refreshed.postprocess_status not in POSTPROCESS_TERMINAL_STATUSES:
            stop_agent(layout.home, refreshed.id)
            stopped = True
        remove_agent(layout, refreshed.id)
        local_removed = True

    openclaw_removed = delete_openclaw_agent(openclaw_name, openclaw_path=config.get("openclaw_path"))
    leftover_root = agent_dir(layout, resolved_id)
    if leftover_root.exists():
        _clear_directory(leftover_root)
        local_removed = True

    return {
        "agent_id": resolved_id,
        "openclaw_agent": openclaw_name,
        "stopped": stopped,
        "local_removed": local_removed,
        "openclaw_removed": openclaw_removed,
    }


def show_agent(home: str | Path | None, agent_id: str) -> dict[str, Any]:
    layout = ensure_layout(home)
    record = load_agent(layout, agent_id)
    if record is None:
        raise OpenClawError(f"Unknown agent: {agent_id}")
    record = _normalize_record_paths(layout, record)
    record = reconcile_agent(layout, record)
    transcript = ""
    if record.transcript_path and Path(record.transcript_path).exists():
        transcript = Path(record.transcript_path).read_text(encoding="utf-8")
    stdout = Path(record.stdout_path).read_text(encoding="utf-8") if record.stdout_path and Path(record.stdout_path).exists() else ""
    stderr = Path(record.stderr_path).read_text(encoding="utf-8") if record.stderr_path and Path(record.stderr_path).exists() else ""
    trace_payload = None
    if record.trace_path and Path(record.trace_path).exists():
        try:
            trace_payload = json.loads(Path(record.trace_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            trace_payload = None
    return {
        "record": record,
        "transcript": transcript,
        "stdout": stdout,
        "stderr": stderr,
        "trace": trace_payload,
    }


def show_aggregate_agent(home: str | Path | None) -> dict[str, Any]:
    layout = ensure_layout(home)
    config = load_config(layout)
    record = load_agent(layout, AGGREGATE_RECORD_ID)
    if record is None:
        raise OpenClawError("Aggregate agent has not been created yet. Run `fotclaw aggregate` first.")
    record = reconcile_agent(layout, record)
    insight_text = layout.insight_markdown.read_text(encoding="utf-8") if layout.insight_markdown.exists() else ""
    stdout = Path(record.stdout_path).read_text(encoding="utf-8") if record.stdout_path and Path(record.stdout_path).exists() else ""
    stderr = Path(record.stderr_path).read_text(encoding="utf-8") if record.stderr_path and Path(record.stderr_path).exists() else ""
    return {
        "record": record,
        "stdout": stdout,
        "stderr": stderr,
        "insight_markdown": insight_text,
        "last_aggregate_at": config.get("last_aggregate_at"),
        "insight_markdown_path": str(layout.insight_markdown),
    }


def clean_agents(home: str | Path | None, reporter: Any | None = None) -> dict[str, Any]:
    layout = ensure_layout(home)
    config = load_config(layout)
    removed_agent_ids: list[str] = []
    stopped_agent_ids: list[str] = []
    removed_openclaw_agents: list[str] = []

    def report(message: str) -> None:
        if reporter is not None:
            reporter(message)

    records = list_store_agents(layout)
    report(f"Scanning {len(records)} FoTClaw agent(s)...")
    for record in records:
        refreshed = reconcile_agent(layout, record)
        report(f"Cleaning {refreshed.id}...")
        if refreshed.status not in TERMINAL_STATUSES or refreshed.postprocess_status not in POSTPROCESS_TERMINAL_STATUSES:
            stop_agent(layout.home, refreshed.id)
            stopped_agent_ids.append(refreshed.id)
            report(f"  stopped {refreshed.id}")
        try:
            if delete_openclaw_agent(refreshed.openclaw_agent_name, openclaw_path=config.get("openclaw_path")):
                removed_openclaw_agents.append(refreshed.openclaw_agent_name)
                report(f"  removed OpenClaw agent {refreshed.openclaw_agent_name}")
        except OpenClawError:
            report(f"  could not remove OpenClaw agent {refreshed.openclaw_agent_name}")
        if agent_dir(layout, refreshed.id).exists():
            _clear_directory(agent_dir(layout, refreshed.id))
            report(f"  removed local state for {refreshed.id}")
        removed_agent_ids.append(refreshed.id)

    report("Cleaning aggregation agents...")
    for agent_name in list_openclaw_agents(openclaw_path=config.get("openclaw_path")):
        if agent_name == AGGREGATION_AGENT_NAME or agent_name.startswith(LEGACY_AGGREGATION_AGENT_PREFIX):
            try:
                if delete_openclaw_agent(agent_name, openclaw_path=config.get("openclaw_path")):
                    removed_openclaw_agents.append(agent_name)
                    report(f"  removed OpenClaw agent {agent_name}")
            except OpenClawError:
                report(f"  could not remove OpenClaw agent {agent_name}")

    report("Clearing stored traces...")
    for path in sorted(layout.traces_dir.glob("problem_*.json")):
        path.unlink(missing_ok=True)

    report("Clearing aggregate scratch data...")
    if layout.aggregate_dir.exists():
        _clear_directory(layout.aggregate_dir)
    layout.aggregate_dir.mkdir(parents=True, exist_ok=True)

    report("Resetting aggregation runtime state...")
    config.update(DEFAULT_RUNTIME_STATE)
    save_config(layout, config)

    return {
        "removed_agent_ids": sorted(set(removed_agent_ids)),
        "stopped_agent_ids": sorted(set(stopped_agent_ids)),
        "removed_openclaw_agents": sorted(set(removed_openclaw_agents)),
        "cleared_traces": True,
        "cleared_aggregate_workspace": True,
        "preserved_insight_markdown": str(layout.insight_markdown),
        "preserved_insight_json": str(layout.insight_json),
    }

def print_agent_table(records: list[AgentRecord]) -> str:
    lines = []
    for record in records:
        lines.append(
            f"{record.id}\t{record.status}\t{record.model}\t"
            f"{_format_timestamp(record.created_at)}"
        )
    return "\n".join(lines)


def _resolve_aggregation_model(trace_files: list[Path]) -> str | None:
    models: list[str] = []
    for path in trace_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("model"):
            models.append(str(payload["model"]))
    return choose_most_common_model(models)


def _split_class_spec(spec: str) -> tuple[str, str]:
    value = spec.strip()
    if ":" in value:
        module_name, class_name = value.split(":", 1)
        if module_name and class_name:
            return module_name, class_name
    module_name, separator, class_name = value.rpartition(".")
    if separator and module_name and class_name:
        return module_name, class_name
    raise OpenClawError(
        f"Invalid class path `{spec}`. Use `package.module:ClassName` or `package.module.ClassName`."
    )


def _load_class(spec: str, expected_base: type[Any], label: str) -> type[Any]:
    module_name, class_name = _split_class_spec(spec)
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise OpenClawError(f"Could not import {label} module `{module_name}` from `{spec}`.") from exc
    cls = getattr(module, class_name, None)
    if not isinstance(cls, type):
        raise OpenClawError(f"{label} `{spec}` does not resolve to a class.")
    if not issubclass(cls, expected_base):
        raise OpenClawError(
            f"{label} `{spec}` must inherit from `{expected_base.__module__}.{expected_base.__name__}`."
        )
    return cls


def _load_local_reasoning_client_class(config: dict[str, Any]) -> type[LocalReasoningClient]:
    spec = str(config.get("local_reasoning_class") or DEFAULT_SETTINGS["local_reasoning_class"])
    loaded = _load_class(spec, LocalReasoningClient, "Local reasoning class")
    return loaded


def _load_global_reasoning_server_class(config: dict[str, Any]) -> type[GlobalReasoningServer]:
    spec = str(config.get("global_reasoning_class") or DEFAULT_SETTINGS["global_reasoning_class"])
    loaded = _load_class(spec, GlobalReasoningServer, "Global reasoning class")
    return loaded


def _generate_agent_id() -> str:
    return f"agt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(3)}"


def sanitize_agent_name(name: str) -> str:
    import re

    value = name.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    if not value:
        raise OpenClawError("Agent name must contain at least one alphanumeric character.")
    if value.startswith("agt-"):
        value = value[4:]
    return value


def _normalize_record_paths(layout: Layout, record: AgentRecord) -> AgentRecord:
    if record.trace_path:
        legacy_prefix = str(layout.legacy_traces_dir) + os.sep
        current_prefix = str(layout.traces_dir) + os.sep
        if record.trace_path.startswith(legacy_prefix):
            suffix = record.trace_path[len(legacy_prefix) :]
            candidate = layout.traces_dir / suffix
            if candidate.exists():
                record.trace_path = str(candidate)
                save_agent(layout, record)
        elif not record.trace_path.startswith(current_prefix):
            candidate = layout.traces_dir / Path(record.trace_path).name
            if candidate.exists():
                record.trace_path = str(candidate)
                save_agent(layout, record)
    return record


def _load_transcript_entries(transcript_path: str | None) -> list[dict[str, Any]]:
    if not transcript_path:
        return []
    path = Path(transcript_path)
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _format_timestamp(value: float | None) -> str:
    if not value:
        return "-"
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            child.rmdir()
    path.rmdir()


@contextlib.contextmanager
def _aggregation_lock(layout: Layout):
    lock_path = layout.aggregate_dir / "aggregate.lock"
    if lock_path.exists():
        try:
            age_seconds = time.time() - lock_path.stat().st_mtime
            if age_seconds > 6 * 60 * 60:
                lock_path.unlink(missing_ok=True)
        except OSError:
            pass

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise OpenClawError("Aggregation is already running.") from exc

    try:
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


@contextlib.contextmanager
def _agent_creation_lock(layout: Layout):
    lock_path = layout.home / "agent-create.lock"
    deadline = time.time() + 180
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() >= deadline:
                raise OpenClawError("Timed out waiting for agent creation lock.")
            time.sleep(0.25)

    try:
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        yield
    finally:
        os.close(fd)
        lock_path.unlink(missing_ok=True)


def _start_postprocess_worker(layout: Layout, agent_id: str) -> None:
    record = load_agent(layout, agent_id)
    if record is None:
        return
    log_path = agent_dir(layout, agent_id) / "postprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_env = dict(os.environ)
    src_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = worker_env.get("PYTHONPATH")
    worker_env["PYTHONPATH"] = (
        f"{src_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_root)
    )

    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fotclaw.supervisor",
                "--home",
                str(layout.home),
                "--postprocess",
                agent_id,
            ],
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            cwd=str(layout.home),
            env=worker_env,
        )

    record.postprocess_pid = proc.pid
    record.postprocess_pgid = proc.pid
    record.postprocess_status = "pending"
    save_agent(layout, record)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return


def _kill_process_group(pgid: int) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        return
