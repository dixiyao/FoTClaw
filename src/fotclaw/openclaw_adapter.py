from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


USE_SHELL = platform.system() == "Windows"
MAX_OPENCLAW_MESSAGE_CHARS = int(os.environ.get("FOT_MAX_MSG_CHARS") or os.environ.get("FOTCLAW_MAX_MSG_CHARS", "8000"))


class OpenClawError(RuntimeError):
    """Raised when OpenClaw execution fails."""


def slugify_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(".", "-").lower()


def normalize_agent_name(agent_name: str) -> str:
    return agent_name.replace(":", "-").lower()


def parse_openclaw_agent_args(raw_args: list[str]) -> dict[str, Any]:
    runtime_args: list[str] = []
    message: str | None = None
    model: str | None = None
    index = 0

    while index < len(raw_args):
        arg = raw_args[index]
        value: str | None = None
        consumed = 1

        if "=" in arg and arg.startswith("--"):
            flag, value = arg.split("=", 1)
        else:
            flag = arg

        if flag in {"--message", "--model", "--agent", "--session-id"} and value is None:
            if index + 1 >= len(raw_args):
                raise OpenClawError(f"Missing value for {flag}.")
            value = raw_args[index + 1]
            consumed = 2

        if flag == "--message":
            message = value
        elif flag == "--model":
            model = value
        elif flag in {"--agent", "--session-id"}:
            pass
        else:
            runtime_args.extend(raw_args[index : index + consumed])

        index += consumed

    if not message:
        raise OpenClawError(
            "FoT requires a prompt message. Pass the same arguments you would after "
            "`openclaw agent`, including `--message \"...\"`."
        )

    return {
        "message": message,
        "model": model,
        "runtime_args": runtime_args,
    }


def build_augmented_message(message: str, has_insights: bool) -> str:
    if not has_insights:
        return message
    prefix = (
        "Before solving the task, inspect the files `INSIGHTS.md` and `insight.md` in "
        "your workspace and apply any relevant guidance from them.\n\n"
    )
    return f"{prefix}{message}"


def _coerce_subprocess_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _get_openclaw_command(openclaw_path: str | None = None) -> list[str]:
    value = (openclaw_path or os.environ.get("OPENCLAW_PATH") or "openclaw").strip()
    return [value or "openclaw"]


def _run_command(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: float | None = None,
    openclaw_path: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = _get_openclaw_command(openclaw_path) + args
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            timeout=timeout,
            shell=USE_SHELL,
        )
    except FileNotFoundError as exc:
        raise OpenClawError(
            "OpenClaw CLI not found. Install it or set OPENCLAW_PATH/FOT openclaw_path."
        ) from exc


def _default_agent_workspace(agent_name: str) -> Path:
    workspace = Path.home() / ".openclaw" / "agents" / normalize_agent_name(agent_name) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def get_agent_store_dir(agent_name: str) -> Path:
    base_dir = Path.home() / ".openclaw" / "agents"
    direct = base_dir / agent_name
    if direct.exists():
        return direct
    normalized = base_dir / normalize_agent_name(agent_name)
    if normalized.exists():
        return normalized
    return direct


def run_openclaw_agents_list(openclaw_path: str | None = None, timeout_seconds: float = 60.0) -> subprocess.CompletedProcess[str] | None:
    try:
        result = _run_command(["agents", "list"], timeout=timeout_seconds, openclaw_path=openclaw_path)
    except OpenClawError:
        return None
    if result.returncode != 0:
        return None
    return result


def list_openclaw_agents(openclaw_path: str | None = None) -> list[str]:
    result = run_openclaw_agents_list(openclaw_path=openclaw_path)
    if result is None:
        return []
    names: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            name_part = stripped[2:].split()[0] if stripped[2:].strip() else ""
            if name_part:
                names.append(name_part)
    return names


def get_agent_workspace(agent_name: str, openclaw_path: str | None = None) -> Path:
    list_result = run_openclaw_agents_list(openclaw_path=openclaw_path)
    if list_result is not None:
        normalized = normalize_agent_name(agent_name)
        found = False
        for line in list_result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith(f"- {agent_name}") or stripped.startswith(f"- {normalized}"):
                found = True
                continue
            if found and "Workspace:" in line:
                workspace_str = line.split("Workspace:", 1)[1].strip()
                if workspace_str.startswith("~/"):
                    workspace_str = str(Path.home() / workspace_str[2:])
                workspace = Path(workspace_str)
                workspace.mkdir(parents=True, exist_ok=True)
                return workspace
            if found and stripped.startswith("-"):
                break
    return _default_agent_workspace(agent_name)


def resolve_default_model() -> str | None:
    env_model = os.environ.get("FOT_DEFAULT_MODEL") or os.environ.get("FOTCLAW_DEFAULT_MODEL")
    if env_model:
        return env_model.strip()
    return "google/gemini-3.1-pro-preview"


def ensure_agent_exists(
    agent_name: str,
    model_id: str,
    workspace_dir: Path,
    *,
    openclaw_path: str | None = None,
) -> bool:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    list_result = run_openclaw_agents_list(openclaw_path=openclaw_path)
    normalized = normalize_agent_name(agent_name)

    if list_result is not None:
        existing_agents: set[str] = set()
        for line in list_result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                name_part = stripped[2:].split()[0] if stripped[2:].strip() else ""
                if name_part:
                    existing_agents.add(name_part.lower())
        if agent_name.lower() in existing_agents or normalized in existing_agents:
            current_workspace = get_agent_workspace(agent_name, openclaw_path=openclaw_path)
            if current_workspace.resolve() == workspace_dir.resolve():
                return False

    create_result = _run_command(
        [
            "agents",
            "add",
            agent_name,
            "--model",
            model_id,
            "--workspace",
            str(workspace_dir),
            "--non-interactive",
        ],
        timeout=180,
        openclaw_path=openclaw_path,
    )

    if create_result.returncode != 0:
        stderr_lower = (create_result.stderr or "").lower()
        if "already exists" in stderr_lower or "exists" in stderr_lower:
            for delete_name in {agent_name, normalized}:
                _run_command(
                    ["agents", "delete", delete_name, "--force"],
                    timeout=30,
                    openclaw_path=openclaw_path,
                )
            create_result = _run_command(
                [
                    "agents",
                    "add",
                    agent_name,
                    "--model",
                    model_id,
                    "--workspace",
                    str(workspace_dir),
                    "--non-interactive",
                ],
                timeout=180,
                openclaw_path=openclaw_path,
            )

    if create_result.returncode != 0:
        raise OpenClawError(create_result.stderr.strip() or create_result.stdout.strip() or "Failed to create OpenClaw agent.")

    bench_agent_dir = get_agent_store_dir(agent_name) / "agent"
    bench_agent_dir.mkdir(parents=True, exist_ok=True)
    bench_models = bench_agent_dir / "models.json"
    main_models = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "models.json"

    if main_models.exists():
        shutil.copy2(main_models, bench_models)
        if "/" in model_id:
            provider_name, model_name = model_id.split("/", 1)
            try:
                payload = json.loads(bench_models.read_text("utf-8-sig"))
                payload["defaultProvider"] = provider_name
                payload["defaultModel"] = model_name
                bench_models.write_text(json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
            except (json.JSONDecodeError, OSError):
                pass

    sessions_store = get_agent_store_dir(agent_name) / "sessions" / "sessions.json"
    if sessions_store.exists():
        sessions_store.unlink(missing_ok=True)

    return True


def delete_agent(agent_name: str, *, openclaw_path: str | None = None) -> bool:
    result = _run_command(
        ["agents", "delete", agent_name, "--force", "--json"],
        timeout=180,
        openclaw_path=openclaw_path,
    )
    if result.returncode == 0:
        return True
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if "not found" in combined or "unknown" in combined or "does not exist" in combined:
        return False
    raise OpenClawError(result.stderr.strip() or result.stdout.strip() or f"Failed to delete OpenClaw agent `{agent_name}`.")


def write_workspace_insights(workspace: Path, insight_markdown: Path) -> bool:
    if not insight_markdown.exists():
        return False
    content = insight_markdown.read_text(encoding="utf-8")
    workspace.mkdir(parents=True, exist_ok=True)
    for name in ("INSIGHTS.md", "insight.md"):
        (workspace / name).write_text(content, encoding="utf-8")
    return True


def _resolve_session_id_from_store(agent_name: str) -> str | None:
    sessions_store = get_agent_store_dir(agent_name) / "sessions" / "sessions.json"
    if not sessions_store.exists():
        return None
    try:
        payload = json.loads(sessions_store.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    normalized = normalize_agent_name(agent_name)
    preferred_keys = [
        f"agent:{agent_name}:main",
        f"agent:{agent_name}:default",
        f"agent:{normalized}:main",
        f"agent:{normalized}:default",
    ]
    for key in preferred_keys:
        value = payload.get(key)
        if isinstance(value, dict) and value.get("sessionId"):
            return str(value["sessionId"])
    newest: dict[str, Any] | None = None
    newest_timestamp = -1
    for value in payload.values():
        if not isinstance(value, dict) or "sessionId" not in value:
            continue
        updated_at = value.get("updatedAt")
        if isinstance(updated_at, (int, float)) and updated_at > newest_timestamp:
            newest = value
            newest_timestamp = updated_at
    if newest:
        return str(newest["sessionId"])
    return None


def _find_transcript_path_from_sessions_store(agent_name: str) -> Path | None:
    agent_dir = get_agent_store_dir(agent_name)
    sessions_store = agent_dir / "sessions" / "sessions.json"
    if not sessions_store.exists():
        return None
    try:
        payload = json.loads(sessions_store.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    def _iter_strings(node: Any):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from _iter_strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from _iter_strings(value)

    suffixes = (".jsonl", ".ndjson")
    sessions_root = agent_dir / "sessions"
    for value in _iter_strings(payload):
        if not value.endswith(suffixes):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = sessions_root / value
        if candidate.exists():
            return candidate
    return None


def _find_recent_session_path(agent_name: str, started_at: float) -> Path | None:
    sessions_dir = get_agent_store_dir(agent_name) / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = list(sessions_dir.rglob("*.jsonl")) + list(sessions_dir.rglob("*.ndjson"))
    if not candidates:
        return None
    tolerance_seconds = 5.0
    recent_candidates = [path for path in candidates if path.stat().st_mtime >= (started_at - tolerance_seconds)]
    pool = recent_candidates or candidates
    return max(pool, key=lambda path: path.stat().st_mtime)


def load_transcript(agent_name: str, session_id: str, started_at: float) -> tuple[list[dict[str, Any]], Path | None]:
    agent_dir = get_agent_store_dir(agent_name)
    transcript_path: Path | None = None

    for attempt in range(15):
        resolved_session_id = _resolve_session_id_from_store(agent_name)
        if resolved_session_id:
            session_dir = agent_dir / "sessions"
            for candidate in (
                session_dir / f"{resolved_session_id}.jsonl",
                session_dir / f"{resolved_session_id}.ndjson",
                session_dir / resolved_session_id / "transcript.jsonl",
                session_dir / resolved_session_id / "events.jsonl",
            ):
                if candidate.exists():
                    transcript_path = candidate
                    break
            if transcript_path is not None:
                break

        candidate_from_store = _find_transcript_path_from_sessions_store(agent_name)
        if candidate_from_store is not None:
            transcript_path = candidate_from_store
            break

        recent_path = _find_recent_session_path(agent_name, started_at)
        if recent_path is not None:
            transcript_path = recent_path
            break

        for direct_path in (
            agent_dir / "sessions" / f"{session_id}.jsonl",
            agent_dir / "sessions" / f"{session_id}.ndjson",
        ):
            if direct_path.exists():
                transcript_path = direct_path
                break
        if transcript_path is not None:
            break
        if attempt < 14:
            time.sleep(1.0)

    if transcript_path is None:
        return [], None

    transcript: list[dict[str, Any]] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            transcript.append(json.loads(line))
        except json.JSONDecodeError as exc:
            transcript.append({"raw": line, "parse_error": str(exc)})
    return transcript, transcript_path


def extract_usage_from_transcript(transcript: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        message = entry.get("message", {})
        if message.get("role") != "assistant":
            continue
        totals["request_count"] += 1
        usage = message.get("usage", {})
        totals["input_tokens"] += usage.get("input", 0)
        totals["output_tokens"] += usage.get("output", 0)
        totals["cache_read_tokens"] += usage.get("cacheRead", 0)
        totals["cache_write_tokens"] += usage.get("cacheWrite", 0)
        totals["total_tokens"] += usage.get("totalTokens", 0)
        cost = usage.get("cost", {})
        totals["cost_usd"] += cost.get("total", 0.0)
    return totals


def extract_transcript_text(transcript: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        message = entry.get("message", {})
        role = message.get("role", "")
        content = message.get("content", "")
        if not content:
            continue
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text and role == "assistant":
                        parts.append(text)
        elif role == "assistant":
            parts.append(str(content))
    return "\n\n".join(parts).strip()


def last_assistant_message(transcript: list[dict[str, Any]]) -> str:
    for entry in reversed(transcript):
        if entry.get("type") != "message":
            continue
        message = entry.get("message", {})
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            blocks = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "") or block.get("content", "")
                    if text:
                        blocks.append(text)
            if blocks:
                return "\n".join(blocks).strip()
        elif content:
            return str(content).strip()
    return ""


def _chunk_message(message: str) -> list[str]:
    if len(message) <= MAX_OPENCLAW_MESSAGE_CHARS:
        return [message]
    chunks = [
        message[index : index + MAX_OPENCLAW_MESSAGE_CHARS]
        for index in range(0, len(message), MAX_OPENCLAW_MESSAGE_CHARS)
    ]
    total = len(chunks)
    prepared: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        if index == 1:
            prepared.append(
                f"You are receiving a long prompt in {total} parts. Ignore and do not respond "
                f"until the final part.\n\nPart {index}/{total}:\n{chunk}"
            )
        elif index < total:
            prepared.append(f"Part {index}/{total}:\n{chunk}")
        else:
            prepared.append(
                f"Part {index}/{total} (final):\n{chunk}\nAll parts received. Proceed now."
            )
    return prepared


def run_openclaw_prompt(
    *,
    agent_name: str,
    prompt: str,
    workspace: Path,
    session_id: str,
    runtime_args: list[str] | None = None,
    timeout_seconds: float = 3600.0,
    openclaw_path: str | None = None,
) -> dict[str, Any]:
    runtime_args = runtime_args or []
    workspace.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False

    for chunk in _chunk_message(prompt):
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            timed_out = True
            break
        send_chunk = (
            chunk.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
            if USE_SHELL
            else chunk
        )
        try:
            result = _run_command(
                [
                    "agent",
                    "--agent",
                    agent_name,
                    "--session-id",
                    session_id,
                    "--message",
                    send_chunk,
                    *runtime_args,
                ],
                cwd=str(workspace),
                timeout=remaining,
                openclaw_path=openclaw_path,
            )
            stdout += result.stdout
            stderr += result.stderr
            exit_code = result.returncode
            if result.returncode not in (0, -1):
                break
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += _coerce_subprocess_output(exc.stdout)
            stderr += _coerce_subprocess_output(exc.stderr)
            break

    transcript, transcript_path = load_transcript(agent_name, session_id, start_time)
    status = "success"
    if timed_out:
        status = "timeout"
    if not transcript:
        status = "error"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"

    return {
        "agent_name": agent_name,
        "status": status,
        "transcript": transcript,
        "transcript_path": str(transcript_path) if transcript_path else None,
        "usage": extract_usage_from_transcript(transcript),
        "response_text": last_assistant_message(transcript),
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": time.time() - start_time,
        "workspace": str(workspace),
    }


def render_insight_markdown(insights: dict[str, str]) -> str:
    lines = [
        "# Insight Library",
        "",
        "This file is generated by FoT from aggregated reasoning traces.",
        "",
    ]
    for name, description in insights.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(description.strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def coerce_boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise OpenClawError(f"Cannot parse boolean value: {value}")


def extract_problem_number(path: Path) -> int:
    match = re.search(r"problem_(\d+)\.json$", path.name)
    if not match:
        return 0
    return int(match.group(1))
