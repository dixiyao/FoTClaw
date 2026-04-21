from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pydoc import pager
from datetime import datetime
from pathlib import Path
from typing import Any

from fotclaw import __version__
from fotclaw.config import ensure_layout
from fotclaw.manager import (
    clean_agents,
    create_background_agent,
    ensure_named_agent,
    list_agents,
    sanitize_agent_name,
    show_agent,
    show_aggregate_agent,
    start_background_aggregate,
    stop_agent,
)
from fotclaw.models import AgentRecord
from fotclaw.openclaw_adapter import OpenClawError


RESET = ""
BOLD = ""
DIM = ""
RED = ""
ORANGE = ""
GREEN = ""
CYAN = ""
GRAY = ""
YELLOW = ""
WHITE = ""
SILVER = ""


def supports_color() -> bool:
    return False


def style(text: str, *codes: str) -> str:
    if not supports_color():
        return text
    return "".join(codes) + text + RESET


def header(title: str, subtitle: str | None = None) -> str:
    lines = [f"FoTClaw {__version__}"]
    if subtitle:
        lines.append(subtitle)
    return "\n".join(lines)


def render_banner(subtitle: str | None = None) -> str:
    return header(subtitle=subtitle, title="FoTClaw")


def section(title: str) -> str:
    return style(title, BOLD, ORANGE)


def divider(width: int = 72) -> str:
    return style("─" * width, GRAY)


def kv_line(key: str, value: Any) -> str:
    return f"{style(key + ':', BOLD, CYAN)} {value}"


def terminal_width(default: int = 92) -> int:
    try:
        return max(72, min(120, shutil.get_terminal_size((default, 40)).columns))
    except OSError:
        return default


def strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def pad_visible(text: str, width: int) -> str:
    return text + " " * max(0, width - visible_len(text))


def wrap_text(text: str, width: int) -> list[str]:
    import textwrap

    lines: list[str] = []
    for part in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(part, width=width) or [""]
        lines.extend(wrapped)
    return lines


def panel(title: str, body_lines: list[str], accent: str = ORANGE, width: int | None = None) -> str:
    width = width or terminal_width()
    inner = width - 4
    top = f"┌─ {style(title, BOLD, accent)} " + style("─" * max(0, inner - len(title) - 1), GRAY) + "┐"
    rows = [top]
    for line in body_lines:
        for wrapped in wrap_text(line, inner):
            rows.append(f"│ {pad_visible(wrapped, inner)} │")
    rows.append(f"└{style('─' * (width - 2), GRAY)}┘")
    return "\n".join(rows)


def status_color(status: str) -> str:
    mapping = {
        "starting": CYAN,
        "running": YELLOW,
        "finished": GREEN,
        "broken": RED,
        "stopped": GRAY,
        "created": CYAN,
    }
    return mapping.get(status, CYAN)


def format_status(status: str) -> str:
    return style(status, BOLD, status_color(status))


def format_timestamp(value: float | None) -> str:
    if not value:
        return "-"
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _help_panel(
    title: str,
    usage_lines: list[str],
    description_lines: list[str],
    *,
    examples: list[str] | None = None,
    accent: str = ORANGE,
    width: int | None = None,
) -> str:
    body = ["Usage:"]
    body.extend(usage_lines)
    if description_lines:
        body.extend(["", "What it does:"])
        body.extend(description_lines)
    if examples:
        body.extend(["", "Examples:"])
        body.extend(examples)
    return panel(title, body, accent=accent, width=width)


def render_root_help() -> str:
    width = terminal_width()
    lines = [
        render_banner("persistent multi-agent orchestration for OpenClaw with FoT aggregation"),
        panel(
            "FoTClaw setup",
            [
                "FoTClaw is a persistent OpenClaw wrapper for background runs, trace extraction, and insight aggregation.",
                "It stores editable settings in `setting.yaml` at the project root and runtime state under `.fotclaw/` in the project by default.",
            ],
            width=width,
        ),
        "",
        panel(
            "Global Options",
            [
                "--home PATH  Override the FoTClaw state directory.",
                "-h, --help   Show help for FoTClaw or for the selected command.",
            ],
            accent=CYAN,
            width=width,
        ),
        "",
        panel(
            "Commands",
            [
                "fotclaw agent        Start a background agent or reuse a named one.",
                "fotclaw list         List all FoTClaw agents.",
                "fotclaw show agent   Inspect one agent or the aggregate agent.",
                "fotclaw stop         Stop a running agent.",
                "fotclaw delete agent Delete an agent by name or id.",
                "fotclaw aggregate    Start a background aggregation job.",
                "fotclaw clean        Remove FoTClaw-managed state while keeping insight files.",
            ],
            width=width,
        ),
        "",
        _help_panel(
            "fotclaw agent",
            [
                "fotclaw agent --message \"Solve the task in the current workspace.\"",
                "fotclaw agent --name math",
                "fotclaw agent --name math --message \"Analyze the repo.\"",
            ],
            [
                "Starts a background agent.",
                "If you pass --name without an OpenClaw message, FoTClaw creates or reopens a stable named shell.",
                "If you pass OpenClaw arguments, FoTClaw runs the task in the background.",
            ],
            accent=GREEN,
            width=width,
        ),
        "",
        _help_panel(
            "fotclaw show agent",
            [
                "fotclaw show agent --name math",
                "fotclaw show agent --id agt-math",
                "fotclaw show agent agt-math",
            ],
            [
                "Shows the current state, output, and trace summary for one agent.",
                "Use the name `aggregate` to inspect the aggregation worker and `insight.md`.",
            ],
            accent=CYAN,
            width=width,
        ),
        "",
        _help_panel(
            "fotclaw aggregate",
            ["fotclaw aggregate"],
            [
                "Starts a background aggregation job that merges reasoning traces into the shared insight library.",
            ],
            examples=["fotclaw show agent --name aggregate"],
            accent=ORANGE,
            width=width,
        ),
        "",
        panel(
            "More Help",
            [
                "Run `fotclaw agent --help` for command-specific help.",
                "Run `fotclaw show agent --help` for focused inspection usage.",
                "Edit `setting.yaml` directly to change models and FoT algorithm classes.",
            ],
            accent=SILVER,
            width=width,
        ),
    ]
    return "\n".join(lines)


def render_agent_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("agent command"),
            _help_panel(
                "fotclaw agent",
                [
                    "fotclaw agent --message \"...\" [--model MODEL] [other openclaw args]",
                    "fotclaw agent --name NAME",
                    "fotclaw agent --name NAME --message \"...\" [--model MODEL] [other openclaw args]",
                ],
                [
                    "Starts a background agent or reuses a stable named agent.",
                    "OpenClaw-style flags such as `--message` and `--model` can be passed directly.",
                    "Use `fotclaw show agent ...` to inspect status and outputs later.",
                ],
                examples=[
                    "fotclaw agent --message \"Solve the task in the current workspace.\"",
                    "fotclaw agent --name math",
                    "fotclaw agent --name math --message \"Review the repository.\"",
                ],
                width=width,
            ),
        ]
    )

def render_list_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("list command"),
            _help_panel(
                "fotclaw list",
                ["fotclaw list"],
                ["Lists all FoTClaw-managed agents and their current status."],
                examples=["fotclaw list"],
                width=width,
            ),
        ]
    )


def render_show_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("show command"),
            _help_panel(
                "fotclaw show",
                ["fotclaw show agent --name NAME", "fotclaw show agent --id AGENT_ID", "fotclaw show agent AGENT_ID"],
                [
                    "Shows one agent.",
                    "Use `aggregate` as the name to inspect the aggregation worker.",
                ],
                examples=[
                    "fotclaw show agent --name math",
                    "fotclaw show agent agt-20260411-abcdef",
                    "fotclaw show agent --name aggregate",
                ],
                width=width,
            ),
        ]
    )


def render_show_agent_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("show agent command"),
            _help_panel(
                "fotclaw show agent",
                ["fotclaw show agent --name NAME", "fotclaw show agent --id AGENT_ID", "fotclaw show agent AGENT_ID"],
                [
                    "Shows one agent's status, output, transcript, and extracted trace summary.",
                    "Use `--name aggregate` to inspect the aggregation worker and shared insight library.",
                ],
                examples=[
                    "fotclaw show agent --name math",
                    "fotclaw show agent --id agt-math",
                    "fotclaw show agent --name aggregate",
                ],
                width=width,
            ),
        ]
    )


def render_stop_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("stop command"),
            _help_panel(
                "fotclaw stop",
                ["fotclaw stop AGENT_ID"],
                ["Stops a running FoTClaw agent."],
                examples=["fotclaw stop agt-math"],
                width=width,
            ),
        ]
    )


def render_delete_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("delete command"),
            _help_panel(
                "fotclaw delete",
                ["fotclaw delete agent --name NAME", "fotclaw delete agent --id AGENT_ID"],
                ["Deletes a FoTClaw-managed agent and its local state in the background."],
                examples=["fotclaw delete agent --name math", "fotclaw delete agent --id agt-math"],
                width=width,
            ),
        ]
    )


def render_delete_agent_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("delete agent command"),
            _help_panel(
                "fotclaw delete agent",
                ["fotclaw delete agent --name NAME", "fotclaw delete agent --id AGENT_ID"],
                [
                    "Deletes one agent by stable name or explicit id.",
                    "This removal runs in the background.",
                ],
                examples=["fotclaw delete agent --name math", "fotclaw delete agent --id agt-20260411-abcdef"],
                width=width,
            ),
        ]
    )


def render_aggregate_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("aggregate command"),
            _help_panel(
                "fotclaw aggregate",
                ["fotclaw aggregate"],
                [
                    "Starts a background aggregation job.",
                    "The aggregation worker reads stored reasoning traces and refreshes the shared insight library.",
                ],
                examples=["fotclaw aggregate", "fotclaw show agent --name aggregate"],
                width=width,
            ),
        ]
    )


def render_clean_help() -> str:
    width = terminal_width()
    return "\n".join(
        [
            render_banner("clean command"),
            _help_panel(
                "fotclaw clean",
                ["fotclaw clean"],
                [
                    "Removes FoTClaw-managed agent state, transient traces, and aggregation scratch files.",
                    "The persistent `insight.md` and `insight.json` files are kept.",
                ],
                examples=["fotclaw clean"],
                width=width,
            ),
        ]
    )

def render_help(command: str | None = None, subcommand: str | None = None) -> str:
    if command == "agent":
        return render_agent_help()
    if command == "list":
        return render_list_help()
    if command == "show" and subcommand == "agent":
        return render_show_agent_help()
    if command == "show":
        return render_show_help()
    if command == "stop":
        return render_stop_help()
    if command == "delete" and subcommand == "agent":
        return render_delete_agent_help()
    if command == "delete":
        return render_delete_help()
    if command == "aggregate":
        return render_aggregate_help()
    if command == "clean":
        return render_clean_help()
    return render_root_help()


def render_help_for_args(args: argparse.Namespace) -> str:
    command = getattr(args, "command", None)
    if command == "show":
        return render_help(command, getattr(args, "show_command", None))
    if command == "delete":
        return render_help(command, getattr(args, "delete_command", None))
    return render_help(command, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--home", default=None, help="FoTClaw state directory (default: ./.fotclaw or FOTCLAW_HOME)")
    parser.add_argument("-h", "--help", action="store_true")
    subparsers = parser.add_subparsers(dest="command")

    agent_parser = subparsers.add_parser("agent", add_help=False)
    agent_parser.add_argument("-h", "--help", action="store_true")
    agent_parser.add_argument("--name", dest="agent_name", default=None)
    agent_parser.add_argument("openclaw_args", nargs=argparse.REMAINDER)

    list_parser = subparsers.add_parser("list", add_help=False)
    list_parser.add_argument("-h", "--help", action="store_true")

    show_parser = subparsers.add_parser("show", add_help=False)
    show_parser.add_argument("-h", "--help", action="store_true")
    show_subparsers = show_parser.add_subparsers(dest="show_command")
    show_agent_parser = show_subparsers.add_parser("agent", add_help=False)
    show_agent_parser.add_argument("-h", "--help", action="store_true")
    show_agent_parser.add_argument("--name", dest="agent_name", default=None)
    show_agent_parser.add_argument("--id", dest="agent_id_flag", default=None)
    show_agent_parser.add_argument("agent_id", nargs="?")

    stop_parser = subparsers.add_parser("stop", add_help=False)
    stop_parser.add_argument("-h", "--help", action="store_true")
    stop_parser.add_argument("agent_id", nargs="?")

    delete_parser = subparsers.add_parser("delete", add_help=False)
    delete_parser.add_argument("-h", "--help", action="store_true")
    delete_subparsers = delete_parser.add_subparsers(dest="delete_command")
    delete_agent_parser = delete_subparsers.add_parser("agent", add_help=False)
    delete_agent_parser.add_argument("-h", "--help", action="store_true")
    delete_agent_parser.add_argument("--name", dest="agent_name", default=None)
    delete_agent_parser.add_argument("--id", dest="agent_id", default=None)

    aggregate_parser = subparsers.add_parser("aggregate", add_help=False)
    aggregate_parser.add_argument("-h", "--help", action="store_true")

    clean_parser = subparsers.add_parser("clean", add_help=False)
    clean_parser.add_argument("-h", "--help", action="store_true")

    return parser


def print_text(text: str = "") -> None:
    sys.stdout.write(text + ("\n" if not text.endswith("\n") else ""))


def print_error(message: str) -> None:
    text = panel("Error", [message], accent=RED)
    sys.stderr.write(text + "\n")


def emit(text: str, *, paged: bool = False) -> None:
    if paged and sys.stdout.isatty():
        pager(text)
        if not text.endswith("\n"):
            print()
        return
    print_text(text)


def render_agent_list(records: list[AgentRecord]) -> str:
    width = terminal_width()
    lines = [render_banner("agent registry")]
    if not records:
        lines.append(panel("Agents", ["No FoTClaw agents found."], accent=GRAY, width=width))
        return "\n".join(lines)

    for record in records:
        body = [
            f"id: {record.id}",
            f"name: {record.name or '-'}",
            f"status: {strip_ansi(format_status(record.status))}",
            f"postprocess: {record.postprocess_status}",
            f"model: {record.model}",
            f"created: {format_timestamp(record.created_at)}",
        ]
        if record.trace_path:
            body.append(f"trace: {record.trace_path}")
        if record.error:
            body.append(f"error: {record.error}")
        lines.append(panel(record.id, body, accent=status_color(record.status), width=width))
        lines.append("")
    return "\n".join(lines)


def render_show(payload: dict[str, Any]) -> str:
    width = terminal_width()
    record = payload["record"]
    lines = [
        render_banner("agent inspection"),
        panel(
            "Agent",
            [
                f"id: {record.id}",
                f"name: {record.name or '-'}",
                f"status: {strip_ansi(format_status(record.status))}",
                f"postprocess: {record.postprocess_status}",
                f"model: {record.model}",
                f"created_at: {format_timestamp(record.created_at)}",
                f"started_at: {format_timestamp(record.started_at)}",
                f"finished_at: {format_timestamp(record.finished_at)}",
                f"postprocess_started_at: {format_timestamp(record.postprocess_started_at)}",
                f"postprocess_finished_at: {format_timestamp(record.postprocess_finished_at)}",
                f"workspace: {record.workspace or '-'}",
                f"transcript_path: {record.transcript_path or '-'}",
                f"trace_path: {record.trace_path or '-'}",
            ],
            accent=status_color(record.status),
            width=width,
        ),
    ]
    if record.error:
        lines.extend(["", panel("Error", [record.error], accent=RED, width=width)])
    if record.postprocess_error:
        lines.extend(["", panel("Postprocess Error", [record.postprocess_error], accent=RED, width=width)])

    trace = payload.get("trace")
    if trace:
        lines.extend(
            [
                "",
                panel(
                    "Trace Summary",
                    [
                        f"problem: {trace.get('problem', '-')}",
                        f"insights: {len(trace.get('insight_book', {}) or {})}",
                    ],
                    accent=GREEN,
                    width=width,
                ),
            ]
        )

    if payload.get("stderr"):
        lines.extend(["", panel("Stderr", payload["stderr"][:4000].splitlines() or [""], accent=RED, width=width)])
    if payload.get("stdout"):
        lines.extend(["", panel("Stdout", payload["stdout"][:4000].splitlines() or [""], accent=SILVER, width=width)])
    if payload.get("transcript"):
        lines.extend(["", panel("Transcript", payload["transcript"][:6000].splitlines() or [""], accent=CYAN, width=width)])
    return "\n".join(lines)


def render_show_compact(payload: dict[str, Any]) -> str | None:
    record = payload["record"]
    raw_stdout = payload.get("stdout") or ""
    cleaned_lines = [
        line
        for line in raw_stdout.splitlines()
        if not line.startswith("[agents/auth-profiles]")
    ]
    stdout = "\n".join(cleaned_lines).strip()
    stderr = (payload.get("stderr") or "").strip()

    if record.status in {"starting", "running"}:
        return f"{record.id} is working on it."
    if record.status == "finished":
        if stdout:
            return stdout
        return f"{record.id} finished."
    if record.status == "broken":
        if record.error:
            return record.error
        if record.postprocess_error:
            return record.postprocess_error
        if stderr:
            return stderr
        return f"{record.id} failed."
    if record.status == "stopped":
        return f"{record.id} was stopped."
    return None


def render_aggregate_show(payload: dict[str, Any]) -> str:
    record = payload["record"]
    lines = [
        f"aggregate status: {record.status}",
        f"last_aggregate_at: {payload.get('last_aggregate_at') or '-'}",
        f"insight_markdown: {payload.get('insight_markdown_path') or '-'}",
    ]
    insight_text = (payload.get("insight_markdown") or "").strip()
    if insight_text:
        lines.extend(["", insight_text])
    elif record.status in {"starting", "running"}:
        lines.extend(["", "Aggregate agent is working on the insight library."])
    elif record.status == "broken":
        error_text = (payload.get("stderr") or record.error or "Aggregate agent failed.").strip()
        lines.extend(["", error_text])
    return "\n".join(lines)

def render_success(title: str, body: list[str]) -> str:
    width = terminal_width()
    lines = [render_banner(title), panel(title.title(), body, accent=GREEN, width=width)]
    return "\n".join(lines)


def normalize_agent_args(raw_args: list[str]) -> list[str]:
    if raw_args and raw_args[0] == "--":
        return raw_args[1:]
    return raw_args


def preprocess_cli_argv(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        return None
    args = list(argv)
    if not args:
        return args

    def _find_command_index(tokens: list[str]) -> int | None:
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "--home":
                index += 2
                continue
            if token.startswith("--home="):
                index += 1
                continue
            if token in {"-h", "--help"}:
                index += 1
                continue
            return index
        return None

    def _rewrite_agent_args(tokens: list[str], start_index: int, known_options: set[str]) -> list[str]:
        if "--" in tokens[start_index:]:
            return tokens
        index = start_index
        while index < len(tokens):
            token = tokens[index]
            if token in {"-h", "--help"}:
                return tokens
            if token == "--name":
                index += 2
                continue
            if token.startswith("--name="):
                index += 1
                continue
            if token.startswith("--") and token not in known_options:
                return tokens[:index] + ["--"] + tokens[index:]
            index += 1
        return tokens

    command_index = _find_command_index(args)
    if command_index is None:
        return args

    if args[command_index] == "agent":
        return _rewrite_agent_args(args, command_index + 1, {"--name", "-h", "--help"})

    return args


def require(value: Any, message: str) -> Any:
    if value in (None, ""):
        raise OpenClawError(message)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(preprocess_cli_argv(raw_argv))

    if getattr(args, "help", False):
        print_text(render_help_for_args(args))
        return 0

    if not args.command:
        print_text(render_help())
        return 0

    try:
        if args.command == "agent":
            raw_args = normalize_agent_args(args.openclaw_args)
            if args.agent_name:
                record, action = ensure_named_agent(args.home, args.agent_name, raw_args or None)
                if action in {"existing", "running"}:
                    emit(render_show(show_agent(args.home, record.id)), paged=True)
                    return 0
                title = {
                    "created": "named agent created",
                    "started": "named agent started",
                    "restarted": "named agent restarted",
                }.get(action, "named agent ready")
                emit(
                    render_success(
                        title,
                        [
                            f"agent_id: {record.id}",
                            f"name: {record.name or '-'}",
                            f"openclaw_agent: {record.openclaw_agent_name}",
                            f"status: {strip_ansi(format_status(record.status))}",
                            f"model: {record.model}",
                            f"workspace: {record.workspace}",
                        ],
                    ),
                    paged=action == "created",
                )
                return 0

            if not raw_args:
                raise OpenClawError("Use `fotclaw agent --message \"...\"` or `fotclaw agent --name <name>`.")
            record = create_background_agent(args.home, raw_args)
            emit(
                render_success(
                    "background agent created",
                    [
                        f"agent_id: {record.id}",
                        f"name: {record.name or '-'}",
                        f"openclaw_agent: {record.openclaw_agent_name}",
                        f"model: {record.model}",
                        f"workspace: {record.workspace}",
                    ],
                )
            )
            return 0

        if args.command == "list":
            emit(render_agent_list(list_agents(args.home)), paged=False)
            return 0

        if args.command == "show":
            require(getattr(args, "show_command", None), "Use `fotclaw show agent --name <name>` or `fotclaw show agent --id <agent_id>`.")
            explicit_id = getattr(args, "agent_id_flag", None) or getattr(args, "agent_id", None)
            if getattr(args, "agent_name", None):
                normalized_name = sanitize_agent_name(args.agent_name)
                if normalized_name == "aggregate":
                    emit(render_aggregate_show(show_aggregate_agent(args.home)), paged=False)
                    return 0
                agent_id = f"agt-{normalized_name}"
            else:
                agent_id = require(explicit_id, "Missing agent id.")
                if agent_id == "agt-aggregate":
                    emit(render_aggregate_show(show_aggregate_agent(args.home)), paged=False)
                    return 0
            payload = show_agent(args.home, agent_id)
            compact = render_show_compact(payload)
            if compact is not None:
                emit(compact, paged=False)
            else:
                emit(render_show(payload), paged=False)
            return 0

        if args.command == "stop":
            agent_id = require(getattr(args, "agent_id", None), "Missing agent id.")
            record = stop_agent(args.home, agent_id)
            emit(
                render_success(
                    "agent stop requested",
                    [f"agent_id: {record.id}", f"status: {strip_ansi(format_status(record.status))}"],
                )
            )
            return 0

        if args.command == "delete":
            require(getattr(args, "delete_command", None), "Use `fotclaw delete agent --name <name>`.")
            if getattr(args, "delete_command", None) != "agent":
                raise OpenClawError("Use `fotclaw delete agent --name <name>`.")
            if not getattr(args, "agent_name", None) and not getattr(args, "agent_id", None):
                raise OpenClawError("Provide `--name <name>` or `--id <agent_id>`.")
            layout = ensure_layout(args.home)
            log_path = layout.home / "delete.log"
            worker_env = dict(os.environ)
            src_root = Path(__file__).resolve().parents[1]
            existing_pythonpath = worker_env.get("PYTHONPATH")
            worker_env["PYTHONPATH"] = (
                f"{src_root}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(src_root)
            )
            worker_cmd = [
                sys.executable,
                "-m",
                "fotclaw.supervisor",
                "--home",
                str(layout.home),
            ]
            if getattr(args, "agent_name", None):
                worker_cmd.extend(["--delete-agent-name", str(args.agent_name)])
            if getattr(args, "agent_id", None):
                worker_cmd.extend(["--delete-agent-id", str(args.agent_id)])
            with log_path.open("a", encoding="utf-8") as handle:
                subprocess.Popen(
                    worker_cmd,
                    stdout=handle,
                    stderr=handle,
                    start_new_session=True,
                    cwd=str(layout.home),
                    env=worker_env,
                )
            emit(
                render_success(
                    "agent deletion started",
                    [
                        f"name: {getattr(args, 'agent_name', None) or '-'}",
                        f"agent_id: {getattr(args, 'agent_id', None) or '-'}",
                        "status: deleting in background",
                    ],
                )
            )
            return 0

        if args.command == "aggregate":
            record = start_background_aggregate(args.home)
            emit(
                render_success(
                    "aggregation started",
                    [
                        f"agent_id: {record.id}",
                        f"name: {record.name or '-'}",
                        f"openclaw_agent: {record.openclaw_agent_name}",
                        f"status: {strip_ansi(format_status(record.status))}",
                        f"model: {record.model}",
                    ],
                )
            )
            return 0

        if args.command == "clean":
            print_text(render_banner("cleaning FoTClaw state").rstrip())

            def _clean_progress(message: str) -> None:
                print_text(f"[clean] {message}")

            removed = clean_agents(args.home, reporter=_clean_progress)
            emit(
                render_success(
                    "clean complete",
                    [
                        f"removed_agent_ids: {', '.join(removed['removed_agent_ids']) if removed['removed_agent_ids'] else '-'}",
                        f"stopped_agent_ids: {', '.join(removed['stopped_agent_ids']) if removed['stopped_agent_ids'] else '-'}",
                        f"removed_openclaw_agents: {', '.join(removed['removed_openclaw_agents']) if removed['removed_openclaw_agents'] else '-'}",
                        f"cleared_traces: {removed['cleared_traces']}",
                        f"cleared_aggregate_workspace: {removed['cleared_aggregate_workspace']}",
                        f"preserved_insight_markdown: {removed['preserved_insight_markdown']}",
                        f"preserved_insight_json: {removed['preserved_insight_json']}",
                    ],
                )
            )
            return 0
    except OpenClawError as exc:
        print_error(str(exc))
        return 1

    print_text(render_help())
    return 1


def clean_main() -> int:
    return main(["clean"])


if __name__ == "__main__":
    raise SystemExit(main())
