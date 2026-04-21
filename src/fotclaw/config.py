from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATIC_DEFAULT_SETTINGS: dict[str, Any] = {
    "openclaw_path": "openclaw",
    "local_reasoning_class": "fot.fot_client:OpenClawFoTClient",
    "global_reasoning_class": "fot.fot_server:OpenClawFoTServer",
    "auto_aggregate_enabled": True,
    "auto_aggregate_trace_threshold": 25,
    "auto_aggregate_min_interval_seconds": 0,
}

DEFAULT_RUNTIME_STATE: dict[str, Any] = {
    "last_aggregate_at": None,
    "last_aggregate_model": None,
    "last_aggregated_trace_count": 0,
}

def default_model_id() -> str:
    value = os.environ.get("FOT_DEFAULT_MODEL") or os.environ.get("FOTCLAW_DEFAULT_MODEL")
    if value and value.strip():
        return value.strip()
    return "google/gemini-3.1-pro-preview"


def default_settings() -> dict[str, Any]:
    model_id = default_model_id()
    settings = dict(STATIC_DEFAULT_SETTINGS)
    settings["default_model"] = model_id
    settings["aggregation_model"] = model_id
    openclaw_path = os.environ.get("OPENCLAW_PATH")
    if openclaw_path and openclaw_path.strip():
        settings["openclaw_path"] = openclaw_path.strip()
    return settings


DEFAULT_SETTINGS = default_settings()
USER_SETTING_KEYS = set(default_settings().keys())
RUNTIME_STATE_KEYS = set(DEFAULT_RUNTIME_STATE.keys())


@dataclass(frozen=True, slots=True)
class Layout:
    home: Path
    agents_dir: Path
    traces_dir: Path
    legacy_traces_dir: Path
    aggregate_dir: Path
    config_file: Path
    insight_markdown: Path
    insight_json: Path
    project_root: Path
    setting_file: Path
    legacy_setting_file: Path


def default_home() -> Path:
    value = os.environ.get("FOT_HOME") or os.environ.get("FOTCLAW_HOME")
    if value:
        return Path(value).expanduser().resolve()
    return (project_root() / ".fot").resolve()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_layout(home: str | Path | None = None) -> Layout:
    root = Path(home).expanduser().resolve() if home else default_home()
    repo_root = project_root()
    return Layout(
        home=root,
        agents_dir=root / "agents",
        traces_dir=root / "reasoning_traces",
        legacy_traces_dir=root / "traces",
        aggregate_dir=root / "aggregate",
        config_file=root / "config.json",
        insight_markdown=root / "insight.md",
        insight_json=root / "insight.json",
        project_root=repo_root,
        setting_file=repo_root / "setting.yaml",
        legacy_setting_file=root / "setting.yaml",
    )


def ensure_layout(home: str | Path | None = None) -> Layout:
    layout = build_layout(home)
    layout.home.mkdir(parents=True, exist_ok=True)
    layout.agents_dir.mkdir(parents=True, exist_ok=True)
    layout.traces_dir.mkdir(parents=True, exist_ok=True)
    if layout.legacy_traces_dir.exists() and layout.legacy_traces_dir != layout.traces_dir:
        for old_trace in sorted(layout.legacy_traces_dir.glob("problem_*.json")):
            target = layout.traces_dir / old_trace.name
            if not target.exists():
                old_trace.replace(target)
    layout.aggregate_dir.mkdir(parents=True, exist_ok=True)
    if not layout.config_file.exists():
        save_json(layout.config_file, DEFAULT_RUNTIME_STATE)
    if not layout.setting_file.exists():
        if layout.legacy_setting_file.exists():
            atomic_write_text(
                layout.setting_file,
                layout.legacy_setting_file.read_text(encoding="utf-8"),
            )
        else:
            save_settings(layout, default_settings())
    return layout


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
    ) as handle:
        handle.write(content)
        temp_name = handle.name
    Path(temp_name).replace(path)


def save_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def _coerce_yaml_scalar(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        data[key.strip()] = _coerce_yaml_scalar(raw_value)
    return data


def _to_yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in '#:"\'{}[]!,&*?|-<>=%@`'):
        escaped = text.replace('"', '\\"')
        return f'"{escaped}"'
    return text


def render_settings_yaml(settings: dict[str, Any]) -> str:
    defaults = default_settings()
    ordered_keys = [
        "default_model",
        "aggregation_model",
        "openclaw_path",
        "local_reasoning_class",
        "global_reasoning_class",
        "auto_aggregate_enabled",
        "auto_aggregate_trace_threshold",
        "auto_aggregate_min_interval_seconds",
    ]
    comments = {
        "default_model": "Default model for new FoT agents when `--model` is omitted.",
        "aggregation_model": "Model used for FoT reflection/aggregation. Keep this aligned with your OpenClaw setup.",
        "openclaw_path": "Path to the OpenClaw CLI binary.",
        "local_reasoning_class": "Python class path for the FoT local pipeline, for example package.module:CustomLocalReasoner.",
        "global_reasoning_class": "Python class path for the FoT global aggregation pipeline, for example package.module:CustomGlobalReasoner.",
        "auto_aggregate_enabled": "Turn automatic aggregation on or off.",
        "auto_aggregate_trace_threshold": "Auto aggregate once at least this many reasoning traces exist.",
        "auto_aggregate_min_interval_seconds": "Minimum delay between automatic aggregation runs.",
    }
    lines = ["# FoT editable settings", "# Edit this file directly.", ""]
    for key in ordered_keys:
        lines.append(f"# {comments[key]}")
        lines.append(f"{key}: {_to_yaml_scalar(settings.get(key, defaults[key]))}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_settings(layout: Layout) -> dict[str, Any]:
    defaults = default_settings()
    if not layout.setting_file.exists():
        return defaults
    raw = _parse_simple_yaml(layout.setting_file.read_text(encoding="utf-8"))
    merged = dict(defaults)
    for key in USER_SETTING_KEYS:
        if key in raw:
            merged[key] = raw[key]
    return merged


def save_settings(layout: Layout, settings: dict[str, Any]) -> None:
    merged = default_settings()
    for key in USER_SETTING_KEYS:
        if key in settings:
            merged[key] = settings[key]
    atomic_write_text(layout.setting_file, render_settings_yaml(merged))


def load_runtime_state(layout: Layout) -> dict[str, Any]:
    data = read_json(layout.config_file, {}) or {}
    merged = dict(DEFAULT_RUNTIME_STATE)
    for key in RUNTIME_STATE_KEYS:
        if key in data:
            merged[key] = data[key]
    return merged


def save_runtime_state(layout: Layout, runtime_state: dict[str, Any]) -> None:
    merged = dict(DEFAULT_RUNTIME_STATE)
    for key in RUNTIME_STATE_KEYS:
        if key in runtime_state:
            merged[key] = runtime_state[key]
    save_json(layout.config_file, merged)


def load_config(layout: Layout) -> dict[str, Any]:
    merged = default_settings()
    merged.update(load_settings(layout))
    merged.update(load_runtime_state(layout))
    return merged


def save_config(layout: Layout, config: dict[str, Any]) -> None:
    save_settings(layout, config)
    save_runtime_state(layout, config)
