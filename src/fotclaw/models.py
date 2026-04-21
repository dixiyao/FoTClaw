from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentRecord:
    id: str
    openclaw_agent_name: str
    model: str
    original_message: str
    name: str | None = None
    runtime_args: list[str] = field(default_factory=list)
    raw_agent_args: list[str] = field(default_factory=list)
    status: str = "created"
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    session_id: str = ""
    workspace: str = ""
    augmented_message: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    transcript_path: str | None = None
    trace_path: str | None = None
    supervisor_pid: int | None = None
    supervisor_pgid: int | None = None
    postprocess_status: str = "idle"
    postprocess_error: str | None = None
    postprocess_pid: int | None = None
    postprocess_pgid: int | None = None
    postprocess_started_at: float | None = None
    postprocess_finished_at: float | None = None
    exit_code: int | None = None
    timed_out: bool = False
    stop_requested: bool = False
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    auto_aggregated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentRecord":
        return cls(**payload)
