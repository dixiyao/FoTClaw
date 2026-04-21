from __future__ import annotations

import re
import time
from pathlib import Path

from fotclaw.config import Layout, read_json, save_json
from fotclaw.models import AgentRecord


def agent_dir(layout: Layout, agent_id: str) -> Path:
    return layout.agents_dir / agent_id


def agent_record_path(layout: Layout, agent_id: str) -> Path:
    return agent_dir(layout, agent_id) / "record.json"


def load_agent(layout: Layout, agent_id: str) -> AgentRecord | None:
    payload = read_json(agent_record_path(layout, agent_id))
    if not isinstance(payload, dict):
        return None
    return AgentRecord.from_dict(payload)


def find_agent_by_name(layout: Layout, name: str) -> AgentRecord | None:
    agent_id = f"agt-{name}"
    return load_agent(layout, agent_id)


def save_agent(layout: Layout, agent: AgentRecord) -> None:
    if not agent.created_at:
        agent.created_at = time.time()
    agent.updated_at = time.time()
    target = agent_record_path(layout, agent.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(target, agent.to_dict())


def list_agents(layout: Layout) -> list[AgentRecord]:
    agents: list[AgentRecord] = []
    for record_path in sorted(layout.agents_dir.glob("*/record.json")):
        payload = read_json(record_path)
        if isinstance(payload, dict):
            agents.append(AgentRecord.from_dict(payload))
    agents.sort(key=lambda item: item.created_at, reverse=True)
    return agents


def remove_agent(layout: Layout, agent_id: str) -> None:
    root = agent_dir(layout, agent_id)
    if not root.exists():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            path.rmdir()
    root.rmdir()


def next_trace_index(layout: Layout) -> int:
    highest = 0
    pattern = re.compile(r"problem_(\d+)\.json$")
    for candidate in layout.traces_dir.glob("problem_*.json"):
        match = pattern.search(candidate.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1
