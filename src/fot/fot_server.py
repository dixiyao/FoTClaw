from __future__ import annotations

import argparse
import json
import math
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any

from fotclaw.openclaw_adapter import extract_problem_number, render_insight_markdown, run_openclaw_prompt


class GlobalReasoningServer(ABC):
    """Abstract FoT global aggregation pipeline with overridable steps 1, 2, and 3."""

    def __init__(
        self,
        *,
        agent_name: str,
        workspace: str | Path,
        openclaw_path: str | None = None,
        input_dirs: list[str] | None = None,
        num_insights: int | None = None,
        custom_prompt_section: str = "",
        timeout_seconds: float = 3600.0,
    ):
        self.agent_name = agent_name
        self.workspace = Path(workspace)
        self.openclaw_path = openclaw_path
        self.input_dirs = input_dirs or ["output"]
        self.num_insights = num_insights
        self.custom_prompt_section = custom_prompt_section.strip()
        self.timeout_seconds = timeout_seconds
        self.insight_store: dict[str, str] = {}
        self.insight_relationships: dict[str, Any] = {}
        self.aggregation_steps: list[dict[str, Any]] = []
        self.encyclopedia: str = ""
        self.encyclopedia_dict: dict[str, str] = {}

    def reset_state(self) -> None:
        self.insight_store = {}
        self.insight_relationships = {}
        self.aggregation_steps = []
        self.encyclopedia = ""
        self.encyclopedia_dict = {}

    @abstractmethod
    def global_step_1(self, json_files: list[str] | None = None) -> dict[str, Any]:
        """Run global aggregation step 1 and return a step payload."""

    @abstractmethod
    def global_step_2(self, collection_result: dict[str, Any]) -> dict[str, Any]:
        """Run global aggregation step 2 and return a step payload."""

    @abstractmethod
    def global_step_3(
        self,
        collection_result: dict[str, Any],
        profiling_result: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run global aggregation step 3 and return a step payload."""

    def _record_step(
        self,
        result: dict[str, Any],
        *,
        step_number: int,
        default_name: str,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError(f"FoT global step {step_number} must return a dict.")
        step = dict(result)
        step["step"] = step_number
        step.setdefault("name", default_name)
        step.setdefault("timestamp", time.time())
        self.aggregation_steps.append(step)
        return step

    def _normalize_str_dict(self, raw: Any) -> dict[str, str]:
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, str):
                normalized[str(key)] = re.sub(r"\s+", " ", value).strip()
        return normalized

    def _sync_collection_state(self, result: dict[str, Any]) -> None:
        self.insight_store = self._normalize_str_dict(result.get("insight_store"))

    def _sync_profiling_state(self, result: dict[str, Any]) -> None:
        profiling = result.get("profiling")
        if isinstance(profiling, dict):
            self.insight_relationships = profiling

    def _sync_extraction_state(self, result: dict[str, Any]) -> None:
        encyclopedia_dict = result.get("encyclopedia_dict")
        if encyclopedia_dict is None and isinstance(result.get("encyclopedia"), dict):
            encyclopedia_dict = result.get("encyclopedia")
        normalized = self._normalize_str_dict(encyclopedia_dict)
        self.encyclopedia_dict = normalized
        self.encyclopedia = json.dumps(normalized, indent=2, ensure_ascii=False)

    def collect_insight_books(self, json_files: list[str] | None = None) -> dict[str, Any]:
        collection = self._record_step(
            self.global_step_1(json_files),
            step_number=1,
            default_name="Global Step 1",
        )
        self._sync_collection_state(collection)
        collection["insight_store"] = dict(self.insight_store)
        return collection

    def aggregate_and_build_encyclopedia(
        self,
        json_files: list[str] | None = None,
        output_dir: str = "output",
        existing_encyclopedia_path: str | None = None,
    ) -> dict[str, Any]:
        self.reset_state()
        collection = self.collect_insight_books(json_files)
        if not self.insight_store:
            return {
                "collection": collection,
                "aggregation_steps": self.aggregation_steps,
                "encyclopedia": "",
                "encyclopedia_dict": {},
                "insight_store": {},
            }

        profiling = self._record_step(
            self.global_step_2(collection),
            step_number=2,
            default_name="Global Step 2",
        )
        self._sync_profiling_state(profiling)
        profiling["profiling"] = self.insight_relationships

        existing_path = Path(existing_encyclopedia_path) if existing_encyclopedia_path else (Path(output_dir) / "insight.json")
        existing = _load_existing_encyclopedia(existing_path)

        extraction = self._record_step(
            self.global_step_3(collection, profiling, existing),
            step_number=3,
            default_name="Global Step 3",
        )
        self._sync_extraction_state(extraction)
        extraction["encyclopedia"] = self.encyclopedia
        extraction["encyclopedia_dict"] = dict(self.encyclopedia_dict)

        return {
            "collection": collection,
            "profiling": profiling,
            "extraction": extraction,
            "aggregation_steps": self.aggregation_steps,
            "encyclopedia": self.encyclopedia,
            "encyclopedia_dict": dict(self.encyclopedia_dict),
            "insight_store": dict(self.insight_store),
            "insight_relationships": self.insight_relationships,
        }

    def save_results(self, result: dict[str, Any], output_dir: str = "output") -> tuple[str, str]:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        json_path = directory / "insight.json"
        markdown_path = directory / "insight.md"
        json_path.write_text(json.dumps(self.encyclopedia_dict, indent=2, ensure_ascii=False), encoding="utf-8")
        markdown_path.write_text(render_insight_markdown(self.encyclopedia_dict), encoding="utf-8")
        return str(json_path), str(markdown_path)


class OpenClawFoTServer(GlobalReasoningServer):
    """Default OpenClaw-backed implementation of the global FoT pipeline."""

    def _call_model(self, prompt: str, step_name: str) -> tuple[str, dict[str, Any]]:
        session_id = f"{step_name}_{int(time.time() * 1000)}"
        if self.custom_prompt_section:
            prompt = f"{self.custom_prompt_section}\n\n{prompt}"
        result = run_openclaw_prompt(
            agent_name=self.agent_name,
            prompt=prompt,
            workspace=self.workspace,
            session_id=session_id,
            timeout_seconds=self.timeout_seconds,
            openclaw_path=self.openclaw_path,
        )
        if result["status"] != "success":
            raise RuntimeError(f"{step_name} failed: {result['stderr'] or result['status']}")
        return result["response_text"], result["usage"]

    def global_step_1(self, json_files: list[str] | None = None) -> dict[str, Any]:
        if json_files is None:
            files: list[Path] = []
            for root in self.input_dirs:
                files.extend(Path(root).rglob("problem_*.json"))
            files = sorted(set(files), key=extract_problem_number)
        else:
            files = [Path(item) for item in json_files]

        all_insights: dict[str, str] = {}
        insight_counter = 0
        files_processed = 0

        for path in files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, dict):
                continue
            insight_book = payload.get("insight_book") or payload.get("behavior_book") or {}
            if not isinstance(insight_book, dict):
                continue
            file_added = 0
            for key, value in insight_book.items():
                if key in {"paper_name", "problem", "problem_id", "iteration", "is_correct"}:
                    continue
                if not isinstance(value, str) or not value.strip():
                    continue
                insight_counter += 1
                file_added += 1
                all_insights[f"{key}_{insight_counter:06d}"] = value.strip()
            if file_added:
                files_processed += 1

        return {
            "name": "Collect Insights",
            "files_processed": files_processed,
            "total_insights_collected": insight_counter,
            "insight_store": all_insights,
        }

    def _get_text_profiling_prompt(self, insight_store: dict[str, str]) -> str:
        insights_text = "\n".join(f"- {name}: {desc}" for name, desc in insight_store.items())
        return f"""
You are analyzing a collection of reasoning traces generated for problem-solving. Understand their relationships and structure and build a profiling of their relationships.

Collected {len(insight_store)} traces:
{insights_text}

Return JSON only with this shape:
{{
  "clusters": [
    {{
      "cluster_id": 0,
      "cluster_name": "Domain/Theme Name",
      "traces": ["trace_name"],
      "theme": "High-level theme"
    }}
  ],
  "relationships": [
    {{
      "trace_a": "trace_name1",
      "trace_b": "trace_name2",
      "relationship_type": "prerequisite/complementary/alternative/similar/derived_from/composes_with",
      "description": "How these traces relate"
    }}
  ]
}}
""".strip()

    def global_step_2(self, collection_result: dict[str, Any]) -> dict[str, Any]:
        insight_store = self._normalize_str_dict(collection_result.get("insight_store"))
        prompt = self._get_text_profiling_prompt(insight_store)
        response, usage = self._call_model(prompt, "profiling")
        profiling = _extract_json_object(response) or {"clusters": [], "relationships": []}
        return {
            "name": "Text-Based Profiling",
            "prompt": prompt,
            "response": response,
            "profiling": profiling,
            "usage": usage,
        }

    def _get_knowledge_extraction_prompt(
        self,
        insight_store: dict[str, str],
        profiling: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> str:
        clusters_text = "\n".join(
            f"- Cluster {cluster.get('cluster_id', '?')} ({cluster.get('cluster_name', 'unnamed')}): "
            f"{', '.join(cluster.get('traces', []))}"
            for cluster in profiling.get("clusters", [])
            if isinstance(cluster, dict)
        )
        relationships_text = json.dumps(profiling.get("relationships", []), indent=2, ensure_ascii=False)
        all_insights_text = "; ".join(f"{name}: {desc}" for name, desc in insight_store.items())
        current_text = json.dumps(existing_encyclopedia or {}, ensure_ascii=False)
        proper_number = (
            str(self.num_insights)
            if self.num_insights is not None
            else str(int(math.log10(max(10, len(insight_store))) * 10 + 1))
        )
        return f"""
Return exactly one valid JSON object and nothing else.

Required JSON shape:
{{
  "insight_name1": "description string",
  "insight_name2": "description string"
}}

Rules:
- Every key must start with "insight_".
- Every value must be a single string.
- No markdown fences.
- No explanations outside JSON.

Build a fundamental but still actionable insight library by combining:
- Existing encyclopedia: {current_text}
- Client reasoning traces: {all_insights_text}
- Trace clusters: {clusters_text if clusters_text else "None"}
- Trace relationships: {relationships_text}

Target about {proper_number} insights. Do not over-merge unrelated insights.
Each description should explain what the insight is, why it works, and when to use it.
""".strip()

    def global_step_3(
        self,
        collection_result: dict[str, Any],
        profiling_result: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        insight_store = self._normalize_str_dict(collection_result.get("insight_store"))
        profiling = profiling_result.get("profiling")
        if not isinstance(profiling, dict):
            profiling = {"clusters": [], "relationships": []}
        prompt = self._get_knowledge_extraction_prompt(insight_store, profiling, existing_encyclopedia)
        response, usage = self._call_model(prompt, "aggregate")
        encyclopedia = _extract_json_object(response)
        if encyclopedia is None:
            raise ValueError("Could not parse aggregated encyclopedia JSON.")
        return {
            "name": "Knowledge Extraction",
            "prompt": prompt,
            "response": response,
            "encyclopedia": json.dumps(encyclopedia, indent=2, ensure_ascii=False),
            "encyclopedia_dict": encyclopedia,
            "usage": usage,
        }


TextBasedInsightAggregationServer = OpenClawFoTServer


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if code_block:
        candidate = code_block.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start : end + 1]
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            normalized[str(key)] = re.sub(r"\s+", " ", value).strip()
    return normalized


def _load_existing_encyclopedia(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): str(value) for key, value in payload.items() if isinstance(value, str)}


def choose_most_common_model(models: list[str]) -> str | None:
    filtered = [model for model in models if model]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenClaw-backed FoT aggregation server.")
    parser.add_argument("--agent", required=True, help="OpenClaw agent name to use")
    parser.add_argument("--workspace", required=True, help="Workspace path for the OpenClaw agent")
    parser.add_argument("--input-dir", nargs="+", required=True, help="Input directories containing reasoning trace JSON")
    parser.add_argument("--output-dir", required=True, help="Output directory for the aggregated insight library")
    parser.add_argument("--openclaw-path", default=None, help="Path to the openclaw binary")
    parser.add_argument("--num-insights", type=int, default=None, help="Optional explicit target number of insights")
    args = parser.parse_args()

    server = OpenClawFoTServer(
        agent_name=args.agent,
        workspace=args.workspace,
        openclaw_path=args.openclaw_path,
        input_dirs=args.input_dir,
        num_insights=args.num_insights,
    )
    result = server.aggregate_and_build_encyclopedia(output_dir=args.output_dir)
    json_path, markdown_path = server.save_results(result, output_dir=args.output_dir)
    print(json.dumps({"insight_json": json_path, "insight_markdown": markdown_path}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
