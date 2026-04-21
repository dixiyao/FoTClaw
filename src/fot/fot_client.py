from __future__ import annotations

import argparse
import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fotclaw.openclaw_adapter import run_openclaw_prompt


class LocalReasoningClient(ABC):
    """Abstract FoT local reasoning pipeline with overridable steps 1, 2, and 3."""

    def __init__(
        self,
        *,
        agent_name: str,
        workspace: str | Path,
        openclaw_path: str | None = None,
        output_dir: str = "output",
        timeout_seconds: float = 3600.0,
    ):
        self.agent_name = agent_name
        self.workspace = Path(workspace)
        self.openclaw_path = openclaw_path
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.reasoning_steps: list[dict[str, Any]] = []
        self.insight_book: dict[str, str] = {}

    def reset_state(self) -> None:
        self.reasoning_steps = []
        self.insight_book = {}

    @abstractmethod
    def local_step_1(
        self,
        problem: str,
        *,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        """Run local reasoning step 1 and return a step payload."""

    @abstractmethod
    def local_step_2(self, problem: str, step1_result: dict[str, Any]) -> dict[str, Any]:
        """Run local reasoning step 2 and return a step payload."""

    @abstractmethod
    def local_step_3(
        self,
        problem: str,
        step1_result: dict[str, Any],
        step2_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run local reasoning step 3 and return a step payload."""

    def build_existing_solution_step(self, *, problem: str, solution: str) -> dict[str, Any]:
        """Build a synthetic step 1 result when FoTClaw already has the solution transcript."""

        return {
            "step": 1,
            "name": "Existing Solution",
            "prompt": problem,
            "response": solution,
            "usage": {},
            "timestamp": time.time(),
            "source": "existing_solution",
        }

    def _record_step(
        self,
        result: dict[str, Any],
        *,
        step_number: int,
        default_name: str,
        require_response: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(result, dict):
            raise TypeError(f"FoT local step {step_number} must return a dict.")
        step = dict(result)
        step["step"] = step_number
        step.setdefault("name", default_name)
        step.setdefault("timestamp", time.time())
        if require_response and not isinstance(step.get("response"), str):
            raise ValueError(f"FoT local step {step_number} must provide a string `response`.")
        self.reasoning_steps.append(step)
        return step

    def _normalize_insight_book(self, step3_result: dict[str, Any]) -> dict[str, str]:
        raw = (
            step3_result.get("valid_skills")
            or step3_result.get("skills_extracted")
            or step3_result.get("insight_book")
            or {}
        )
        if not isinstance(raw, dict):
            return {}
        normalized: dict[str, str] = {}
        for name, description in raw.items():
            if not isinstance(description, str):
                continue
            cleaned_name = str(name)
            if not cleaned_name.startswith("insight_"):
                cleaned_name = f"insight_{cleaned_name}"
            cleaned_description = re.sub(r"\s+", " ", description).strip()
            if cleaned_description:
                normalized[cleaned_name] = cleaned_description
        return normalized

    def solve_problem(
        self,
        task: str,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        self.reset_state()
        step1 = self._record_step(
            self.local_step_1(
                task,
                custom_solution_instruction=custom_solution_instruction,
                insights_section=insights_section,
            ),
            step_number=1,
            default_name="Local Step 1",
        )
        step2 = self._record_step(
            self.local_step_2(task, step1),
            step_number=2,
            default_name="Local Step 2",
        )
        step3 = self._record_step(
            self.local_step_3(task, step1, step2),
            step_number=3,
            default_name="Local Step 3",
            require_response=False,
        )
        valid_skills = self._normalize_insight_book(step3)
        self.insight_book.update(valid_skills)
        return {
            "problem": task,
            "task": task,
            "solution": step1.get("response", ""),
            "reflection": step2.get("response", ""),
            "skills_extracted": valid_skills,
            "skills_used": list(valid_skills.keys()),
            "insight_book": self.insight_book,
            "total_steps": len(self.reasoning_steps),
            "usage": {
                "solution": step1.get("usage", {}),
                "reflection": step2.get("usage", {}),
                "insight_extraction": step3.get("usage", {}),
            },
        }

    def extract_from_trace(self, *, problem: str, solution: str) -> dict[str, Any]:
        self.reset_state()
        step1 = self.build_existing_solution_step(problem=problem, solution=solution)
        step2 = self._record_step(
            self.local_step_2(problem, step1),
            step_number=2,
            default_name="Local Step 2",
        )
        step3 = self._record_step(
            self.local_step_3(problem, step1, step2),
            step_number=3,
            default_name="Local Step 3",
            require_response=False,
        )
        valid_skills = self._normalize_insight_book(step3)
        self.insight_book.update(valid_skills)
        return {
            "problem": problem,
            "task": problem,
            "solution": solution,
            "reflection": step2.get("response", ""),
            "skills_extracted": valid_skills,
            "skills_used": list(valid_skills.keys()),
            "insight_book": self.insight_book,
            "total_steps": len(self.reasoning_steps),
            "usage": {
                "reflection": step2.get("usage", {}),
                "insight_extraction": step3.get("usage", {}),
            },
        }

    def save_reasoning(self, reasoning_result: dict[str, Any], output_path: str | None = None) -> str:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        insight_book = reasoning_result.get("insight_book", {})
        if not output_path:
            safe_name = re.sub(r"[^\w\s-]", "", reasoning_result.get("problem", "reasoning")[:50])
            safe_name = re.sub(r"[-\s]+", "_", safe_name).strip("_") or "reasoning"
            output_path = str(output_dir / f"{safe_name}.json")
        path = Path(output_path)
        if not path.is_absolute():
            path = output_dir / path
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.write_text(json.dumps(insight_book, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


class OpenClawFoTClient(LocalReasoningClient):
    """Default OpenClaw-backed implementation of the local FoT pipeline."""

    def _call_model(self, prompt: str, step_name: str) -> tuple[str, dict[str, Any]]:
        session_id = f"{step_name}_{int(time.time() * 1000)}"
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

    def _get_solution_prompt(
        self,
        problem: str,
        custom_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> str:
        insights_text = insights_section or ""
        custom_section = f"\n\n{custom_instruction}" if custom_instruction else ""
        return f"{insights_text}Problem: {problem}{custom_section}"

    def _get_reflection_prompt(self, problem: str, solution: str) -> str:
        return f"""
Analyze the solution below to extract procedural knowledge that reflect the reasoning traces.

Problem:
{problem}

Step-by-Step Solution:
{solution}

Your task: Extract the fundamental techniques used in resolution that can be packaged as reasoning traces. Focus on:

1. What step-by-step procedures were used? How can these be repeated?
2. What conditions determined which approach to use? When should each technique apply?
3. What methods, strategies, or workflows can be applied to similar problems?
4. What made this approach effective? What should someone know to use it correctly?
5. What types of problems would benefit from these techniques?

Output your analysis covering:

### I. Procedural Knowledge
- Break down the solution into clear, repeatable procedures
- The extracted traces should be concrete solutions rather than general principles.

### II. Reusable Techniques and Methods
- List specific techniques, strategies, or workflows used
- The techniques should be solid on practical questions rather than very general and high-level principles.
- For each technique, identify:
  * When it should be used (conditions/triggers)
  * How it was applied (concrete steps)
  * Why it was effective (insights)
  * What problems it could solve (applicability)

### III. Critical Insights and Guidelines
- What key insights made this solution work?
- What common pitfalls should be avoided?
- What variations or edge cases should be considered?

Focus on extracting actionable, procedural knowledge that can be packaged as reusable insights for similar problems.
""".strip()

    def _get_behavior_prompt(self, problem: str, solution: str, reflection: str) -> str:
        return """
Extract reasoning traces from the solution below. Analyze the solution and reflection to identify concrete, actionable traces that similar problems can be solved via the traces.

Problem: {problem}

Solution: {solution}

Reflection: {reflection}

Your Task:
Identify and extract all reusable reasoning traces, techniques, and methods used in the solution. Each trace should be a concrete procedure that can guide someone to solve similar problems.

What Makes a Good Reasoning Trace:
- A specific technique or method that was used in the solution.
- Something that can be applied to similar problems, not just this one.
- Includes guidance on when and how to use it with clear steps that can be followed if necessary.
- Not repetition of already well-known or commonly adopted techniques.
- Not too general and high-level but contains actionable procedural knowledge.

Description Must Include:
1. Core idea: The fundamental concept of what this trace is about.
2. When to use: Explain when this skill should be applied.

Output Format:
Return JSON only in the shape {{"trace_name": "description"}}.
- Use valid JSON.
- Every key must start with "insight_".
- Keep values as strings only.
""".format(problem=problem, solution=solution, reflection=reflection).strip()

    def local_step_1(
        self,
        problem: str,
        *,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        prompt = self._get_solution_prompt(problem, custom_solution_instruction, insights_section)
        response, usage = self._call_model(prompt, "solution")
        return {
            "name": "Solution Generation",
            "prompt": prompt,
            "response": response,
            "usage": usage,
        }

    def local_step_2(self, problem: str, step1_result: dict[str, Any]) -> dict[str, Any]:
        prompt = self._get_reflection_prompt(problem, str(step1_result.get("response", "")))
        response, usage = self._call_model(prompt, "reflection")
        return {
            "name": "Reflection",
            "prompt": prompt,
            "response": response,
            "usage": usage,
        }

    def local_step_3(
        self,
        problem: str,
        step1_result: dict[str, Any],
        step2_result: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._get_behavior_prompt(
            problem,
            str(step1_result.get("response", "")),
            str(step2_result.get("response", "")),
        )
        response, usage = self._call_model(prompt, "insights")
        skills = _extract_json_dict(response)
        valid_skills = {
            name if name.startswith("insight_") else f"insight_{name}": description.strip()
            for name, description in skills.items()
            if isinstance(description, str) and len(description.strip()) >= 20
        }
        return {
            "name": "Insight Extraction",
            "prompt": prompt,
            "response": response,
            "skills": skills,
            "valid_skills": valid_skills,
            "usage": usage,
        }


ChainOfThoughtReader = OpenClawFoTClient


def _extract_json_dict(text: str) -> dict[str, str]:
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
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            normalized[str(key)] = re.sub(r"\s+", " ", value).strip()
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenClaw-backed FoT client.")
    parser.add_argument("--agent", required=True, help="OpenClaw agent name to use")
    parser.add_argument("--workspace", required=True, help="Workspace path for the OpenClaw agent")
    parser.add_argument("--task", required=True, help="Problem or task prompt")
    parser.add_argument("--output", default="output", help="Output directory for reasoning trace JSON")
    parser.add_argument("--openclaw-path", default=None, help="Path to the openclaw binary")
    args = parser.parse_args()

    reader = OpenClawFoTClient(
        agent_name=args.agent,
        workspace=args.workspace,
        openclaw_path=args.openclaw_path,
        output_dir=args.output,
    )
    result = reader.solve_problem(args.task)
    path = reader.save_reasoning(result)
    print(json.dumps({"output_path": path, "insights": result["insight_book"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
