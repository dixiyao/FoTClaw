<div align="center">

# FoT

[![arXiv](https://img.shields.io/badge/arXiv-2604.16778-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2604.16778)
[![Homepage](https://img.shields.io/badge/Homepage-Visit%20Site-2563eb?style=for-the-badge&logo=googlechrome&logoColor=white)](https://dixiyao.github.io/fot)

</div>

# Federation over Text
Federation over Text (FoT) is a federated-learning-like paradigm for multi-agent reasoning.

Instead of sharing gradients or model weights, agents share **reasoning traces** distilled from completed tasks. A central aggregation process then organizes and compresses those traces into reusable **insights** that can help future agents solve related problems more effectively.

There are strong connections between FL and FoT. In FL, clients may adopt different optimization methods to solve local subproblems. Analogously, in FoT, each agent may use distinct local reasoning strategies and prompt designs to generate traces.

Inspired by techniques from distributed and federated learning, FoT opens an interesting design space for improving the efficiency and effectiveness of multi-agent collaborative reasoning.

# FoTClaw
FoTClaw is an orchestration framework for **Federation over Text (FoT)** built on top of `openclaw`.

It lets you run multiple OpenClaw agents in parallel, recover their reasoning traces after execution, and aggregate those traces into a persistent shared insight library. The result is a practical testbed for studying how agents can improve collectively through text-based reasoning exchange rather than parameter sharing.

- `⚡` Run multiple OpenClaw agents concurrently under FoTClaw supervision.
- `🧠` Recover transcripts from finished or broken runs.
- `📝` Convert transcripts into structured local reasoning traces.
- `🔗` Aggregate traces into a persistent shared insight library.
- `📚` Inject the current insight library into new agent workspaces automatically.

## Table of Contents
- [FoT](#fot)
- [Federation over Text](#federation-over-text)
- [FoTClaw](#fotclaw)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Setup](#setup)
  - [Quick Start](#quick-start)
  - [Command Overview](#command-overview)
  - [How the FoT Pipeline Works](#how-the-fot-pipeline-works)
  - [Algorithm Interfaces](#algorithm-interfaces)
    - [Local Reasoning Interface](#local-reasoning-interface)
    - [Global Aggregation Interface](#global-aggregation-interface)
    - [Return Format](#return-format)
  - [Replacing the Default Algorithms](#replacing-the-default-algorithms)
    - [Minimal Example](#minimal-example)
  - [Configuration](#configuration)
  - [State and Artifacts](#state-and-artifacts)
  - [Repository Layout](#repository-layout)
  - [OpenClaw Integration Notes](#openclaw-integration-notes)
  - [Research Framing](#research-framing)
- [Citation](#citation)

## Architecture

The repository is split into two layers:

- `src/fot/`
  The FoT algorithm layer. This contains the local reasoning pipeline and the global aggregation pipeline.
- `src/fotclaw/`
  The orchestration layer. This contains the CLI, state management, OpenClaw integration, supervision, and persistence.

At a high level:

- `fotclaw` hosts and manages agents.
- `fot` defines how local reasoning traces are extracted and how global insights are aggregated.

## Installation

### Requirements

- Python `3.11+`
- OpenClaw installed separately and available as `openclaw`, or configured through `OPENCLAW_PATH`
- An OpenClaw setup that can run the model ids configured in the project root `setting.yaml`

### Setup

```bash
conda create -n fotclaw python=3.12 -y
conda activate fotclaw
python -m pip install --upgrade pip
python -m pip install -e .
```

Development extras:

```bash
python -m pip install -e ".[dev]"
```

## Quick Start

Start a background agent:

```bash
fotclaw agent --message "Solve the task in the current workspace."
```

Create or reuse a stable named agent shell:

```bash
fotclaw agent --name math
```

Run work on a named agent:

```bash
fotclaw agent --name math --message "Work on the math task."
```

Inspect agent state:

```bash
fotclaw show agent --name math
```

List all FoTClaw-managed agents:

```bash
fotclaw list
```

Start aggregation:

```bash
fotclaw aggregate
```

Inspect the aggregation worker and the shared insight library:

```bash
fotclaw show agent --name aggregate
```

For detailed command usage:

```bash
fotclaw --help
fotclaw agent --help
fotclaw show agent --help
```

## Command Overview

FoTClaw provides these main commands:

- `fotclaw agent`
- `fotclaw list`
- `fotclaw show agent`
- `fotclaw stop`
- `fotclaw delete agent`
- `fotclaw aggregate`
- `fotclaw clean`

The root CLI help now describes how each command is used, and command-specific help is available for the major subcommands.

## How the FoT Pipeline Works

When an agent finishes or breaks, FoTClaw separates execution from FoT postprocessing:

1. **Execution**
   OpenClaw runs the task and produces a transcript.
2. **Local FoT postprocessing**
   FoTClaw runs the local step over the task result or transcript and extracts reusable reasoning artifacts.
3. **Trace persistence**
   The extracted reasoning trace is stored under the FoTClaw state directory.
4. **Global aggregation**
   The server step aggregates trace files and rebuilds the shared insight library.

At the project level, FoTClaw exposes two algorithm hooks:

- Local step: transform one task result or transcript into reusable local reasoning artifacts and insights.
- Server step: merge many local insight artifacts into the shared global insight library.

Every new FoTClaw run copies the current shared library into the agent workspace as both `INSIGHTS.md` and `insight.md`, then prefixes the prompt so the agent is instructed to read and use it.

## Algorithm Interfaces

One of the main changes in the current codebase is that the FoT algorithm layer is now explicitly exposed through abstract interfaces.

Conceptually, users can think about FoTClaw as having:

- a `local step` interface for per-task reasoning and insight extraction
- a `server step` interface for cross-task aggregation

Internally, the default implementation breaks each interface into staged abstract methods, but users do not need to think in terms of "step 1, step 2, step 3" when understanding the project at a high level.

### Local Reasoning Interface

The abstract base class is:

- `fot.fot_client.LocalReasoningClient`

Users can replace the default local FoT pipeline by subclassing this abstract local reasoning client.

The default implementation is:

- `fot.fot_client.OpenClawFoTClient`

### Global Aggregation Interface

The abstract base class is:

- `fot.fot_server.GlobalReasoningServer`

Users can replace the default global FoT aggregation pipeline by subclassing this abstract server-side reasoning interface.

The default implementation is:

- `fot.fot_server.OpenClawFoTServer`

### Return Format

The local reasoning interface and the server-side aggregation interface should each behave like a complete wrapper over their own algorithm.

Each abstract method should return a Python `dict`, but users should think in terms of:

- a local reasoning wrapper that turns one task result or transcript into reusable reasoning artifacts and insights
- an aggregation wrapper that merges many local reasoning artifacts into a final global insight library

FoTClaw handles the orchestration around these interfaces; users only need to implement the algorithmic behavior for local reasoning and aggregation.

## Replacing the Default Algorithms

FoTClaw loads the local and global algorithm classes from editable settings in the project root `setting.yaml`:

- `local_reasoning_class`
- `global_reasoning_class`

Default values:

- `fot.fot_client:OpenClawFoTClient`
- `fot.fot_server:OpenClawFoTServer`

Set custom implementations by editing:

```yaml
local_reasoning_class: mypkg.reasoning:MyLocalReasoner
global_reasoning_class: mypkg.reasoning:MyGlobalReasoner
```

Your module must be importable from the Python environment that runs `fotclaw`.

### Minimal Example

```python
from typing import Any

from fot.fot_client import LocalReasoningClient
from fot.fot_server import GlobalReasoningServer


class MyLocalReasoner(LocalReasoningClient):
    def local_step_1(
        self,
        problem: str,
        *,
        custom_solution_instruction: str | None = None,
        insights_section: str | None = None,
    ) -> dict[str, Any]:
        return {"response": f"custom step 1 for {problem}", "usage": {}}

    def local_step_2(self, problem: str, step1_result: dict[str, Any]) -> dict[str, Any]:
        return {"response": "custom local reflection", "usage": {}}

    def local_step_3(
        self,
        problem: str,
        step1_result: dict[str, Any],
        step2_result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "valid_skills": {
                "insight_custom_local": "A custom local reasoning trace."
            },
            "usage": {},
        }


class MyGlobalReasoner(GlobalReasoningServer):
    def global_step_1(self, json_files: list[str] | None = None) -> dict[str, Any]:
        return {"insight_store": {"trace_000001": "custom trace"}}

    def global_step_2(self, collection_result: dict[str, Any]) -> dict[str, Any]:
        return {"profiling": {"clusters": [], "relationships": []}, "usage": {}}

    def global_step_3(
        self,
        collection_result: dict[str, Any],
        profiling_result: dict[str, Any],
        existing_encyclopedia: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "encyclopedia_dict": {
                "insight_custom_global": "A custom aggregated insight."
            },
            "usage": {},
        }
```

## Configuration

FoTClaw stores runtime state under project-local `.fotclaw/` by default. Override this with `FOTCLAW_HOME`.

User-editable settings live in the project root `setting.yaml`:

- `default_model`
- `aggregation_model`
- `openclaw_path`
- `local_reasoning_class`
- `global_reasoning_class`
- `auto_aggregate_enabled`
- `auto_aggregate_trace_threshold`
- `auto_aggregate_min_interval_seconds`

Edit `setting.yaml` directly to change these values.

Runtime aggregation metadata is stored separately in `./.fotclaw/config.json` by default.

## State and Artifacts

By default FoTClaw stores:

- editable settings at `./setting.yaml`
- per-agent state under `./.fotclaw/agents/<agent_id>/`
- extracted reasoning traces under `./.fotclaw/reasoning_traces/problem_XXXXXX.json`
- aggregation workspace under `./.fotclaw/aggregate/`
- persistent shared insights at `./.fotclaw/insight.json` and `./.fotclaw/insight.md`

Each agent directory contains its record, logs, workspace, and any recovered transcript path.

`fotclaw clean` removes FoTClaw-managed per-agent state, transient traces, and aggregation scratch files while preserving the persistent shared insight library.

## Repository Layout

- `src/fot/`
  FoT algorithm package
- `src/fot/fot_client.py`
  local reasoning interfaces and default implementation
- `src/fot/fot_server.py`
  global aggregation interfaces and default implementation
- `src/fotclaw/`
  FoTClaw host, CLI, supervisor, configuration, and OpenClaw integration

## OpenClaw Integration Notes

- FoTClaw creates a dedicated OpenClaw agent per background run so workspaces and transcripts stay isolated.
- FoTClaw serializes only OpenClaw agent creation to avoid `agents add` races; actual task execution still runs in parallel.
- FoTClaw uses OpenClaw for task execution, local FoT processing, and global aggregation so the full pipeline follows one model/runtime path.
- Global aggregation runs through the persistent OpenClaw agent `fotaggregation`.
- If `openclaw` is missing, FoTClaw fails fast with an explicit error.

## Research Framing

An intuitive way to think about FoTClaw is:

- each OpenClaw agent is like a researcher working on its own problem
- each local reasoning trace is that researcher's distilled procedural experience
- the aggregation stage is like a group meeting that consolidates those experiences
- `insight.md` is the shared lab notebook that future researchers can reuse

It is worth continuing to explore the design space of FoT, including personalization strategies, evaluation methodology, handling distribution drift across agents, and optimizing communication efficiency between agents and the server.

# Citation
```
@misc{yao2026federationtextinsightsharing,
      title={Federation over Text: Insight Sharing for Multi-Agent Reasoning}, 
      author={Dixi Yao and Tahseen Rabbani and Tian Li},
      year={2026},
      eprint={2604.16778},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.16778}, 
}
```
