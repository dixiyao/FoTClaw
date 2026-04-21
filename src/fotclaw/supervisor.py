from __future__ import annotations

import argparse

from fotclaw.manager import delete_agent, run_agent_bootstrap, run_agent_postprocessor, run_agent_supervisor, run_aggregate_supervisor


def main() -> int:
    parser = argparse.ArgumentParser(description="Internal FoTClaw supervisor process.")
    parser.add_argument("--home", required=True, help="FoTClaw state directory")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap the agent before execution")
    parser.add_argument("--postprocess", action="store_true", help="Run FoT postprocessing instead of execution")
    parser.add_argument("--aggregate", action="store_true", help="Run background aggregation")
    parser.add_argument("--delete-agent-id", default=None, help="Delete the given FoTClaw agent id")
    parser.add_argument("--delete-agent-name", default=None, help="Delete the given FoTClaw agent name")
    parser.add_argument("agent_id", nargs="?", help="Agent id to supervise")
    args = parser.parse_args()
    if args.delete_agent_id or args.delete_agent_name:
        delete_agent(args.home, agent_id=args.delete_agent_id, name=args.delete_agent_name)
    elif args.aggregate:
        run_aggregate_supervisor(args.home, args.agent_id or "agt-aggregate")
    elif args.postprocess:
        run_agent_postprocessor(args.home, args.agent_id)
    elif args.bootstrap:
        run_agent_bootstrap(args.home, args.agent_id)
    else:
        run_agent_supervisor(args.home, args.agent_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
