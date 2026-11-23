import argparse
import sys

from .common import config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real local security-model robustness experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    dashboard = commands.add_parser("dashboard", help="Open the local results dashboard")
    dashboard.add_argument("--port", type=int, default=8873)
    for name in ["prepare", "train", "attack-evasion", "attack-poison", "defend", "evaluate", "verify-all"]:
        command = commands.add_parser(name)
        if name in ["prepare", "verify-all"]:
            command.add_argument("--download", action="store_true", help="Allow one-time official UCI dataset download")
    args = parser.parse_args(argv)
    try:
        cfg = config(args.config)
        if args.command == "dashboard":
            from .dashboard import serve
            serve(cfg, args.port)
            return
        from .data import prepare, verify
        from .eval.workflow import train, evasion, poisoning, defend, evaluate
        actions = {"prepare": lambda: prepare(cfg, args.download), "train": lambda: train(cfg), "attack-evasion": lambda: evasion(cfg), "attack-poison": lambda: poisoning(cfg), "defend": lambda: defend(cfg), "evaluate": lambda: evaluate(cfg)}
        if args.command == "verify-all":
            for label, action in [("prepare", lambda: prepare(cfg, args.download)), ("train", lambda: train(cfg)), ("attack-evasion", lambda: evasion(cfg)), ("attack-poison", lambda: poisoning(cfg)), ("defend", lambda: defend(cfg)), ("evaluate", lambda: evaluate(cfg))]:
                print(f"Running {label}", flush=True)
                result = action()
                if label == "evaluate":
                    print(result, flush=True)
            verify(cfg)
            print("Verification completed on checksum-verified real data.", flush=True)
        else:
            result = actions[args.command]()
            print(result if args.command == "evaluate" else f"Completed {args.command}", flush=True)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
