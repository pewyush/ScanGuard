"""Command-line entry point for ScanGuard."""

import argparse
import os
from typing import Optional, Sequence

from scanguard.experiments import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scanguard",
        description="Unsupervised visual anomaly detection on MVTec AD 2",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser(
        "train",
        aliases=["compare"],
        help="Train the autoencoder, fit PaDiM, compare them, and write results",
    )
    train.add_argument("--data-dir", default="data/raw", help="Dataset root (contains <category>/)")
    train.add_argument("--category", default="can", help="MVTec AD 2 category")
    train.add_argument(
        "--full",
        action="store_true",
        help="Use every training/validation/test image (default: small quick slice)",
    )
    train.add_argument("--epochs", type=int, default=None, help="Autoencoder epochs")
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--latent-dim", type=int, default=128, help="Autoencoder latent size")
    train.add_argument("--base-channels", type=int, default=32)
    train.add_argument("--padim-latent-dim", type=int, default=100)
    train.add_argument("--padim-reg", type=float, default=0.01)
    train.add_argument("--img-size", type=int, default=256)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--checkpoint-dir", default="checkpoints")
    train.add_argument("--results-dir", default="results")
    train.add_argument("--figures-dir", default="results/figures")
    train.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse an existing autoencoder checkpoint instead of retraining",
    )
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(handler=_train)

    serve = subparsers.add_parser("serve", help="Start the comparison dashboard")
    serve.add_argument("--host", default=os.getenv("SCANGUARD_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=int(os.getenv("SCANGUARD_PORT", "8000")))
    serve.add_argument("--category", default=os.getenv("SCANGUARD_CATEGORY", "can"))
    serve.add_argument("--results", default=None, help="Path to results JSON")
    serve.add_argument("--checkpoint", default=None, help="Path to autoencoder checkpoint")
    serve.add_argument("--padim", default=None, help="Path to PaDiM statistics (.npz)")
    serve.add_argument("--figures", default=None, help="Directory with result figures")
    serve.set_defaults(handler=_serve)

    return parser


def _train(args: argparse.Namespace) -> None:
    run(args)


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    os.environ["SCANGUARD_CATEGORY"] = args.category
    if args.results:
        os.environ["SCANGUARD_RESULTS"] = args.results
    if args.checkpoint:
        os.environ["SCANGUARD_CHECKPOINT"] = args.checkpoint
    if args.padim:
        os.environ["SCANGUARD_PADIM"] = args.padim
    if args.figures:
        os.environ["SCANGUARD_FIGURES"] = args.figures

    uvicorn.run("scanguard.serving.app:app", host=args.host, port=args.port)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
