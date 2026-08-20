"""``python -m pkcv`` command line."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pkcv.config import Config


def _log(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _emit(result) -> None:
    payload = result.to_dict() if hasattr(result, "to_dict") else result
    print(json.dumps(payload, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pkcv", description="Soccer penalty-kick CV dataset pipeline")
    p.add_argument("--config", default=None, help="path to a YAML config (default: configs/default.yaml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", help="probe every source: what exists, licence, accessibility")
    inv.add_argument("--source", action="append", help="source slug or alias; repeatable")

    ing = sub.add_parser("ingest", help="download accessible sources and map them onto the schema")
    ing.add_argument("--source", action="append")
    ing.add_argument("--limit", type=int, default=None, help="max kicks per source")
    ing.add_argument("--include-gated", action="store_true",
                     help="also record placeholder rows for sources whose media is gated")

    pro = sub.add_parser("process", help="run vision (video sources) and build the temporal tables")
    pro.add_argument("--limit", type=int, default=5)
    pro.add_argument("--all", action="store_true")
    pro.add_argument("--pk-id", action="append", dest="pk_ids")
    pro.add_argument("--force", action="store_true", help="reprocess kicks that already have outputs")

    q = sub.add_parser("qc", help="grade every kick and render visual QC artifacts")
    q.add_argument("--overlays", type=int, default=5, help="how many kicks to render (0 disables)")
    q.add_argument("--pk-id", action="append", dest="pk_ids")

    pub = sub.add_parser("publish", help="upload licence-filtered derived artifacts to Hugging Face")
    pub.add_argument("--repo-id", default=None)
    pub.add_argument("--dry-run", action="store_true", help="stage and report without uploading")
    pub.add_argument("--public", action="store_true", help="create the repo public (default: private)")

    rep = sub.add_parser("report", help="print the end-to-end corpus report")
    rep.add_argument("--json", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _log(args.verbose)
    cfg = Config.load(args.config)

    if args.command == "inventory":
        from pkcv.pipeline import inventory

        _emit(inventory(cfg, args.source))
    elif args.command == "ingest":
        from pkcv.pipeline import ingest

        _emit(ingest(cfg, args.source, args.limit, args.include_gated))
    elif args.command == "process":
        from pkcv.pipeline import process

        _emit(process(cfg, args.limit, args.all, args.pk_ids, args.force))
    elif args.command == "qc":
        from pkcv.pipeline import run_qc

        _emit(run_qc(cfg, args.overlays, args.pk_ids))
    elif args.command == "publish":
        from pkcv.hf.publish import publish

        _emit(publish(cfg, args.repo_id, args.dry_run, private=not args.public))
    elif args.command == "report":
        from pkcv.report import corpus_report, render_text

        rep = corpus_report(cfg)
        print(json.dumps(rep, indent=2, default=str) if args.json else render_text(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
