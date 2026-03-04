"""Ingest local lore files into session RAG vector store.

Usage:
python scripts/ingest_lore.py --session <session_id> --path ./docs/lore
python scripts/ingest_lore.py --session <session_id> --path lore.md --tag canon
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rpg_story.config import load_config
from rpg_story.rag.embedder import make_embedder
from rpg_story.rag.index import RAGIndex
from rpg_story.rag.sources.lore_docs import build_lore_docs_from_paths
from rpg_story.rag.stores.hybrid import PersistentHybridStore
from rpg_story.rag.stores.memory import InMemoryStore


def _collect_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files: list[Path] = []
    for ext in ("*.txt", "*.md", "*.markdown"):
        files.extend(sorted(path.rglob(ext)))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--session", required=True)
    parser.add_argument("--path", required=True, help="File or directory with lore text files")
    parser.add_argument("--tag", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    source_path = Path(args.path)
    paths = _collect_paths(source_path)
    if not paths:
        raise FileNotFoundError(f"No lore files found: {source_path}")

    if str(cfg.rag.retrieval_backend).lower() == "in_memory":
        store = InMemoryStore()
        backend = "in_memory"
        embedder_name = "none"
    else:
        embedder, embedder_name = make_embedder(cfg)
        store = PersistentHybridStore(
            Path(cfg.app.vectorstore_dir) / args.session,
            embedder=embedder,
            lexical_weight=cfg.rag.lexical_weight,
            vector_weight=cfg.rag.vector_weight,
            recency_weight=cfg.rag.recency_weight,
            min_score=cfg.rag.min_score,
        )
        backend = "persistent_hybrid"

    index = RAGIndex(
        store,
        chunk_size_chars=cfg.rag.chunk_size_chars,
        chunk_overlap_chars=cfg.rag.chunk_overlap_chars,
    )

    docs = build_lore_docs_from_paths(
        session_id=args.session,
        paths=paths,
        tags=list(args.tag or []),
    )
    index.upsert(docs)

    print(
        f"ingested={len(docs)} files={len(paths)} store_total={store.count()} "
        f"backend={backend} embedder={embedder_name}"
    )


if __name__ == "__main__":
    main()
