"""Automated benchmark runner for thesis-ready evaluation metrics and plots."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable
import argparse
import csv
import json
import shutil

import yaml

from rpg_story.config import AppConfig, load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.eval.latency import summarize_durations, timed_call, utc_now_iso
from rpg_story.llm.client import BaseLLMClient, QwenOpenAICompatibleClient
from rpg_story.models.world import GameState, LocationSpec, NPCProfile, WorldBibleRules, WorldSpec
from rpg_story.rag.embedder import make_embedder
from rpg_story.rag.index import RAGIndex
from rpg_story.rag.retriever import RAGRetriever
from rpg_story.rag.stores.hybrid import PersistentHybridStore
from rpg_story.rag.stores.memory import InMemoryStore
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata
from rpg_story.world.generator import create_new_session, initialize_game_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "configs" / "eval_benchmark.yaml"

DEFAULT_DIALOGUE_TEXTS = {
    "zh": [
        "请你先用一句话介绍这里。",
        "最近这里最值得注意的事情是什么？",
        "如果我要继续推进剧情，你会给我什么建议？",
    ],
    "en": [
        "Introduce this place in one sentence.",
        "What is the most important recent event here?",
        "What would you suggest I do next to advance the story?",
    ],
}


@dataclass(frozen=True)
class WorldPromptCase:
    prompt_id: str
    prompt: str
    repeats: int
    dialogue_texts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RagDocumentCase:
    ref: str
    doc_type: str
    text: str
    npc_id: str | None = None
    location_id: str | None = None
    turn_id: int | None = None
    timestamp: str | None = None
    source: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RagCase:
    case_id: str
    query_text: str
    npc_id: str
    location_id: str
    relevant_refs: list[str]
    docs: list[RagDocumentCase]


@dataclass(frozen=True)
class BenchmarkConfig:
    output_base_dir: Path
    world_prompts: list[WorldPromptCase]
    rag_top_k: int
    rag_cases: list[RagCase]


@dataclass(frozen=True)
class WorldGenerationSample:
    prompt_id: str
    run_index: int
    session_id: str
    world_id: str
    world_title: str
    seconds: float | None
    success: bool
    started_at: str
    ended_at: str
    error: str = ""


@dataclass(frozen=True)
class DialogueSample:
    prompt_id: str
    run_index: int
    session_id: str
    npc_id: str
    npc_name: str
    turn_index: int
    player_text: str
    seconds: float | None
    success: bool
    started_at: str
    ended_at: str
    rag_retrieved_count: int = 0
    rag_always_include_count: int = 0
    rag_retrieved_ids: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(frozen=True)
class RagSample:
    case_id: str
    query_text: str
    top_k: int
    recall_at_k: float
    hit_at_k: float
    relevant_count: int
    retrieved_relevant_count: int
    retrieved_refs: list[str]
    relevant_refs: list[str]
    retrieval_backend: str
    embedding_backend: str
    filters: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkArtifacts:
    output_root: Path
    summary: dict[str, Any]
    figure_paths: list[Path]
    world_generation_samples: list[WorldGenerationSample]
    dialogue_samples: list[DialogueSample]
    rag_samples: list[RagSample]


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Benchmark config root must be a mapping")

    output_block = raw.get("output", {})
    experiment_block = raw.get("experiment", {})
    rag_block = raw.get("rag", {})

    base_dir = Path(str(output_block.get("base_dir", "data/eval")))
    if not base_dir.is_absolute():
        base_dir = (PROJECT_ROOT / base_dir).resolve()

    default_repeats = max(1, int(experiment_block.get("runs_per_prompt", 3)))
    world_prompts: list[WorldPromptCase] = []
    for index, item in enumerate(raw.get("world_prompts", []), start=1):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"world_prompts[{index}] is missing prompt text")
        world_prompts.append(
            WorldPromptCase(
                prompt_id=str(item.get("prompt_id") or f"prompt_{index}").strip(),
                prompt=prompt,
                repeats=max(1, int(item.get("repeats", default_repeats))),
                dialogue_texts=_clean_str_list(item.get("dialogue_texts", [])),
            )
        )

    rag_top_k = max(1, int(rag_block.get("top_k", experiment_block.get("rag_top_k", 3))))
    rag_cases: list[RagCase] = []
    for index, item in enumerate(rag_block.get("cases", []), start=1):
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or f"rag_case_{index}").strip()
        query_text = str(item.get("query_text", "")).strip()
        npc_id = str(item.get("npc_id", "")).strip()
        location_id = str(item.get("location_id", "")).strip()
        relevant_refs = _clean_str_list(item.get("relevant_refs", []))
        if not query_text or not npc_id or not location_id or not relevant_refs:
            raise ValueError(f"RAG case {case_id} is missing required fields")
        docs: list[RagDocumentCase] = []
        for doc_index, doc_row in enumerate(item.get("docs", []), start=1):
            if not isinstance(doc_row, dict):
                continue
            text = str(doc_row.get("text", "")).strip()
            if not text:
                raise ValueError(f"RAG case {case_id} doc #{doc_index} is missing text")
            docs.append(
                RagDocumentCase(
                    ref=str(doc_row.get("ref") or f"{case_id}_doc_{doc_index}").strip(),
                    doc_type=str(doc_row.get("doc_type") or "memory").strip(),
                    text=text,
                    npc_id=_optional_text(doc_row.get("npc_id")),
                    location_id=_optional_text(doc_row.get("location_id")),
                    turn_id=_optional_int(doc_row.get("turn_id")),
                    timestamp=_optional_text(doc_row.get("timestamp")),
                    source=_optional_text(doc_row.get("source")),
                    tags=_clean_str_list(doc_row.get("tags", [])),
                )
            )
        rag_cases.append(
            RagCase(
                case_id=case_id,
                query_text=query_text,
                npc_id=npc_id,
                location_id=location_id,
                relevant_refs=relevant_refs,
                docs=docs,
            )
        )

    return BenchmarkConfig(
        output_base_dir=base_dir,
        world_prompts=world_prompts,
        rag_top_k=rag_top_k,
        rag_cases=rag_cases,
    )


def run_benchmark_suite(
    cfg: AppConfig,
    benchmark_cfg: BenchmarkConfig,
    *,
    output_root: Path | None = None,
    benchmark_source_path: Path | None = None,
    llm_factory: Callable[[AppConfig], BaseLLMClient] | None = None,
    generate_plots: bool = True,
    console: Callable[[str], None] | None = None,
) -> BenchmarkArtifacts:
    console = console or (lambda _msg: None)
    llm_factory = llm_factory or (lambda app_cfg: QwenOpenAICompatibleClient(app_cfg))
    run_root = output_root or _new_output_root(benchmark_cfg.output_base_dir)
    run_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = replace(
        cfg,
        app=replace(
            cfg.app,
            sessions_dir=(run_root / "sessions").resolve(),
            worlds_dir=(run_root / "worlds").resolve(),
            vectorstore_dir=(run_root / "vectorstore").resolve(),
        ),
    )
    eval_cfg.app.sessions_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.app.worlds_dir.mkdir(parents=True, exist_ok=True)
    eval_cfg.app.vectorstore_dir.mkdir(parents=True, exist_ok=True)

    world_samples: list[WorldGenerationSample] = []
    dialogue_samples: list[DialogueSample] = []
    rag_samples: list[RagSample] = []

    if benchmark_cfg.world_prompts:
        world_samples, dialogue_samples = _run_world_and_dialogue_benchmarks(
            eval_cfg,
            benchmark_cfg,
            llm_factory=llm_factory,
            console=console,
        )
    if benchmark_cfg.rag_cases:
        rag_samples = _run_rag_benchmarks(eval_cfg, benchmark_cfg, run_root=run_root, console=console)

    summary = _build_summary(world_samples, dialogue_samples, rag_samples)
    figure_paths: list[Path] = []

    _write_samples_csv(run_root / "world_generation.csv", world_samples)
    _write_samples_csv(run_root / "dialogue_latency.csv", dialogue_samples)
    _write_samples_csv(run_root / "rag_recall.csv", rag_samples)
    _write_json(run_root / "summary.json", summary)
    _write_json(run_root / "benchmark_config_resolved.json", _benchmark_config_to_dict(benchmark_cfg))

    source_path = benchmark_source_path or DEFAULT_BENCHMARK_PATH
    if source_path.exists():
        shutil.copy2(source_path, run_root / "benchmark_config.yaml")

    if generate_plots:
        figure_paths = _generate_figures(
            world_samples=world_samples,
            dialogue_samples=dialogue_samples,
            rag_samples=rag_samples,
            output_dir=run_root / "figures",
        )

    _write_report(run_root / "report.md", summary=summary, figure_paths=figure_paths)
    return BenchmarkArtifacts(
        output_root=run_root,
        summary=summary,
        figure_paths=figure_paths,
        world_generation_samples=world_samples,
        dialogue_samples=dialogue_samples,
        rag_samples=rag_samples,
    )


def _run_world_and_dialogue_benchmarks(
    cfg: AppConfig,
    benchmark_cfg: BenchmarkConfig,
    *,
    llm_factory: Callable[[AppConfig], BaseLLMClient],
    console: Callable[[str], None],
) -> tuple[list[WorldGenerationSample], list[DialogueSample]]:
    world_samples: list[WorldGenerationSample] = []
    dialogue_samples: list[DialogueSample] = []
    total_runs = sum(case.repeats for case in benchmark_cfg.world_prompts)
    completed_runs = 0

    for case in benchmark_cfg.world_prompts:
        dialogue_texts = case.dialogue_texts or _default_dialogue_texts(case.prompt)
        for run_index in range(1, case.repeats + 1):
            completed_runs += 1
            console(f"[World {completed_runs}/{total_runs}] {case.prompt_id} run {run_index}")
            try:
                timing, payload = timed_call(
                    lambda: create_new_session(
                        cfg,
                        llm_factory(cfg),
                        case.prompt,
                        sessions_root=cfg.app.sessions_dir,
                        worlds_root=cfg.app.worlds_dir,
                    )
                )
                session_id, world, state = payload
                world_samples.append(
                    WorldGenerationSample(
                        prompt_id=case.prompt_id,
                        run_index=run_index,
                        session_id=session_id,
                        world_id=world.world_id,
                        world_title=world.title,
                        seconds=round(timing.seconds, 6),
                        success=True,
                        started_at=timing.started_at,
                        ended_at=timing.ended_at,
                    )
                )
            except Exception as exc:
                world_samples.append(
                    WorldGenerationSample(
                        prompt_id=case.prompt_id,
                        run_index=run_index,
                        session_id="",
                        world_id="",
                        world_title="",
                        seconds=None,
                        success=False,
                        started_at=utc_now_iso(),
                        ended_at=utc_now_iso(),
                        error=str(exc),
                    )
                )
                console(f"  world generation failed: {exc}")
                continue

            if not dialogue_texts:
                continue

            try:
                state, npc_id, npc_name = _prepare_dialogue_state(state)
            except Exception as exc:
                for turn_index, player_text in enumerate(dialogue_texts, start=1):
                    dialogue_samples.append(
                        DialogueSample(
                            prompt_id=case.prompt_id,
                            run_index=run_index,
                            session_id=session_id,
                            npc_id="",
                            npc_name="",
                            turn_index=turn_index,
                            player_text=player_text,
                            seconds=None,
                            success=False,
                            started_at=utc_now_iso(),
                            ended_at=utc_now_iso(),
                            error=str(exc),
                        )
                    )
                console(f"  dialogue skipped: {exc}")
                continue

            for turn_index, player_text in enumerate(dialogue_texts, start=1):
                console(f"  [Dialogue] {case.prompt_id} run {run_index} turn {turn_index}")
                try:
                    timing, payload = timed_call(
                        lambda: TurnPipeline(
                            cfg=cfg,
                            llm_client=llm_factory(cfg),
                            sessions_root=cfg.app.sessions_dir,
                        ).run_turn(state, player_text, npc_id)
                    )
                    state, _output, log_record = payload
                    rag_debug = log_record.get("rag", {}) if isinstance(log_record, dict) else {}
                    dialogue_samples.append(
                        DialogueSample(
                            prompt_id=case.prompt_id,
                            run_index=run_index,
                            session_id=session_id,
                            npc_id=npc_id,
                            npc_name=npc_name,
                            turn_index=turn_index,
                            player_text=player_text,
                            seconds=round(timing.seconds, 6),
                            success=True,
                            started_at=timing.started_at,
                            ended_at=timing.ended_at,
                            rag_retrieved_count=int(rag_debug.get("counts", {}).get("retrieved", 0)),
                            rag_always_include_count=int(rag_debug.get("counts", {}).get("always_include", 0)),
                            rag_retrieved_ids=[str(v) for v in rag_debug.get("retrieved_ids", [])],
                        )
                    )
                except Exception as exc:
                    dialogue_samples.append(
                        DialogueSample(
                            prompt_id=case.prompt_id,
                            run_index=run_index,
                            session_id=session_id,
                            npc_id=npc_id,
                            npc_name=npc_name,
                            turn_index=turn_index,
                            player_text=player_text,
                            seconds=None,
                            success=False,
                            started_at=utc_now_iso(),
                            ended_at=utc_now_iso(),
                            error=str(exc),
                        )
                    )
                    console(f"    dialogue failed: {exc}")

    return world_samples, dialogue_samples


def _run_rag_benchmarks(
    cfg: AppConfig,
    benchmark_cfg: BenchmarkConfig,
    *,
    run_root: Path,
    console: Callable[[str], None],
) -> list[RagSample]:
    samples: list[RagSample] = []
    for index, case in enumerate(benchmark_cfg.rag_cases, start=1):
        console(f"[RAG {index}/{len(benchmark_cfg.rag_cases)}] {case.case_id}")
        session_id = f"rag_{case.case_id}"
        world = _build_rag_benchmark_world()
        state = initialize_game_state(world, session_id=session_id)
        if not world.get_location(case.location_id):
            raise ValueError(f"Unknown RAG benchmark location_id: {case.location_id}")
        if not any(npc.npc_id == case.npc_id for npc in world.npcs):
            raise ValueError(f"Unknown RAG benchmark npc_id: {case.npc_id}")
        state.player_location = case.location_id
        state.npc_locations[case.npc_id] = case.location_id

        store, retrieval_backend, embedding_backend = _build_rag_store(
            cfg,
            store_path=run_root / "rag_vectorstore" / case.case_id,
        )
        index = RAGIndex(
            store,
            chunk_size_chars=cfg.rag.chunk_size_chars,
            chunk_overlap_chars=cfg.rag.chunk_overlap_chars,
        )
        index.build_default(session_id, world)
        documents, ref_to_id = _build_rag_documents(session_id, case)
        index.upsert(documents)

        retriever = RAGRetriever(index)
        pack = retriever.get_forced_context_pack(
            session_id=session_id,
            world=world,
            state=state,
            npc_id=case.npc_id,
            sessions_root=run_root / "rag_sessions",
            last_n_summaries=cfg.rag.summary_window,
            top_k=benchmark_cfg.rag_top_k,
            query_text=case.query_text,
        )
        retrieved_docs = list(pack.get("retrieved", []))
        retrieved_source_ids = [_parent_or_self(doc) for doc in retrieved_docs]
        relevant_ids = [ref_to_id[ref] for ref in case.relevant_refs if ref in ref_to_id]
        hits = len(set(retrieved_source_ids) & set(relevant_ids))
        id_to_ref = {doc_id: ref for ref, doc_id in ref_to_id.items()}
        retrieved_refs = [id_to_ref.get(doc_id, doc_id) for doc_id in retrieved_source_ids]

        samples.append(
            RagSample(
                case_id=case.case_id,
                query_text=case.query_text,
                top_k=benchmark_cfg.rag_top_k,
                recall_at_k=round(hits / len(relevant_ids), 6) if relevant_ids else 0.0,
                hit_at_k=1.0 if hits > 0 else 0.0,
                relevant_count=len(relevant_ids),
                retrieved_relevant_count=hits,
                retrieved_refs=retrieved_refs,
                relevant_refs=list(case.relevant_refs),
                retrieval_backend=retrieval_backend,
                embedding_backend=embedding_backend,
                filters=list(pack.get("debug", {}).get("filters", [])),
            )
        )
    return samples


def _build_rag_store(cfg: AppConfig, *, store_path: Path):
    backend = str(cfg.rag.retrieval_backend or "persistent_hybrid").lower()
    if backend == "in_memory":
        return InMemoryStore(), "in_memory", "none"
    embedder, embedder_backend = make_embedder(cfg)
    store = PersistentHybridStore(
        store_path,
        embedder=embedder,
        lexical_weight=cfg.rag.lexical_weight,
        vector_weight=cfg.rag.vector_weight,
        recency_weight=cfg.rag.recency_weight,
        min_score=cfg.rag.min_score,
    )
    return store, store.backend_name, embedder_backend


def _build_rag_documents(session_id: str, case: RagCase) -> tuple[list[Document], dict[str, str]]:
    documents: list[Document] = []
    ref_to_id: dict[str, str] = {}
    for index, spec in enumerate(case.docs, start=1):
        metadata = {
            "doc_type": spec.doc_type,
            "session_id": session_id,
            "npc_id": spec.npc_id,
            "location_id": spec.location_id,
            "turn_id": spec.turn_id if spec.turn_id is not None else index,
            "timestamp": spec.timestamp or f"2026-04-01T00:00:{index:02d}+00:00",
            "source": spec.source or "benchmark",
            "tags": spec.tags,
        }
        normalized = normalize_metadata(metadata, strict=True)
        doc_id = make_doc_id(normalized, spec.text)
        ref_to_id[spec.ref] = doc_id
        documents.append(Document(id=doc_id, text=spec.text, metadata=normalized))
    missing = [ref for ref in case.relevant_refs if ref not in ref_to_id]
    if missing:
        raise ValueError(f"RAG case {case.case_id} references unknown relevant docs: {missing}")
    return documents, ref_to_id


def _prepare_dialogue_state(state: GameState) -> tuple[GameState, str, str]:
    local_npc_ids = state.npcs_at(state.player_location)
    if local_npc_ids:
        npc_id = local_npc_ids[0]
        npc = next((item for item in state.world.npcs if item.npc_id == npc_id), None)
        if npc is None:
            raise ValueError(f"NPC {npc_id} not found in world roster")
        return state, npc.npc_id, npc.name
    if not state.world.npcs:
        raise ValueError("generated world has no NPCs")
    npc = state.world.npcs[0]
    updated = state.model_copy(deep=True)
    updated.player_location = updated.npc_locations.get(npc.npc_id, updated.player_location)
    return updated, npc.npc_id, npc.name


def _build_summary(
    world_samples: list[WorldGenerationSample],
    dialogue_samples: list[DialogueSample],
    rag_samples: list[RagSample],
) -> dict[str, Any]:
    world_success = [sample.seconds for sample in world_samples if sample.success and sample.seconds is not None]
    dialogue_success = [sample.seconds for sample in dialogue_samples if sample.success and sample.seconds is not None]
    rag_recalls = [sample.recall_at_k for sample in rag_samples]

    summary = {
        "generated_at": utc_now_iso(),
        "world_generation": {
            "success_count": sum(1 for sample in world_samples if sample.success),
            "total_count": len(world_samples),
            "success_rate": round(
                sum(1 for sample in world_samples if sample.success) / len(world_samples),
                6,
            )
            if world_samples
            else None,
            "duration_seconds": summarize_durations(world_success),
            "by_prompt": _group_success_stats(world_samples, key_name="prompt_id"),
        },
        "dialogue": {
            "success_count": sum(1 for sample in dialogue_samples if sample.success),
            "total_count": len(dialogue_samples),
            "success_rate": round(
                sum(1 for sample in dialogue_samples if sample.success) / len(dialogue_samples),
                6,
            )
            if dialogue_samples
            else None,
            "duration_seconds": summarize_durations(dialogue_success),
            "by_prompt": _group_success_stats(dialogue_samples, key_name="prompt_id"),
        },
    }
    if rag_samples:
        summary["rag"] = {
            "case_count": len(rag_samples),
            "average_recall_at_k": round(fmean(rag_recalls), 6) if rag_recalls else None,
            "average_hit_at_k": round(fmean(sample.hit_at_k for sample in rag_samples), 6) if rag_samples else None,
            "top_k": rag_samples[0].top_k if rag_samples else None,
        }
    return summary


def _group_success_stats(samples: Iterable[Any], *, key_name: str) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        if not getattr(sample, "success", False):
            continue
        seconds = getattr(sample, "seconds", None)
        if not isinstance(seconds, (int, float)):
            continue
        key = str(getattr(sample, key_name))
        grouped.setdefault(key, []).append(float(seconds))
    return {key: summarize_durations(values) for key, values in grouped.items()}


def _write_samples_csv(path: Path, samples: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        path.write_text("", encoding="utf-8")
        return
    rows = [_serialize_row(asdict(sample)) for sample in samples]
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_report(path: Path, *, summary: dict[str, Any], figure_paths: list[Path]) -> None:
    lines = [
        "# 自动实验报告",
        "",
        f"- 生成时间：{summary.get('generated_at', '')}",
        f"- 世界生成平均耗时：{_fmt_metric(summary.get('world_generation', {}).get('duration_seconds', {}).get('mean'))} 秒",
        f"- NPC 对话平均耗时：{_fmt_metric(summary.get('dialogue', {}).get('duration_seconds', {}).get('mean'))} 秒",
        "",
        "## 图表",
    ]
    rag_summary = summary.get("rag")
    if isinstance(rag_summary, dict):
        lines.insert(
            5,
            f"- RAG 平均 Recall@K：{_fmt_metric(rag_summary.get('average_recall_at_k'))}",
        )
    if figure_paths:
        for fig_path in figure_paths:
            rel = fig_path.relative_to(path.parent)
            lines.append(f"- `{rel.as_posix()}`")
    else:
        lines.append("- 未生成图表")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _generate_figures(
    *,
    world_samples: list[WorldGenerationSample],
    dialogue_samples: list[DialogueSample],
    rag_samples: list[RagSample],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    figure_paths: list[Path] = []

    figure_paths.extend(_plot_world_generation_bar(plt, world_samples, output_dir))
    figure_paths.extend(_plot_world_generation_line(plt, world_samples, output_dir))
    figure_paths.extend(_plot_dialogue_bar(plt, dialogue_samples, output_dir))
    figure_paths.extend(_plot_dialogue_line(plt, dialogue_samples, output_dir))
    figure_paths.extend(_plot_rag_bar(plt, rag_samples, output_dir))
    return figure_paths


def _plot_world_generation_bar(plt, samples: list[WorldGenerationSample], output_dir: Path) -> list[Path]:
    grouped = _collect_success_seconds(samples, key_name="prompt_id")
    if not grouped:
        return []
    labels = list(grouped.keys())
    means = [fmean(values) for values in grouped.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(labels)])
    _annotate_bars(ax, bars, suffix="s")
    ax.set_title("Average World Generation Latency")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Latency (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "world_generation_bar")


def _plot_world_generation_line(plt, samples: list[WorldGenerationSample], output_dir: Path) -> list[Path]:
    grouped_rows: dict[str, list[WorldGenerationSample]] = {}
    for sample in samples:
        if sample.success and sample.seconds is not None:
            grouped_rows.setdefault(sample.prompt_id, []).append(sample)
    if not grouped_rows:
        return []
    fig, ax = plt.subplots(figsize=(10, 6))
    for prompt_id, rows in grouped_rows.items():
        rows = sorted(rows, key=lambda row: row.run_index)
        ax.plot(
            [row.run_index for row in rows],
            [row.seconds for row in rows if row.seconds is not None],
            marker="o",
            linewidth=2,
            label=prompt_id,
        )
    ax.set_title("World Generation Latency by Run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Latency (s)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, "world_generation_line")


def _plot_dialogue_bar(plt, samples: list[DialogueSample], output_dir: Path) -> list[Path]:
    grouped = _collect_success_seconds(samples, key_name="prompt_id")
    if not grouped:
        return []
    labels = list(grouped.keys())
    means = [fmean(values) for values in grouped.values()]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, means, color=["#17becf", "#bcbd22", "#d62728"][: len(labels)])
    _annotate_bars(ax, bars, suffix="s")
    ax.set_title("Average NPC Dialogue Latency")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Latency (s)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "dialogue_latency_bar")


def _plot_dialogue_line(plt, samples: list[DialogueSample], output_dir: Path) -> list[Path]:
    grouped_rows: dict[str, list[DialogueSample]] = {}
    for sample in samples:
        if sample.success and sample.seconds is not None:
            grouped_rows.setdefault(sample.prompt_id, []).append(sample)
    if not grouped_rows:
        return []
    fig, ax = plt.subplots(figsize=(10, 6))
    for prompt_id, rows in grouped_rows.items():
        rows = sorted(rows, key=lambda row: (row.run_index, row.turn_index))
        ax.plot(
            list(range(1, len(rows) + 1)),
            [row.seconds for row in rows if row.seconds is not None],
            marker="o",
            linewidth=2,
            label=prompt_id,
        )
    ax.set_title("NPC Dialogue Latency by Turn")
    ax.set_xlabel("Dialogue Sample")
    ax.set_ylabel("Latency (s)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save_figure(fig, output_dir, "dialogue_latency_line")


def _plot_rag_bar(plt, samples: list[RagSample], output_dir: Path) -> list[Path]:
    if not samples:
        return []
    labels = [sample.case_id for sample in samples]
    recalls = [sample.recall_at_k for sample in samples]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, recalls, color="#9467bd")
    _annotate_bars(ax, bars, decimals=3)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"RAG Recall@{samples[0].top_k}")
    ax.set_xlabel("检索样例")
    ax.set_ylabel("召回率")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return _save_figure(fig, output_dir, "rag_recall_bar")


def _save_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.svg"]
    for path in paths:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.clf()
    return paths


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except Exception as exc:  # pragma: no cover - import error depends on runtime env
        raise RuntimeError(
            "matplotlib is required for figure generation. "
            "Install dependencies with `py -3.14 -m pip install -r requirements.txt`."
        ) from exc
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _annotate_bars(ax, bars, *, suffix: str = "", decimals: int = 2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.{decimals}f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def _collect_success_seconds(samples: list[Any], *, key_name: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        if not getattr(sample, "success", False):
            continue
        seconds = getattr(sample, "seconds", None)
        if not isinstance(seconds, (int, float)):
            continue
        grouped.setdefault(str(getattr(sample, key_name)), []).append(float(seconds))
    return grouped


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (list, dict)):
            serialized[key] = json.dumps(value, ensure_ascii=False)
        else:
            serialized[key] = value
    return serialized


def _benchmark_config_to_dict(cfg: BenchmarkConfig) -> dict[str, Any]:
    payload = asdict(cfg)
    payload["output_base_dir"] = str(cfg.output_base_dir)
    return payload


def _build_rag_benchmark_world() -> WorldSpec:
    return WorldSpec(
        world_id="rag_eval_world",
        title="RAG Benchmark World",
        world_bible=WorldBibleRules(
            tech_level="modern",
            narrative_language="zh",
            magic_rules="none",
            tone="grounded",
        ),
        locations=[
            LocationSpec(
                location_id="loc_1",
                name="中央广场",
                kind="town",
                description="商队与情报在这里汇集。",
                connected_to=["loc_2"],
                tags=["market"],
            ),
            LocationSpec(
                location_id="loc_2",
                name="北侧机库",
                kind="hangar",
                description="旧设备和维护档案堆满机库。",
                connected_to=["loc_1"],
                tags=["repair"],
            ),
        ],
        npcs=[
            NPCProfile(
                npc_id="npc_1",
                name="托姆",
                profession="铁匠",
                traits=["谨慎"],
                goals=["修好铁砧"],
                starting_location="loc_1",
                obedience_level=0.5,
                stubbornness=0.5,
                risk_tolerance=0.4,
                disposition_to_player=0,
                refusal_style="直接",
            ),
            NPCProfile(
                npc_id="npc_2",
                name="伊芙",
                profession="工程师",
                traits=["理性"],
                goals=["重启灯塔"],
                starting_location="loc_2",
                obedience_level=0.5,
                stubbornness=0.4,
                risk_tolerance=0.6,
                disposition_to_player=0,
                refusal_style="简洁",
            ),
        ],
        starting_location="loc_1",
        starting_hook="一座依赖情报与修理维持运转的前哨城。",
        initial_quest="先找本地居民了解现状。",
    )


def _default_dialogue_texts(prompt: str) -> list[str]:
    language = "zh" if any("\u4e00" <= ch <= "\u9fff" for ch in prompt) else "en"
    return list(DEFAULT_DIALOGUE_TEXTS[language])


def _new_output_root(base_dir: Path) -> Path:
    timestamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0].replace("+00:00", "Z")
    return base_dir / timestamp.replace("T", "_")


def _parent_or_self(doc: Document) -> str:
    return str(doc.metadata.get("parent_id") or doc.id)


def _clean_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _optional_text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _fmt_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automated thesis benchmarks and generate figures.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--output-dir", default=None, help="Optional fixed output directory.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config).resolve_paths(PROJECT_ROOT)
    benchmark_cfg = load_benchmark_config(args.benchmark)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    artifacts = run_benchmark_suite(
        cfg,
        benchmark_cfg,
        output_root=output_dir,
        benchmark_source_path=Path(args.benchmark).resolve(),
        generate_plots=not args.no_plots,
        console=lambda msg: print(msg, flush=True),
    )
    summary = artifacts.summary
    print("")
    print(f"Output directory: {artifacts.output_root}")
    print(
        "World generation avg (s): "
        f"{_fmt_metric(summary.get('world_generation', {}).get('duration_seconds', {}).get('mean'))}"
    )
    print(
        "Dialogue avg (s): "
        f"{_fmt_metric(summary.get('dialogue', {}).get('duration_seconds', {}).get('mean'))}"
    )
    rag_summary = summary.get("rag")
    if isinstance(rag_summary, dict):
        print(f"RAG average recall@k: {_fmt_metric(rag_summary.get('average_recall_at_k'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
