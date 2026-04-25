from __future__ import annotations

from pathlib import Path
import json

from rpg_story.config import load_config
from rpg_story.eval.benchmark import (
    BenchmarkConfig,
    load_benchmark_config,
    run_benchmark_suite,
)
from rpg_story.llm.client import BaseLLMClient


class StaticBenchmarkClient(BaseLLMClient):
    def __init__(self) -> None:
        self.world_payload = {
            "world_id": "bench_world",
            "title": "Benchmark World",
            "world_bible": {
                "tech_level": "modern",
                "narrative_language": "en",
                "magic_rules": "none",
                "tone": "grounded",
            },
            "locations": [
                {
                    "location_id": "loc_001",
                    "name": "Square",
                    "kind": "town",
                    "description": "A square used for conversation benchmarks.",
                    "connected_to": ["loc_002"],
                    "tags": [],
                },
                {
                    "location_id": "loc_002",
                    "name": "Library",
                    "kind": "library",
                    "description": "A library that stores reference documents.",
                    "connected_to": ["loc_001"],
                    "tags": [],
                },
            ],
            "npcs": [
                {
                    "npc_id": "npc_001",
                    "name": "Lina",
                    "profession": "Caretaker",
                    "traits": ["calm"],
                    "goals": ["keep order"],
                    "starting_location": "loc_001",
                    "obedience_level": 0.5,
                    "stubbornness": 0.5,
                    "risk_tolerance": 0.4,
                    "disposition_to_player": 0,
                    "refusal_style": "brief",
                }
            ],
            "starting_location": "loc_001",
            "starting_hook": "You arrive at the square.",
            "initial_quest": "Learn what is happening here.",
        }
        self.turn_payload = {
            "narration": "Lina looks up for a moment.",
            "npc_dialogue": [{"npc_id": "npc_001", "text": "Everything has been calm lately."}],
            "world_updates": {
                "player_location": "loc_001",
                "npc_moves": [],
                "flags_delta": {},
                "quest_updates": {},
                "quest_progress_updates": [],
                "inventory_delta": {},
                "npc_personality_updates": [],
            },
            "memory_summary": "The player asked Lina about recent events.",
            "safety": {"refuse": False, "reason": None},
        }

    def generate_text(self, messages, *, temperature=None, top_p=None) -> str:
        return json.dumps(self.turn_payload, ensure_ascii=False)

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        schema_hint: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        schema_name = (response_format or {}).get("json_schema", {}).get("name")
        if schema_name == "WorldSpec":
            return json.loads(json.dumps(self.world_payload, ensure_ascii=False))
        if schema_name == "TurnOutput":
            return json.loads(json.dumps(self.turn_payload, ensure_ascii=False))
        raise AssertionError(f"Unexpected schema request: {schema_name}")


def test_load_benchmark_config(tmp_path: Path) -> None:
    bench_path = tmp_path / "benchmark.yaml"
    bench_path.write_text(
        """
output:
  base_dir: "data/eval"
experiment:
  runs_per_prompt: 1
rag:
  top_k: 2
  cases:
    - case_id: "simple_case"
      npc_id: "npc_1"
      location_id: "loc_1"
      query_text: "moon herb blacksmith"
      relevant_refs: ["gold_1"]
      docs:
        - ref: "gold_1"
          doc_type: "memory"
          npc_id: "npc_1"
          location_id: "loc_1"
          turn_id: 1
          timestamp: "2026-04-01T00:00:01+00:00"
          text: "The blacksmith needs moon herb for repairs."
world_prompts:
  - prompt_id: "demo"
    prompt: "Generate a benchmark world."
""".strip(),
        encoding="utf-8",
    )

    cfg = load_benchmark_config(bench_path)
    assert isinstance(cfg, BenchmarkConfig)
    assert cfg.rag_top_k == 2
    assert cfg.world_prompts[0].prompt_id == "demo"
    assert cfg.rag_cases[0].case_id == "simple_case"


def test_run_benchmark_suite_offline(tmp_path: Path) -> None:
    bench_path = tmp_path / "benchmark.yaml"
    output_base = tmp_path / "eval_output"
    bench_path.write_text(
        f"""
output:
  base_dir: "{output_base.as_posix()}"
experiment:
  runs_per_prompt: 1
  rag_top_k: 2
world_prompts:
  - prompt_id: "demo_prompt"
    prompt: "Generate a benchmark world."
    dialogue_texts:
      - "Introduce this place."
rag:
  top_k: 2
  cases:
    - case_id: "simple_case"
      npc_id: "npc_1"
      location_id: "loc_1"
      query_text: "moon herb blacksmith"
      relevant_refs: ["gold_1"]
      docs:
        - ref: "gold_1"
          doc_type: "memory"
          npc_id: "npc_1"
          location_id: "loc_1"
          turn_id: 1
          timestamp: "2026-04-01T00:00:01+00:00"
          text: "Blacksmith Tom needs moon herb to repair the anvil."
        - ref: "noise_1"
          doc_type: "lore"
          timestamp: "2026-04-01T00:00:02+00:00"
          text: "There is an old bell tower next to the square."
""".strip(),
        encoding="utf-8",
    )

    app_cfg = load_config("configs/config.yaml").resolve_paths(Path.cwd())
    bench_cfg = load_benchmark_config(bench_path)
    artifacts = run_benchmark_suite(
        app_cfg,
        bench_cfg,
        generate_plots=False,
        llm_factory=lambda _cfg: StaticBenchmarkClient(),
    )

    assert artifacts.summary["world_generation"]["success_count"] == 1
    assert artifacts.summary["dialogue"]["success_count"] == 1
    assert artifacts.summary["rag"]["average_recall_at_k"] == 1.0
    assert artifacts.figure_paths == []

    assert (artifacts.output_root / "world_generation.csv").exists()
    assert (artifacts.output_root / "dialogue_latency.csv").exists()
    assert (artifacts.output_root / "rag_recall.csv").exists()
    assert (artifacts.output_root / "summary.json").exists()
    assert (artifacts.output_root / "report.md").exists()


def test_run_benchmark_suite_without_rag(tmp_path: Path) -> None:
    bench_path = tmp_path / "benchmark_no_rag.yaml"
    output_base = tmp_path / "eval_output_no_rag"
    bench_path.write_text(
        f"""
output:
  base_dir: "{output_base.as_posix()}"
experiment:
  runs_per_prompt: 1
world_prompts:
  - prompt_id: "demo_prompt"
    prompt: "Generate a benchmark world."
    dialogue_texts:
      - "Introduce this place."
""".strip(),
        encoding="utf-8",
    )

    app_cfg = load_config("configs/config.yaml").resolve_paths(Path.cwd())
    bench_cfg = load_benchmark_config(bench_path)
    artifacts = run_benchmark_suite(
        app_cfg,
        bench_cfg,
        generate_plots=False,
        llm_factory=lambda _cfg: StaticBenchmarkClient(),
    )

    assert artifacts.summary["world_generation"]["success_count"] == 1
    assert artifacts.summary["dialogue"]["success_count"] == 1
    assert "rag" not in artifacts.summary
    report_text = (artifacts.output_root / "report.md").read_text(encoding="utf-8")
    assert "RAG" not in report_text
