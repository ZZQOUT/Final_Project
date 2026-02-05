from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rpg_story.config import load_config
from rpg_story.engine.orchestrator import TurnPipeline
from rpg_story.llm.client import MockLLMClient
from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState
from rpg_story.persistence.store import append_turn_log
from rpg_story.rag.index import RAGIndex
from rpg_story.rag.retriever import RAGRetriever
from rpg_story.rag.stores.memory import InMemoryStore
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata
from rpg_story.rag.sources.summaries import build_summary_docs_from_turn_logs


def make_world() -> WorldSpec:
    locations = [
        LocationSpec(
            location_id="loc_1",
            name="Town",
            kind="town",
            description="A small town.",
            connected_to=["loc_2"],
        ),
        LocationSpec(
            location_id="loc_2",
            name="Bridge",
            kind="bridge",
            description="An old bridge.",
            connected_to=["loc_1"],
        ),
    ]
    npc = NPCProfile(
        npc_id="npc_1",
        name="Mara",
        profession="Merchant",
        traits=["practical"],
        goals=["trade"],
        starting_location="loc_1",
        obedience_level=0.5,
        stubbornness=0.5,
        risk_tolerance=0.5,
        disposition_to_player=0,
        refusal_style="polite",
    )
    bible = WorldBibleRules(tech_level="medieval", magic_rules="low", tone="grounded")
    return WorldSpec(
        world_id="world_rag",
        title="RAG World",
        world_bible=bible,
        locations=locations,
        npcs=[npc],
        starting_location="loc_1",
        starting_hook="A rumor spreads.",
        initial_quest="Deliver a message.",
    )


def make_state(world: WorldSpec) -> GameState:
    return GameState(
        session_id="sess_rag",
        created_at=datetime.now(timezone.utc).isoformat(),
        world=world,
        player_location="loc_1",
        npc_locations={"npc_1": "loc_1"},
    )


def test_always_include_present(tmp_path: Path):
    world = make_world()
    state = make_state(world)
    store = InMemoryStore()
    index = RAGIndex(store)
    index.build_default(state.session_id, world)
    retriever = RAGRetriever(index)

    pack = retriever.get_forced_context_pack(
        session_id=state.session_id,
        world=world,
        state=state,
        npc_id="npc_1",
        sessions_root=tmp_path,
        last_n_summaries=2,
        top_k=2,
        query_text="hello",
    )
    always = pack["always_include"]
    assert len([doc for doc in always if doc.metadata.get("doc_type") == "world_bible"]) == 1
    assert len([doc for doc in always if doc.metadata.get("doc_type") == "location"]) == 1
    assert len([doc for doc in always if doc.metadata.get("doc_type") == "npc_profile"]) == 1


def test_filter_by_npc_and_location():
    store = InMemoryStore()
    index = RAGIndex(store)
    docs = []
    meta1 = normalize_metadata(
        {
            "doc_type": "memory",
            "session_id": "sess",
            "npc_id": "npc_1",
            "location_id": "loc_1",
            "turn_id": 1,
            "timestamp": "t1",
        }
    )
    docs.append(Document(id=make_doc_id(meta1, "npc1 memory"), text="npc1 memory", metadata=meta1))
    meta2 = normalize_metadata(
        {
            "doc_type": "memory",
            "session_id": "sess",
            "npc_id": "npc_2",
            "location_id": "loc_2",
            "turn_id": 2,
            "timestamp": "t2",
        }
    )
    docs.append(Document(id=make_doc_id(meta2, "npc2 memory"), text="npc2 memory", metadata=meta2))
    store.add(docs)
    retriever = RAGRetriever(index)

    world = make_world()
    state = make_state(world)
    pack = retriever.get_forced_context_pack(
        session_id="sess",
        world=world,
        state=state,
        npc_id="npc_1",
        sessions_root=Path("data/sessions"),
        last_n_summaries=0,
        top_k=1,
        query_text="memory",
    )
    retrieved = pack["retrieved"]
    assert retrieved
    assert retrieved[0].metadata.get("npc_id") == "npc_1"


def test_last_n_summaries(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    session_id = "sess_sum"
    for i in range(3):
        record = {
            "session_id": session_id,
            "turn_index": i,
            "timestamp": f"t{i}",
            "output": {"memory_summary": f"summary {i}"},
        }
        append_turn_log(session_id, record, sessions_root)
    docs = build_summary_docs_from_turn_logs(session_id, sessions_root, limit=2)
    assert len(docs) == 2
    texts = [doc.text for doc in docs]
    assert texts == ["summary 1", "summary 2"]


def test_forced_pack_includes_last_summaries(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    session_id = "sess_pack"
    for i in range(3):
        record = {
            "session_id": session_id,
            "turn_index": i,
            "timestamp": f"t{i}",
            "output": {"memory_summary": f"summary {i}"},
        }
        append_turn_log(session_id, record, sessions_root)

    world = make_world()
    state = make_state(world)
    store = InMemoryStore()
    index = RAGIndex(store)
    index.build_default(session_id, world)
    retriever = RAGRetriever(index)

    pack = retriever.get_forced_context_pack(
        session_id=session_id,
        world=world,
        state=state,
        npc_id="npc_1",
        sessions_root=sessions_root,
        last_n_summaries=2,
        top_k=1,
        query_text="hello",
    )
    summary_docs = [doc for doc in pack["always_include"] if doc.metadata.get("doc_type") == "summary"]
    assert len(summary_docs) == 2
    assert [doc.text for doc in summary_docs] == ["summary 1", "summary 2"]


def test_orchestrator_injects_rag(tmp_path: Path):
    cfg = load_config("configs/config.yaml")
    sessions_root = tmp_path / "sessions"
    world = make_world()
    state = make_state(world)

    output_json = (
        "{"
        '"narration":"OK",'
        '"npc_dialogue":[], '
        '"world_updates":{'
        '  "player_location":"loc_1",'
        '  "npc_moves":[],'
        '  "flags_delta":{},'
        '  "quest_updates":{}'
        '},'
        '"memory_summary":"Summary.",'
        '"safety":{"refuse":false,"reason":null}'
        "}"
    )
    llm = MockLLMClient([output_json])
    pipeline = TurnPipeline(cfg=cfg, llm_client=llm, sessions_root=sessions_root)

    _updated_state, _output, log_record = pipeline.run_turn(state, "Hello", "npc_1")
    assert llm.last_system_prompt is not None
    assert "=== WORLD BIBLE ===" in llm.last_system_prompt
    assert "=== LOCATION ===" in llm.last_system_prompt
    assert "=== NPC PROFILE ===" in llm.last_system_prompt
    assert "rag" in log_record
