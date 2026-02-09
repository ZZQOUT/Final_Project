import re
from pathlib import Path
import pytest

from rpg_story.models.world import WorldBibleRules, LocationSpec, NPCProfile, WorldSpec, GameState
from rpg_story.persistence.store import (
    generate_session_id,
    save_state,
    load_state,
    append_turn_log,
    read_turn_logs,
    append_story_summary,
    read_story_summaries,
    validate_session_id,
)


def make_min_world() -> WorldSpec:
    locs = [
        LocationSpec(
            location_id="loc_001",
            name="Town",
            kind="town",
            description="A small town.",
            connected_to=["loc_002"],
        ),
        LocationSpec(
            location_id="loc_002",
            name="Forest",
            kind="forest",
            description="A dark forest.",
            connected_to=["loc_001"],
        ),
    ]
    npcs = [
        NPCProfile(
            npc_id="npc_001",
            name="Ala",
            profession="Merchant",
            traits=["curious"],
            goals=["sell goods"],
            starting_location="loc_001",
            obedience_level=0.5,
            stubbornness=0.5,
            risk_tolerance=0.5,
            disposition_to_player=0,
            refusal_style="polite",
        )
    ]
    bible = WorldBibleRules(
        tech_level="medieval",
        magic_rules="low",
        tone="adventurous",
    )
    return WorldSpec(
        world_id="world_001",
        title="Test World",
        world_bible=bible,
        locations=locs,
        npcs=npcs,
        starting_location="loc_001",
        starting_hook="You arrive.",
        initial_quest="Find the relic.",
    )


def make_state(session_id: str) -> GameState:
    world = make_min_world()
    return GameState(
        session_id=session_id,
        created_at="2025-01-01T00:00:00Z",
        world=world,
        player_location="loc_001",
        npc_locations={"npc_001": "loc_001"},
    )


def test_save_load_roundtrip(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    session_id = generate_session_id()
    state = make_state(session_id)
    save_state(session_id, state, sessions_root)
    loaded = load_state(session_id, sessions_root)
    assert loaded.session_id == state.session_id
    assert loaded.player_location == state.player_location
    assert loaded.npc_locations == state.npc_locations
    assert loaded.world.world_id == state.world.world_id


def test_append_turn_log_multiple_lines(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    session_id = generate_session_id()
    for i in range(3):
        append_turn_log(session_id, {"turn_id": i}, sessions_root)
    path = sessions_root / session_id / "turns.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    records = read_turn_logs(session_id, sessions_root)
    assert [r["turn_id"] for r in records] == [0, 1, 2]


def test_load_missing_raises(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    session_id = "sess_missing"
    with pytest.raises(FileNotFoundError):
        load_state(session_id, sessions_root)


def test_read_turn_logs_skips_corrupt_lines(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    session_id = generate_session_id()
    log_path = sessions_root / session_id / "turns.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "{\"turn_id\": 1}\n" + "{bad json}\n" + "{\"turn_id\": 2}\n",
        encoding="utf-8",
    )
    records = read_turn_logs(session_id, sessions_root)
    assert [r["turn_id"] for r in records] == [1, 2]


def test_session_id_format():
    session_id = generate_session_id()
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{6,8}$", session_id)


def test_invalid_session_id_rejected():
    bad_ids = ["../x", "..\\x", "a/b", "a\\b", "", "a..b", "a b", "💥"]
    for sid in bad_ids:
        with pytest.raises(ValueError):
            validate_session_id(sid)


def test_story_summary_roundtrip(tmp_path: Path):
    sessions_root = tmp_path / "sessions"
    record_a = {"session_id": "s1", "world_title": "A", "summary": "first"}
    record_b = {"session_id": "s2", "world_title": "B", "summary": "second"}
    append_story_summary(record_a, sessions_root)
    append_story_summary(record_b, sessions_root)
    got = read_story_summaries(sessions_root, limit=10)
    assert len(got) == 2
    # Read order is newest first.
    assert got[0]["session_id"] == "s2"
    assert got[1]["session_id"] == "s1"
