from rpg_story.world.term_guard import DEFAULT_ANACHRONISM_TERMS, detect_first_mention


def test_first_mention_medieval_flags_new_terms():
    player_text = "Hello there."
    npc_texts = ["We have wifi in the tavern."]
    new_terms = detect_first_mention(player_text, npc_texts, DEFAULT_ANACHRONISM_TERMS)
    assert new_terms == {"wifi"}


def test_first_mention_medieval_allows_if_player_said_it():
    player_text = "What is wifi?"
    npc_texts = ["I don't know wifi."]
    new_terms = detect_first_mention(player_text, npc_texts, DEFAULT_ANACHRONISM_TERMS)
    assert new_terms == set()


def test_first_mention_modern_not_enforced():
    tech_level = "modern"
    player_text = "Hello."
    npc_texts = ["We have wifi in the office."]
    if tech_level != "medieval":
        new_terms = set()
    else:
        new_terms = detect_first_mention(player_text, npc_texts, DEFAULT_ANACHRONISM_TERMS)
    assert new_terms == set()
