from conftest import MergingTokenizer

from localdecision.backends.mock import CharTokenizer
from localdecision.prompts import Prompter, build_spec, render_question, render_state
from localdecision.schema import Choice, Noul, Score


def test_prompter_verifies_the_split_and_letters():
    p = Prompter(CharTokenizer())
    assert p.template_ok and p.split_ok
    assert p.letter_ids[:3] == [ord("A"), ord("B"), ord("C")]
    assert p.max_letters == 26
    state, block = {"a": 1}, "Question: q?\n\nOptions:\nA. x\nB. y"
    assert p.prefix_text(state) + p.suffix_text(block) == p.full_text(state, block)


def test_non_additive_tokenizer_is_detected():
    assert not Prompter(MergingTokenizer()).split_ok


def test_state_rendering_keeps_key_order_and_unicode():
    assert render_state({"z": 1, "a": "è"}) == '<state>\n{\n  "z": 1,\n  "a": "è"\n}\n</state>\n\n'
    assert render_state("  plain text  ") == "<state>\nplain text\n</state>\n\n"


def test_specs_for_each_primitive():
    noul = build_spec("u", Noul("Urgent?", true="needs action today"))
    assert noul.keys == ("true", "false") and noul.lines == ("Yes: needs action today", "No")
    choice = build_spec("c", Choice("Team?", {"billing": "Payments", "sales": None}))
    assert choice.lines == ("billing: Payments", "sales")
    score = build_spec("s", Score("How angry?", ["Calm", {"level": "angry"}]))
    assert score.keys == ("0", "1")
    assert score.legend == {"0": "Calm", "1": '{"level": "angry"}'}


def test_score_views_state_the_direction_of_the_scale():
    spec = build_spec("s", Score("How angry?", ["Calm", "Annoyed", "Furious"]))
    forward = render_question(spec, [0, 1, 2])
    backward = render_question(spec, [2, 1, 0])
    assert "from level 0, the lowest, to level 2" in forward and "A. Level 0: Calm" in forward
    assert (
        "from level 2, the highest, down to level 0" in backward
        and "A. Level 2: Furious" in backward
    )
