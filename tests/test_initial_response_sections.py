from agentic.shortlist import present_shortlist
from knowledge_engine.loader import load_agent_vocabulary
from tests.test_shortlist_presentation import context


def test_explicit_sections_surround_tables(context):
    _, _, view, pack = context
    text = "[[MARKET_COMMENTARY]]\nMarket context. Selection implications.\n\n[[TRADE_NOTES]]\nRisk and trade-offs."
    reply = present_shortlist(text, pack, view, automatic=True)
    assert reply.index("Market state") < reply.index("Market context.")
    assert reply.index("Market context.") < reply.index("Structure Fit")
    assert reply.index("Structure Fit") < reply.index("Ranked packages")
    assert reply.index("Ranked packages") < reply.index("Risk and trade-offs.")
    assert reply.endswith(load_agent_vocabulary()["chat_invitation"])
    assert "[[MARKET_COMMENTARY]]" not in reply and "[[TRADE_NOTES]]" not in reply


def test_missing_delimiters_move_extra_paragraphs_below_tables(context):
    _, _, view, pack = context
    reply = present_shortlist("Market context.\n\nRisk note.\n\nFurther qualification.", pack, view, automatic=True)
    assert reply.index("Market context.") < reply.index("Structure Fit")
    assert reply.index("Ranked packages") < reply.index("Risk note.") < reply.index("Further qualification.")


def test_duplicate_model_table_is_removed_without_losing_notes(context):
    _, _, view, pack = context
    text = "[[MARKET_COMMENTARY]]Market context.\n\n| Fake | Amount |\n| --- | --- |\n| Wrong | 123 |\n\n[[TRADE_NOTES]]Risk note."
    reply = present_shortlist(text, pack, view, automatic=True)
    assert "Wrong" not in reply
    assert reply.index("Ranked packages") < reply.index("Risk note.")


def test_ordinary_followup_is_unchanged(context):
    _, _, view, pack = context
    assert present_shortlist("A follow-up answer.", pack, view) == "A follow-up answer."


def test_comparison_stays_compact_without_initial_sections(context):
    _, _, view, pack = context
    reply = present_shortlist("Comparison explanation.", pack, view, automatic=True, ranks=[1, 3])
    assert "Market state" not in reply and "Structure Fit" not in reply
    assert reply.endswith("Comparison explanation.")
