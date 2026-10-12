from streamlit.testing.v1 import AppTest

from interface.kelly_sizing_ui import meaning_banner
from knowledge_engine import ui_labels as UL


def linked_summary_app():
    from types import SimpleNamespace
    from interface.kelly_v2.app import render_trade_rec_summary

    render_trade_rec_summary(SimpleNamespace(pair="USDBRL", horizon_days=90,
        structure_label="1x2", variant_label="test", entry_spot=5.0, forward=5.2,
        sigma=0.15, max_loss_pct=0.0017))


def test_linked_kelly_reference_is_not_called_max_loss():
    app = AppTest.from_function(linked_summary_app).run()
    assert not app.exception
    assert any(item.label == "Sizing loss reference" for item in app.metric)
    assert not any(item.label == "Max loss" for item in app.metric)
    assert any("not its contractual maximum loss" in item.value for item in app.caption)


def test_shared_labels_do_not_promise_equal_maximum_loss():
    assert "Sizing exposure" in UL.label("kelly_risk")
    assert "not a maximum-loss bound" in UL.tip("kelly_risk")
    assert "does not equalise or guarantee maximum losses" in meaning_banner("fixed_loss")
