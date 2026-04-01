from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.core
def test_docs_constructor_is_wired_into_mkdocs():
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Interactive Constructor: constructor.html" in mkdocs
    assert "Interactive Constructor: Интерактивный конструктор" in mkdocs


@pytest.mark.core
def test_docs_constructor_page_contains_interactive_builder_markers():
    html = (ROOT / "docs" / "constructor.html").read_text(encoding="utf-8")
    assert "Интерактивный конструктор цепочек plantain2asr" in html
    assert "preset-choices" in html
    assert "generated-code" in html
    assert "install-command" in html
    assert "Experiment.prepare_thesis_tables()" in html
