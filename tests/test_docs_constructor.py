from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.core
def test_docs_constructor_is_wired_into_mkdocs():
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Interactive Constructor: constructor.md" in mkdocs
    assert "Interactive Constructor: Интерактивный конструктор" in mkdocs


@pytest.mark.core
def test_docs_constructor_page_contains_interactive_builder_markers():
    en = (ROOT / "docs" / "constructor.md").read_text(encoding="utf-8")
    assert "interactive-builder" in en
    assert "p2a-code" in en
    assert "p2a-install" in en
    assert "Experiment" in en

    ru = (ROOT / "docs" / "constructor.ru.md").read_text(encoding="utf-8")
    assert "interactive-builder" in ru
    assert "p2a-code" in ru
    assert "p2a-install" in ru
    assert "Experiment" in ru
