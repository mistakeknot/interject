"""Tests for internal awareness signals."""

from pathlib import Path

from interject.awareness import InternalAwareness


def test_reads_active_plans(tmp_path: Path):
    plans_dir = tmp_path / "docs" / "plans"
    plans_dir.mkdir(parents=True)
    (plans_dir / "2026-02-15-plan.md").write_text(
        "# Internal Awareness Plan\n\n"
        "**Goal:** Add awareness scoring inputs\n",
        encoding="utf-8",
    )

    awareness = InternalAwareness(interverse_root=tmp_path)
    topics = awareness._topics_from_plans()

    assert "Internal Awareness Plan" in topics
    assert "Add awareness scoring inputs" in topics


def test_reads_brainstorms(tmp_path: Path):
    brainstorms_dir = tmp_path / "docs" / "brainstorms"
    brainstorms_dir.mkdir(parents=True)
    (brainstorms_dir / "2026-02-15-ideas.md").write_text(
        "# Discovery Scoring Brainstorm\n\nSome notes.\n",
        encoding="utf-8",
    )

    awareness = InternalAwareness(interverse_root=tmp_path)
    topics = awareness._topics_from_brainstorms()

    assert "Discovery Scoring Brainstorm" in topics


def test_no_plans_returns_empty(tmp_path: Path):
    (tmp_path / "docs").mkdir(parents=True)

    awareness = InternalAwareness(interverse_root=tmp_path)

    assert awareness._topics_from_plans() == []


def test_get_active_bead_topics():
    awareness = InternalAwareness()
    topics = awareness.get_bead_topics()

    assert isinstance(topics, list)


def test_get_suppression_topics():
    awareness = InternalAwareness()
    topics = awareness.get_suppression_topics()

    assert isinstance(topics, list)

