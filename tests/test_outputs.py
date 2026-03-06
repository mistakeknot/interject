"""Tests for output pipeline docs and routing behavior."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from interject.outputs import OutputPipeline


@pytest.fixture
def base_discovery() -> dict:
    return {
        "id": "ij-github-test-1",
        "title": "Fast MCP Planning Toolkit",
        "summary": "A toolkit for drafting and routing implementation plans.",
        "source": "github",
        "url": "https://github.com/example/toolkit",
        "relevance_score": 0.91,
        "confidence_tier": "high",
        "discovered_at": "2026-02-15T12:00:00Z",
        "raw_metadata": json.dumps(
            {"stars": 1420, "language": "Python", "topics": ["mcp", "agents"]}
        ),
    }


@pytest.fixture
def mock_bd_cli():
    with patch("interject.outputs.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["bd", "create"],
            returncode=0,
            stdout="Created issue: iv-123\n",
            stderr="",
        )
        yield run_mock


@pytest.fixture
def mock_cli():
    """Mock subprocess.run for both bd and ic CLI calls."""
    with patch("interject.outputs.subprocess.run") as run_mock:
        def side_effect(cmd, **kwargs):
            if cmd[0] == "ic":
                if "submit" in cmd:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout="disc-test-123\n", stderr=""
                    )
                if "promote" in cmd:
                    return subprocess.CompletedProcess(
                        args=cmd, returncode=0, stdout="promoted\n", stderr=""
                    )
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout="Created issue: iv-123\n", stderr=""
            )
        run_mock.side_effect = side_effect
        yield run_mock


def test_high_tier_creates_brainstorm(
    tmp_path: Path, base_discovery: dict, mock_cli
) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)

    result = pipeline.process(base_discovery, "high")

    assert result["tier"] == "high"
    assert result["bead_id"] == "iv-123"
    assert "briefing_path" in result
    assert "brainstorm_path" in result
    assert "plan_path" not in result
    assert result.get("kernel_discovery_id") == "disc-test-123"

    brainstorm_path = Path(result["brainstorm_path"])
    assert brainstorm_path.exists()
    assert brainstorm_path.parent == tmp_path / "brainstorms"
    assert brainstorm_path.name.endswith("-brainstorm.md")

    content = brainstorm_path.read_text()
    assert "## What We're Building" in content
    assert "## Why This Approach" in content


def test_medium_tier_creates_briefing_only(
    tmp_path: Path, base_discovery: dict, mock_cli
) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)
    medium_discovery = dict(base_discovery)
    medium_discovery["confidence_tier"] = "medium"

    result = pipeline.process(medium_discovery, "medium")

    assert result["tier"] == "medium"
    assert result["bead_id"] == "iv-123"
    assert "briefing_path" in result
    assert "brainstorm_path" not in result
    assert not (tmp_path / "brainstorms").exists()
    assert result.get("kernel_discovery_id") == "disc-test-123"

    # Verify bead was created with P4 and pending_triage label
    bd_calls = [c for c in mock_cli.call_args_list if c[0][0][0] == "bd"]
    assert len(bd_calls) >= 1
    bd_args = bd_calls[0][0][0]
    assert "--priority=4" in bd_args
    assert any("pending_triage" in a for a in bd_args)

    briefing_path = Path(result["briefing_path"])
    assert briefing_path.exists()
    assert briefing_path.parent == tmp_path / "research"


def test_low_tier_no_output(tmp_path: Path, base_discovery: dict, mock_cli) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)

    result = pipeline.process(base_discovery, "low")

    assert result["tier"] == "low"
    assert result.get("kernel_discovery_id") == "disc-test-123"
    assert "bead_id" not in result
    assert list(tmp_path.rglob("*")) == []

    # ic submit IS called (kernel record), but bd create is NOT called
    ic_calls = [c for c in mock_cli.call_args_list if c[0][0][0] == "ic"]
    bd_calls = [c for c in mock_cli.call_args_list if c[0][0][0] == "bd"]
    assert len(ic_calls) >= 1
    assert len(bd_calls) == 0


def test_medium_tier_creates_p4_bead_with_pending_triage(
    tmp_path: Path, base_discovery: dict, mock_cli
) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)
    medium_discovery = dict(base_discovery)
    medium_discovery["confidence_tier"] = "medium"

    result = pipeline.process(medium_discovery, "medium", discovery_id="ij-test-1")

    assert result["tier"] == "medium"
    assert result["bead_id"] == "iv-123"
    assert "briefing_path" in result
    assert "brainstorm_path" not in result
    assert result.get("kernel_discovery_id") == "disc-test-123"

    # Verify bd create was called with P4 and pending_triage
    bd_calls = [c for c in mock_cli.call_args_list if c[0][0][0] == "bd"]
    assert len(bd_calls) >= 1
    bd_args = bd_calls[0][0][0]
    assert "--priority=4" in bd_args
    assert any("pending_triage" in a for a in bd_args)


def test_generate_digest(tmp_path: Path) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)
    discoveries = [
        {
            "title": "Paper A",
            "summary": "First arXiv discovery.",
            "source": "arxiv",
            "url": "https://arxiv.org/abs/1234.5678",
            "relevance_score": 0.88,
            "confidence_tier": "high",
        },
        {
            "title": "Repo B",
            "summary": "Useful GitHub repository.",
            "source": "github",
            "url": "https://github.com/example/repo-b",
            "relevance_score": 0.74,
            "confidence_tier": "medium",
        },
        {
            "title": "Paper C",
            "summary": "Second arXiv discovery.",
            "source": "arxiv",
            "url": "https://arxiv.org/abs/9999.0000",
            "relevance_score": 0.62,
            "confidence_tier": "low",
        },
    ]

    digest_path = pipeline.generate_digest(discoveries)

    assert digest_path.exists()
    assert digest_path.parent == tmp_path / "research"
    assert digest_path.name.endswith("-interject-digest.md")

    content = digest_path.read_text()
    assert "## Summary Table" in content
    assert "| arxiv | 2 |" in content
    assert "| github | 1 |" in content
    assert "### arxiv (2)" in content
    assert "### github (1)" in content


def test_submit_to_kernel(tmp_path: Path, base_discovery: dict) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)

    with patch("interject.outputs.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["ic"], returncode=0, stdout="disc-123\n", stderr=""
        )
        result = pipeline._submit_to_kernel(base_discovery)

    assert result == "disc-123"
    call_args = run_mock.call_args[0][0]
    assert call_args[0:3] == ["ic", "discovery", "submit"]
    assert any("--source=github" in a for a in call_args)
    assert any("--title=" in a for a in call_args)
    assert any("--score=" in a for a in call_args)


def test_submit_to_kernel_ic_not_found(tmp_path: Path, base_discovery: dict) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)

    with patch("interject.outputs.subprocess.run", side_effect=FileNotFoundError("ic")):
        result = pipeline._submit_to_kernel(base_discovery)

    assert result is None


def test_promote_in_kernel(tmp_path: Path) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)

    with patch("interject.outputs.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["ic"], returncode=0, stdout="promoted\n", stderr=""
        )
        result = pipeline._promote_in_kernel("disc-123", "iv-456")

    assert result is True
    call_args = run_mock.call_args[0][0]
    assert call_args == ["ic", "discovery", "promote", "disc-123", "--bead-id=iv-456"]


def test_promote_in_kernel_failure(tmp_path: Path) -> None:
    pipeline = OutputPipeline(docs_root=tmp_path, interverse_root=tmp_path)

    with patch("interject.outputs.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["ic"], returncode=1, stdout="", stderr="not found"
        )
        result = pipeline._promote_in_kernel("disc-123", "iv-456")

    assert result is False
