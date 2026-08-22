"""
route_after_critic is the one piece of "real" control-flow logic in the
whole graph - it's a pure function, so it gets a plain table-driven test
with no mocking required.
"""

import pytest

from graph import route_after_critic, SCORE_THRESHOLD


@pytest.mark.parametrize(
    "score, revision_count, max_revisions, expected",
    [
        (9.0, 0, 2, "editor"),                      # comfortably above threshold
        (SCORE_THRESHOLD, 1, 2, "editor"),           # exactly at threshold approves
        (7.4, 0, 2, "writer"),                       # just below threshold, budget left
        (2.0, 1, 2, "writer"),                       # low score, revisions remain
        (2.0, 2, 2, "editor"),                       # low score but cap hit -> forced through
        (0.0, 0, 0, "editor"),                       # max_revisions=0 -> never loop
    ],
)
def test_route_after_critic(score, revision_count, max_revisions, expected):
    state = {
        "critique_score": score,
        "revision_count": revision_count,
        "max_revisions": max_revisions,
    }
    assert route_after_critic(state) == expected


def test_route_after_critic_defaults_missing_fields_safely():
    # A state with no score/revision info yet shouldn't crash - it should
    # fall back to "not good enough, but no revisions taken yet" -> writer.
    assert route_after_critic({}) == "writer"
