"""Corpus-wide equivalence lock: the render-diff's learned-score geometry
source against extract_features.py's own routing pass.

``datasets/layout_preferences/scripts/scored_objective.py`` reads a render's
actual drawn geometry (offsets applied via ``drawn_polylines``);
``extract_features.py`` -- what produced the fitted weights' training data --
resolves its own routing pass. The two are meant to measure the same thing by
a different decomposition, not by construction, so this locks the gap between
them: a future engine change that quietly decalibrates the reported score
against the weights it is scored with should fail here, not go unnoticed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "layout_preferences"
    / "scripts"
)
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import scored_objective  # noqa: E402
from check_feature_parity import _fixtures, legacy_features, live_features  # noqa: E402

from nf_metro.layout import compute_layout  # noqa: E402
from nf_metro.parser import parse_metro_mermaid  # noqa: E402

# Comfortably above the measured max (1.42 across the whole corpus) so a real
# engine regression trips this before it silently decalibrates the shadow score.
MAX_WEIGHTED_DIVERGENCE = 3.0

# The two heaviest-weighted iter2 features (12.3 and 3.0 respectively) must
# match exactly: together they carry most of the objective's weight mass, and
# any divergence here means the two geometry sources disagree on something a
# human would actually see (a bend either happened or it didn't).
ZERO_TOLERANCE_FEATURES = ("lone_diagonals_per_route", "bends_per_route")


def test_learned_score_geometry_sources_agree_within_epsilon() -> None:
    weights = scored_objective.load_weights()["weights"]
    keys = scored_objective.feature_keys()

    worst = 0.0
    worst_stem = None
    zero_tol_divergent: list[str] = []

    for mmd in _fixtures():
        try:
            graph = parse_metro_mermaid(mmd.read_text())
            compute_layout(graph)
            legacy = legacy_features(graph)
            live = live_features(graph)
        except Exception:  # noqa: BLE001 - deliberately-invalid fixtures exist
            continue

        delta = sum(
            weights[k] * (legacy[k] - live[k])
            for k in keys
            if legacy.get(k) is not None and live.get(k) is not None
        )
        if abs(delta) > worst:
            worst, worst_stem = abs(delta), mmd.stem

        for k in ZERO_TOLERANCE_FEATURES:
            a, b = legacy.get(k), live.get(k)
            if a is not None and b is not None and abs(a - b) > 1e-6:
                zero_tol_divergent.append(f"{mmd.stem}:{k} legacy={a} live={b}")

    assert not zero_tol_divergent, (
        "the dominant-weight features disagree between the two geometry "
        f"sources: {zero_tol_divergent}"
    )
    assert worst < MAX_WEIGHTED_DIVERGENCE, (
        f"weighted score divergence {worst:.3f} on {worst_stem!r} exceeds the "
        f"{MAX_WEIGHTED_DIVERGENCE} epsilon locked by #1587 -- regenerate "
        "datasets/layout_preferences/feature_parity_report.txt and "
        "investigate before trusting the render-diff's learned-score column"
    )
