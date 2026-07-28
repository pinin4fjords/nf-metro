"""How a feature vector may be read: sentinels, derived terms, admissibility.

Every consumer of a vector produced by ``extract_features`` needs the same three
rules, and each one has been got wrong somewhere at least once:

1. **Two features carry a sentinel.** ``min_marker_gap`` and
   ``min_station_distance`` emit ``-1.0`` for "no such measurement exists".
   Read as the number -1 the sentinel is the tightest clearance in the corpus,
   which inverts the feature.
2. **A clearance cannot be read as a penalty directly.** More clearance is
   better and unbounded, so a term that pays for it is minimised by spreading
   the drawing out. It has to be turned into a saturating one-sided penalty
   first, and the sentinel state -- no foreign segment anywhere near a marker --
   is the *cleanest* state, so it scores zero penalty rather than the maximum.
3. **Not every feature can appear in a score that will be minimised.** A
   feature that is signed, or that improves as the drawing grows, makes the
   score unbounded below however its weight is fitted. ``ADMISSIBILITY`` gives
   every feature a verdict, so an objective assembled from it cannot silently
   include one that has no bound.

Held here rather than in ``extract_features`` because that module runs inside a
worktree parked at a historical SHA and must stay standalone; this one is
imported by today's consumers only.
"""

from __future__ import annotations

UNDEFINED = -1.0
"""``extract_features``'s "no such measurement exists" value."""

SENTINEL = frozenset({"min_marker_gap", "min_station_distance"})
"""Features that emit :data:`UNDEFINED`.

A map with no foreign line near any marker has no minimum gap, and a map with
fewer than two real stations has no minimum station distance.
"""

CROWDING_PITCH = 40.0
"""One lane pitch, mirroring ``nf_metro.layout.constants.Y_SPACING``.

Clearance beyond one pitch is room enough, so this is where
:func:`marker_crowding` saturates. Hardcoded to match the literal in
``extract_features``: a feature's meaning must not track a constant the engine
may retune, or a vector measured today would not be comparable with one
measured at an older revision.
"""


# --------------------------------------------------------------------------- #
# Sentinels
# --------------------------------------------------------------------------- #


def defined(key: str, value: float) -> float | None:
    """``value``, or ``None`` when it is this feature's undefined sentinel."""
    return None if key in SENTINEL and value == UNDEFINED else value


# --------------------------------------------------------------------------- #
# Derived terms
# --------------------------------------------------------------------------- #


def marker_crowding(gap: float | None) -> float:
    """How far the nearest foreign line intrudes on a marker, as a 0..1 fraction.

    One-sided and saturating, so the term penalises tight clearance without ever
    paying for loose clearance: a term that kept paying would be minimised by
    spreading the map out, which is the failure mode
    :data:`ADMISSIBILITY` exists to exclude.

    An absent measurement scores **zero** crowding. No foreign line coming near
    any marker is the cleanest state a map can be in, not the worst: 104 of the
    278 fixtures carrying a vector in the committed corpus are in it, and
    reading their ``-1.0`` as a gap would score every one of them as maximally
    crowded and drive a non-negative weight to penalise the fixtures with the
    best clearance in the corpus.
    """
    if gap is None or gap == UNDEFINED:
        return 0.0
    return min(1.0, max(0.0, (CROWDING_PITCH - gap) / CROWDING_PITCH))


DERIVED = {"marker_crowding": ("min_marker_gap", marker_crowding)}
"""Admissible terms computed from an inadmissible feature.

Keyed by the derived name, valued by the source feature and the transform. A
consumer that wants the source's signal in a minimisable score reaches for the
derived term instead of repairing the raw feature locally.
"""


def derived_values(vector: dict[str, float]) -> dict[str, float]:
    """The :data:`DERIVED` terms this vector supports."""
    return {
        name: transform(defined(source, vector[source]))
        for name, (source, transform) in DERIVED.items()
        if source in vector
    }


def readable(vector: dict[str, float]) -> dict[str, float | None]:
    """A vector with its sentinels masked and its derived terms added.

    The one way to turn what the extractor emitted into what a score may read.
    ``None`` marks a measurement that does not exist for this map, which a
    consumer must skip rather than substitute a number for.
    """
    out: dict[str, float | None] = {k: defined(k, v) for k, v in vector.items()}
    out.update(derived_values(vector))
    return out


# --------------------------------------------------------------------------- #
# Admissibility
# --------------------------------------------------------------------------- #

ADMISSIBLE = "admissible"
"""Non-negative on every layout, and larger means worse.

A non-negative weight on such a feature can only penalise, so a score built
from these alone is bounded below by zero.
"""

ANTITONE = "antitone"
"""Non-negative, but larger means *better*, so its useful weight is negative.

Unbounded above -- a clearance can always be widened -- so a negative weight
makes the score unbounded below. Only reachable through :data:`DERIVED`.
"""

SIGNED = "signed"
"""Takes either sign, so either sign of weight is unbounded below.

``aspect_log`` is ``log10(bbox_w / bbox_h)``: a negative weight buys an
arbitrarily tall drawing and a positive one an arbitrarily wide drawing. A
minimisable form would have to be one-sided, e.g. its magnitude.
"""

ADMISSIBILITY = {
    # Angle and turn quality: counts and sums of positive quantities.
    "bends_per_route": ADMISSIBLE,
    "corners_total": ADMISSIBLE,
    "max_bends_one_route": ADMISSIBLE,
    "turn_angle_per_route": ADMISSIBLE,
    "lone_diagonals": ADMISSIBLE,
    "lone_diagonals_per_route": ADMISSIBLE,
    "non_45_frac": ADMISSIBLE,
    "non_45_segments": ADMISSIBLE,
    "near_horizontal": ADMISSIBLE,
    "near_horizontal_frac": ADMISSIBLE,
    # Crossings and strikes: counts.
    "crossings": ADMISSIBLE,
    "crossings_per_route": ADMISSIBLE,
    "marker_strikes": ADMISSIBLE,
    "marker_strikes_per_station": ADMISSIBLE,
    # Extent and length. Admissible but unbounded above, so these are exactly
    # the terms whose weight sign decides whether a search can inflate the
    # drawing for free.
    "bbox_h": ADMISSIBLE,
    "bbox_w": ADMISSIBLE,
    "path_len_per_route": ADMISSIBLE,
    "path_len_per_station": ADMISSIBLE,
    "detour_max": ADMISSIBLE,
    "detour_mean": ADMISSIBLE,
    "lane_gap_excess": ADMISSIBLE,
    "exit_port_misalignment": ADMISSIBLE,
    # Size of the input, not of the drawing: identical on both sides of a pair
    # by construction, and not something a layout search can move at all.
    "n_stations": ADMISSIBLE,
    "n_routes": ADMISSIBLE,
    "n_sections": ADMISSIBLE,
    "n_ports": ADMISSIBLE,
    "stations_per_route": ADMISSIBLE,
    "ports_per_section": ADMISSIBLE,
    # Clearances: better when larger, and always wideable.
    "min_marker_gap": ANTITONE,
    "min_station_distance": ANTITONE,
    # Shape.
    "aspect_log": SIGNED,
    # Derived.
    "marker_crowding": ADMISSIBLE,
}
"""Verdict per feature name, covering everything a vector can contain.

Assembling a minimisable objective from anything but :data:`ADMISSIBLE` terms
leaves the score unbounded below, so this table is what
``check_objective_safety`` checks a candidate feature set against.
"""


BOUNDED_ABOVE = frozenset(
    {
        # Fractions of a total, and a saturating penalty, so all three are
        # confined to [0, 1] by construction.
        "non_45_frac",
        "near_horizontal_frac",
        "marker_crowding",
    }
)
"""Admissible features that are also bounded above.

The distinction matters because it decides how *much* of a box constraint the
safety property actually needs. A negative weight on a term confined to [0, 1]
can improve the score by at most that weight's magnitude, so the score stays
bounded below; a negative weight on a term that is unbounded above makes it
unbounded below. Only the latter has to be pinned, which is what
``fit_objective``'s ``safe_min`` arm tests: the weakest constraint that still
buys boundedness, so a collapse cannot be blamed on over-constraining.
"""


def must_be_non_negative(keys: object) -> list[str]:
    """The subset of ``keys`` whose weight has to be pinned for boundedness."""
    return [
        k for k in keys if ADMISSIBILITY.get(k) == ADMISSIBLE and k not in BOUNDED_ABOVE
    ]


def inadmissible(keys: object) -> dict[str, str]:
    """The keys that cannot appear in a minimisable score, and why.

    An unknown name is reported too: a feature with no verdict has not been
    reasoned about, which is not the same as being safe.
    """
    out = {}
    for key in keys:
        verdict = ADMISSIBILITY.get(key)
        if verdict is None:
            out[key] = "unclassified"
        elif verdict != ADMISSIBLE:
            out[key] = verdict
    return out


VERDICT_REASON = {
    ANTITONE: (
        "larger is better and always reachable, so its useful weight is "
        "negative and the score has no floor"
    ),
    SIGNED: "signed, so a weight of either sign can be driven without bound",
    "unclassified": "carries no admissibility verdict, so it is unexamined, not safe",
}
"""Why a non-:data:`ADMISSIBLE` feature cannot appear in a minimisable score."""


def unbounded_below(weights: dict[str, float]) -> dict[str, str]:
    """Every reason a score with these weights has no lower bound, by feature.

    Empty means the score is bounded below, which is the whole safety property:
    an objective a search can improve without limit degenerates every candidate
    it produces, rather than occasionally offering a poor one.
    """
    out = {
        key: VERDICT_REASON[verdict] for key, verdict in inadmissible(weights).items()
    }
    for key in must_be_non_negative(weights):
        if weights[key] < 0.0:
            out[key] = (
                f"weight {weights[key]:+.6f} rewards a feature that is unbounded "
                "above, so the score has no floor"
            )
    return out


def lower_bound(weights: dict[str, float]) -> float | None:
    """The greatest lower bound on the score, or ``None`` when there is none.

    Only a :data:`BOUNDED_ABOVE` term may carry a negative weight and still
    leave the score bounded, and it can subtract at most its weight's magnitude,
    since those terms are confined to ``[0, 1]``.
    """
    if unbounded_below(weights):
        return None
    return sum(w for w in weights.values() if w < 0.0)
