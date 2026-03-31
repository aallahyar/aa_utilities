"""
repel_text — Non-overlapping text label placement for matplotlib.

Greedy candidate-based placement with limited backtracking.
Designed for volcano plots and scatter plots with <100 annotations.

Usage:
    from repel_text import repel_text
    sc = ax.scatter(all_x, all_y)
    repel_text(ax, x, y, texts, avoid=[sc])
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Sequence

from matplotlib.artist import Artist
from matplotlib.axes import Axes


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

@dataclass
class _Label:
    idx: int
    x_disp: float
    y_disp: float
    width: float
    height: float
    text: str
    bbox: np.ndarray | None = None   # [x0, y0, x1, y1] display coords
    cand_rank: int = -1


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _generate_candidates(
    cx: float,
    cy: float,
    w: float,
    h: float,
    min_dist: float,
    max_dist: float,
    n: int,
) -> np.ndarray:
    """Return up to *n* candidate bboxes ``[x0, y0, x1, y1]`` (display coords).

    Candidates are placed on concentric rings around *(cx, cy)* at 16 evenly
    spaced angles, then filtered so the nearest edge is at least *min_dist*
    from the anchor, and sorted closest-first.
    """
    n_angles = 16
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Generate extra rings to compensate for the min-distance filter.
    n_rings = max(1, int(2.0 * n / n_angles))
    half_diag = math.hypot(w, h) / 2
    radii = np.linspace(
        min_dist + half_diag * 0.5,
        max_dist + half_diag,
        n_rings,
    )

    r_grid, a_grid = np.meshgrid(radii, angles)
    dx = r_grid.ravel() * np.cos(a_grid.ravel())
    dy = r_grid.ravel() * np.sin(a_grid.ravel())

    x0 = cx + dx - w / 2
    y0 = cy + dy - h / 2
    candidates = np.column_stack([x0, y0, x0 + w, y0 + h])

    # Keep only candidates whose nearest edge is >= min_dist from anchor.
    dx_edge = np.maximum(0, np.maximum(candidates[:, 0] - cx, cx - candidates[:, 2]))
    dy_edge = np.maximum(0, np.maximum(candidates[:, 1] - cy, cy - candidates[:, 3]))
    edge_dist = np.sqrt(dx_edge**2 + dy_edge**2)
    candidates = candidates[edge_dist >= min_dist]

    # Sort by distance from anchor to candidate centre (closest first).
    centres = (candidates[:, :2] + candidates[:, 2:]) / 2
    dist_sq = (centres[:, 0] - cx) ** 2 + (centres[:, 1] - cy) ** 2
    candidates = candidates[np.argsort(dist_sq)]

    return candidates[:n]


# ---------------------------------------------------------------------------
# Overlap helpers
# ---------------------------------------------------------------------------

def _boxes_overlap(a: np.ndarray, bs: np.ndarray, margin: float) -> np.ndarray:
    """Return bool array — True where box *a* (4,) overlaps a row in *bs* (N,4)."""
    return (
        (a[0] - margin < bs[:, 2] + margin)
        & (a[2] + margin > bs[:, 0] - margin)
        & (a[1] - margin < bs[:, 3] + margin)
        & (a[3] + margin > bs[:, 1] - margin)
    )


def _box_hits_points(box: np.ndarray, pts: np.ndarray | None, margin: float) -> bool:
    """True if *box* (4,) overlaps any point in *pts* (N,2)."""
    if pts is None or len(pts) == 0:
        return False
    return bool(np.any(
        (box[0] - margin < pts[:, 0])
        & (box[2] + margin > pts[:, 0])
        & (box[1] - margin < pts[:, 1])
        & (box[3] + margin > pts[:, 1])
    ))


def _in_bounds(box: np.ndarray, xlim: tuple, ylim: tuple) -> bool:
    return (
        box[0] >= xlim[0]
        and box[2] <= xlim[1]
        and box[1] >= ylim[0]
        and box[3] <= ylim[1]
    )


def _segments_cross(a0, a1, b0, b1) -> bool:
    """True if segment a0->a1 properly crosses segment b0->b1."""
    d1 = (b1[0] - b0[0]) * (a0[1] - b0[1]) - (b1[1] - b0[1]) * (a0[0] - b0[0])
    d2 = (b1[0] - b0[0]) * (a1[1] - b0[1]) - (b1[1] - b0[1]) * (a1[0] - b0[0])
    d3 = (a1[0] - a0[0]) * (b0[1] - a0[1]) - (a1[1] - a0[1]) * (b0[0] - a0[0])
    d4 = (a1[0] - a0[0]) * (b1[1] - a0[1]) - (a1[1] - a0[1]) * (b1[0] - a0[0])
    return d1 * d2 < 0 and d3 * d4 < 0


def _arrow_crosses_any(center, anchor, arrows) -> bool:
    """True if segment center->anchor crosses any existing arrow."""
    for a_start, a_end in arrows:
        if _segments_cross(center, anchor, a_start, a_end):
            return True
    return False


def _segment_hits_box(p0, p1, box) -> bool:
    """True if segment p0->p1 passes through axis-aligned box [x0,y0,x1,y1]."""
    x0, y0, x1, y1 = box
    # Check all four edges of the box against the segment.
    edges = [
        ((x0, y0), (x1, y0)),  # bottom
        ((x1, y0), (x1, y1)),  # right
        ((x0, y1), (x1, y1)),  # top
        ((x0, y0), (x0, y1)),  # left
    ]
    for e0, e1 in edges:
        if _segments_cross(p0, p1, e0, e1):
            return True
    return False


def _arrow_hits_any_box(center, anchor, placed) -> bool:
    """True if segment center->anchor passes through any placed text box."""
    for box in placed:
        if _segment_hits_box(center, anchor, box):
            return True
    return False


def _extract_obstacles(artists, renderer):
    """Extract point and box obstacles from matplotlib artists."""
    from matplotlib.collections import PathCollection

    points = []
    boxes = []
    for artist in artists:
        if isinstance(artist, PathCollection):
            offsets = artist.get_offsets()
            pts_disp = artist.get_offset_transform().transform(offsets)
            # Convert marker sizes (area in points²) to display-coord bboxes.
            sizes = artist.get_sizes()  # area in points²
            if len(sizes) > 0:
                dpi_scale = renderer.points_to_pixels(1.0)
                radii = np.sqrt(sizes) / 2 * dpi_scale  # half-width in display units
                if len(radii) == 1:
                    radii = np.full(len(pts_disp), radii[0])
                for pt, r in zip(pts_disp, radii):
                    boxes.append([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r])
            else:
                points.append(pts_disp)
        else:
            try:
                bb = artist.get_window_extent(renderer)
                boxes.append([bb.x0, bb.y0, bb.x1, bb.y1])
            except Exception:
                pass

    pt_array = np.vstack(points) if points else np.empty((0, 2))
    box_array = np.array(boxes).reshape(-1, 4) if boxes else np.empty((0, 4))
    return pt_array, box_array


# ---------------------------------------------------------------------------
# Core placement
# ---------------------------------------------------------------------------

def _first_valid(
    cands: np.ndarray,
    placed: np.ndarray,
    scatter: np.ndarray | None,
    margin: float,
    xlim: tuple,
    ylim: tuple,
    anchor: tuple | None = None,
    arrows: list | None = None,
    n_fixed: int = 0,
    start: int = 0,
) -> int:
    """Index of the first non-overlapping, non-crossing candidate, or ``-1``."""
    for i in range(start, len(cands)):
        c = cands[i]
        if not _in_bounds(c, xlim, ylim):
            continue
        if _box_hits_points(c, scatter, margin):
            continue
        if len(placed) > 0 and np.any(_boxes_overlap(c, placed, margin)):
            continue
        if anchor is not None:
            center = ((c[0] + c[2]) / 2, (c[1] + c[3]) / 2)
            if arrows and _arrow_crosses_any(center, anchor, arrows):
                continue
            # Only test against placed *text* boxes (skip obstacle boxes).
            text_boxes = placed[n_fixed:]
            if len(text_boxes) > 0 and _arrow_hits_any_box(center, anchor, text_boxes):
                continue
        return i
    return -1


def _place_labels(
    labels: list[_Label],
    scatter: np.ndarray | None,
    obstacle_boxes: np.ndarray | None,
    margin: float,
    min_dist: float,
    max_dist: float,
    n_cands: int,
    xlim: tuple,
    ylim: tuple,
    max_bt: int,
) -> None:
    """Greedy placement with single-swap backtracking. Modifies *labels* in place."""

    all_cands = [
        _generate_candidates(
            lb.x_disp, lb.y_disp, lb.width, lb.height,
            min_dist, max_dist, n_cands,
        )
        for lb in labels
    ]

    # Seed with immovable obstacle boxes so _first_valid avoids them.
    if obstacle_boxes is not None and len(obstacle_boxes) > 0:
        placed = obstacle_boxes.copy()
        n_fixed = len(obstacle_boxes)
    else:
        placed = np.empty((0, 4))
        n_fixed = 0

    placed_order: list[int] = []      # label indices in placement order
    arrows: list[tuple] = []          # (center, anchor) display-coord segments

    def _anchor(i):
        return (labels[i].x_disp, labels[i].y_disp)

    def _center(box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    for i, lb in enumerate(labels):
        rank = _first_valid(
            all_cands[i], placed, scatter, margin, xlim, ylim,
            anchor=_anchor(i), arrows=arrows, n_fixed=n_fixed,
        )

        if rank >= 0:
            lb.bbox = all_cands[i][rank]
            lb.cand_rank = rank
            placed = np.vstack([placed, lb.bbox])
            placed_order.append(i)
            arrows.append((_center(lb.bbox), _anchor(i)))
            continue

        # --- backtracking: try displacing one recently-placed label ----------
        resolved = False
        for bt in range(min(max_bt, len(placed_order))):
            j_slot = len(placed_order) - 1 - bt
            j = placed_order[j_slot]

            tmp_placed = np.delete(placed, n_fixed + j_slot, axis=0)
            tmp_arrows = arrows[:j_slot] + arrows[j_slot + 1:]

            # Can *i* be placed without *j*?
            ri = _first_valid(
                all_cands[i], tmp_placed, scatter, margin, xlim, ylim,
                anchor=_anchor(i), arrows=tmp_arrows, n_fixed=n_fixed,
            )
            if ri < 0:
                continue

            # Can *j* be re-placed after *i*?
            tmp2 = np.vstack([tmp_placed, all_cands[i][ri]])
            arrow_i = (_center(all_cands[i][ri]), _anchor(i))
            tmp_arrows2 = tmp_arrows + [arrow_i]
            rj = _first_valid(
                all_cands[j], tmp2, scatter, margin, xlim, ylim,
                anchor=_anchor(j), arrows=tmp_arrows2, n_fixed=n_fixed,
            )
            if rj < 0:
                continue

            # Accept the swap.
            lb.bbox = all_cands[i][ri]
            lb.cand_rank = ri
            labels[j].bbox = all_cands[j][rj]
            labels[j].cand_rank = rj
            placed = np.vstack([tmp2, all_cands[j][rj]])
            placed_order.pop(j_slot)
            arrows.pop(j_slot)
            placed_order.extend([i, j])
            arrows.append(arrow_i)
            arrows.append((_center(all_cands[j][rj]), _anchor(j)))
            resolved = True
            break

        if not resolved:
            # Fallback: closest in-bounds candidate (may overlap other labels).
            for k in range(len(all_cands[i])):
                if _in_bounds(all_cands[i][k], xlim, ylim):
                    lb.bbox = all_cands[i][k]
                    lb.cand_rank = k
                    break
            else:
                lb.bbox = all_cands[i][0]
                lb.cand_rank = 0
            placed = np.vstack([placed, lb.bbox])
            placed_order.append(i)
            arrows.append((_center(lb.bbox), _anchor(i)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def repel_text(
    ax: Axes,
    x: Sequence[float],
    y: Sequence[float],
    texts: Sequence[str],
    avoid: Sequence[Artist] | None = None,
    margin: float = 4.0,
    min_distance: float = 10.0,
    max_distance: float = 80.0,
    n_candidates: int = 200,
    draw_arrows: bool = True,
    max_backtrack: int = 5,
    text_kwargs: dict | None = None,
    arrow_kwargs: dict | None = None,
) -> list:
    """Place non-overlapping text labels near their anchor points.

    Labels are placed in the order given — put higher-priority labels first.
    Call **after** setting axis limits and **before** ``savefig`` / ``show``.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    x, y : array-like
        Anchor points for the labels (data coordinates).
    texts : list of str
        One label string per anchor point.
    avoid : list of Artist, optional
        Matplotlib artists whose areas the labels must not overlap.
        ``PathCollection`` (scatter) artists are decomposed into individual
        points; other artists are treated as rectangular bounding boxes.
        The anchor points *(x, y)* are always avoided automatically.
    margin : float
        Minimum gap between labels / points, in display points.
    min_distance : float
        Minimum distance from anchor to the nearest label edge (display pts).
    max_distance : float
        Maximum search radius for placement (display pts).
    n_candidates : int
        Candidate positions evaluated per label.
    draw_arrows : bool
        Draw connector lines from each label to its anchor.
    max_backtrack : int
        How many already-placed labels may be displaced when a new label
        cannot be placed.
    text_kwargs : dict, optional
        Forwarded to ``ax.annotate`` (e.g. *fontsize*, *color*, …).
    arrow_kwargs : dict, optional
        Merged into the *arrowprops* dict (e.g. *arrowstyle*, *color*, …).

    Returns
    -------
    list of matplotlib.text.Annotation
    """
    text_kwargs = dict(text_kwargs) if text_kwargs else {}
    arrow_kwargs = dict(arrow_kwargs) if arrow_kwargs else {}

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) != len(texts):
        raise ValueError("x, y, and texts must have the same length")

    fig = ax.get_figure()
    try:
        fig.draw_without_rendering()
    except AttributeError:
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    trans = ax.transData

    # ---- coordinate transforms ------------------------------------------------
    anchors_disp = trans.transform(np.column_stack([x, y]))

    # Always avoid the anchor points themselves.
    scatter_disp = anchors_disp.copy()
    obstacle_boxes = np.empty((0, 4))

    if avoid:
        extra_pts, extra_boxes = _extract_obstacles(avoid, renderer)
        if len(extra_pts) > 0:
            scatter_disp = np.vstack([scatter_disp, extra_pts])
        if len(extra_boxes) > 0:
            obstacle_boxes = extra_boxes

    ax_bbox = ax.get_window_extent(renderer)
    xlim_d = (ax_bbox.x0, ax_bbox.x1)
    ylim_d = (ax_bbox.y0, ax_bbox.y1)

    # Force centre alignment so bbox centres map predictably.
    for key in ("ha", "va", "horizontalalignment", "verticalalignment"):
        text_kwargs.pop(key, None)

    # ---- measure text sizes ---------------------------------------------------
    labels: list[_Label] = []
    for i, txt in enumerate(texts):
        t = ax.text(0, 0, txt, ha="center", va="center", **text_kwargs)
        bb = t.get_window_extent(renderer)
        labels.append(_Label(
            idx=i,
            x_disp=anchors_disp[i, 0],
            y_disp=anchors_disp[i, 1],
            width=bb.width,
            height=bb.height,
            text=txt,
        ))
        t.remove()

    # ---- run placement --------------------------------------------------------
    _place_labels(
        labels, scatter_disp, obstacle_boxes, margin,
        min_distance, max_distance, n_candidates,
        xlim_d, ylim_d, max_backtrack,
    )

    # ---- draw -----------------------------------------------------------------
    inv = trans.inverted()

    default_arrow = dict(arrowstyle="-", color="0.5", lw=0.75, shrinkA=0, shrinkB=0)
    default_arrow.update(arrow_kwargs)

    results = []
    for lb in labels:
        cx_d = (lb.bbox[0] + lb.bbox[2]) / 2
        cy_d = (lb.bbox[1] + lb.bbox[3]) / 2
        cx_data, cy_data = inv.transform_point((cx_d, cy_d))

        ann = ax.annotate(
            lb.text,
            xy=(x[lb.idx], y[lb.idx]),
            xytext=(cx_data, cy_data),
            ha="center",
            va="center",
            arrowprops=default_arrow if draw_arrows else None,
            **text_kwargs,
        )
        results.append(ann)

    return results
