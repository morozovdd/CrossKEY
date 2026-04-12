"""Plotly 3D visualization for CrossKEY matching results.

Builds side-by-side volume isosurfaces with keypoints and match lines.
MR volume on the left, US volume on the right, offset along the X axis.
"""

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom
from skimage.measure import marching_cubes


def downsample_volume(volume: np.ndarray, target_size: int = 64) -> np.ndarray:
    """Downsample volume to target_size^3 for browser-friendly rendering."""
    factors = [target_size / s for s in volume.shape]
    return zoom(volume, factors, order=1).astype(np.float32)


def scale_points(
    points: np.ndarray,
    padded_shape: tuple,
    volume_shape: tuple,
) -> np.ndarray:
    """Scale point coordinates from padded volume space to downsampled volume space."""
    scale = np.array(volume_shape, dtype=float) / np.array(padded_shape, dtype=float)
    return points * scale


def create_isosurface_trace(
    volume: np.ndarray,
    level: float,
    color: str,
    opacity: float = 0.15,
    name: str = "",
    offset_x: float = 0.0,
) -> go.Mesh3d:
    """Create a Mesh3d trace from a volume isosurface via marching cubes."""
    verts, faces, _, _ = marching_cubes(volume, level=level)
    return go.Mesh3d(
        x=verts[:, 0] + offset_x,
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        opacity=opacity,
        color=color,
        name=name,
        showlegend=True,
    )


def create_keypoint_trace(
    points: np.ndarray,
    color: str,
    size: float = 3.0,
    opacity: float = 1.0,
    name: str = "",
    offset_x: float = 0.0,
) -> go.Scatter3d:
    """Create Scatter3d markers for keypoints."""
    return go.Scatter3d(
        x=points[:, 0] + offset_x,
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name,
        showlegend=True,
    )


def create_match_lines(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    color: str,
    width: float = 2.0,
    name: str = "",
    offset_x: float = 0.0,
) -> go.Scatter3d:
    """Create lines connecting matched source points to offset target points."""
    lx, ly, lz = [], [], []
    for s, t in zip(src_pts, tgt_pts):
        lx.extend([float(s[0]), float(t[0]) + offset_x, None])
        ly.extend([float(s[1]), float(t[1]), None])
        lz.extend([float(s[2]), float(t[2]), None])
    return go.Scatter3d(
        x=lx, y=ly, z=lz,
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        showlegend=True,
    )


def build_matching_figure(
    volume_mr: np.ndarray,
    volume_us: np.ndarray,
    points_mr: np.ndarray,
    points_us: np.ndarray,
    padded_shape_mr: tuple,
    padded_shape_us: tuple,
    match_pairs: list,
    metrics: dict,
    evaluation_threshold: float = 5.0,
    mr_level: float = 0.3,
    us_level: float = 0.1,
) -> go.Figure:
    """Build the full 3D matching visualization."""
    fig = go.Figure()

    # Scale keypoints to match downsampled volume coordinates
    pts_mr_viz = scale_points(points_mr, padded_shape_mr, volume_mr.shape)
    pts_us_viz = scale_points(points_us, padded_shape_us, volume_us.shape)

    # Side-by-side offset: MR on left, US on right
    gap = volume_mr.shape[0] * 0.3
    offset_x = volume_mr.shape[0] + gap

    # Volume isosurfaces
    try:
        fig.add_trace(create_isosurface_trace(
            volume_mr, level=mr_level, color="royalblue",
            opacity=0.15, name="MR Surface",
        ))
    except ValueError:
        pass

    try:
        fig.add_trace(create_isosurface_trace(
            volume_us, level=us_level, color="crimson",
            opacity=0.15, name="US Surface", offset_x=offset_x,
        ))
    except ValueError:
        pass

    # Process matches
    src_indices = [p[0] for p in match_pairs]
    tgt_indices = [p[1] for p in match_pairs]

    if match_pairs:
        mr_matched = points_mr[src_indices]
        us_matched = points_us[tgt_indices]
        spatial_dist = np.linalg.norm(mr_matched - us_matched, axis=1)
        correct = spatial_dist < evaluation_threshold

        mr_matched_viz = pts_mr_viz[src_indices]
        us_matched_viz = pts_us_viz[tgt_indices]

        if correct.any():
            fig.add_trace(create_match_lines(
                mr_matched_viz[correct], us_matched_viz[correct],
                color="rgba(0,200,0,0.6)", width=2,
                name=f"Correct ({correct.sum()})", offset_x=offset_x,
            ))

        if (~correct).any():
            fig.add_trace(create_match_lines(
                mr_matched_viz[~correct], us_matched_viz[~correct],
                color="rgba(255,0,0,0.3)", width=1,
                name=f"Incorrect ({(~correct).sum()})", offset_x=offset_x,
            ))

        fig.add_trace(create_keypoint_trace(
            mr_matched_viz, color="royalblue", size=4,
            name=f"MR Matched ({len(mr_matched_viz)})",
        ))
        fig.add_trace(create_keypoint_trace(
            us_matched_viz, color="crimson", size=4,
            name=f"US Matched ({len(us_matched_viz)})", offset_x=offset_x,
        ))

    # Unmatched keypoints (faded)
    matched_mr_set = set(src_indices)
    matched_us_set = set(tgt_indices)
    unmatched_mr = np.array([i not in matched_mr_set for i in range(len(pts_mr_viz))])
    unmatched_us = np.array([i not in matched_us_set for i in range(len(pts_us_viz))])

    if unmatched_mr.any():
        fig.add_trace(create_keypoint_trace(
            pts_mr_viz[unmatched_mr], color="royalblue",
            size=1.5, opacity=0.2, name="MR Unmatched",
        ))
    if unmatched_us.any():
        fig.add_trace(create_keypoint_trace(
            pts_us_viz[unmatched_us], color="crimson",
            size=1.5, opacity=0.2, name="US Unmatched", offset_x=offset_x,
        ))

    # Layout
    prec = metrics.get("precision", 0)
    n_matches = metrics.get("num_matches", 0)
    fig.update_layout(
        title=f"CrossKEY Matching -- {n_matches} matches, {prec:.1f}% precision",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        width=900,
        height=650,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top", y=0.99,
            xanchor="left", x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white"),
        ),
    )

    return fig
