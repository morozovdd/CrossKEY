"""Tests for the visualization module."""

import numpy as np
import plotly.graph_objects as go
import pytest

from visualization import (
    build_matching_figure,
    create_isosurface_trace,
    create_keypoint_trace,
    create_match_lines,
    downsample_volume,
    scale_points,
)


def test_downsample_volume():
    vol = np.random.rand(128, 128, 128).astype(np.float32)
    small = downsample_volume(vol, target_size=32)
    assert small.shape == (32, 32, 32)


def test_downsample_non_cubic():
    vol = np.random.rand(100, 200, 150).astype(np.float32)
    small = downsample_volume(vol, target_size=64)
    assert small.shape == (64, 64, 64)


def test_scale_points():
    points = np.array([[100.0, 200.0, 150.0], [50.0, 100.0, 75.0]])
    padded_shape = (200, 400, 300)
    volume_shape = (64, 64, 64)
    scaled = scale_points(points, padded_shape, volume_shape)
    expected = points * np.array([64 / 200, 64 / 400, 64 / 300])
    np.testing.assert_allclose(scaled, expected)


def test_create_isosurface_trace():
    coords = np.mgrid[:32, :32, :32].astype(float)
    center = np.array([16, 16, 16])[:, None, None, None]
    vol = np.exp(-np.sum((coords - center) ** 2, axis=0) / 50).astype(np.float32)
    trace = create_isosurface_trace(vol, level=0.3, colorscale="Gray", name="test")
    assert isinstance(trace, go.Mesh3d)
    assert len(trace.x) > 0


def test_create_keypoint_trace():
    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    trace = create_keypoint_trace(pts, color="red", name="kp")
    assert isinstance(trace, go.Scatter3d)
    assert len(trace.x) == 2


def test_create_keypoint_trace_with_offset():
    pts = np.array([[1, 2, 3]], dtype=float)
    trace = create_keypoint_trace(pts, color="red", name="kp", offset_x=100)
    # Axis remap: data[1] -> Plotly x, so x = pts[1] + offset = 2 + 100 = 102
    assert trace.x[0] == 102.0


def test_create_match_lines():
    src = np.array([[0, 0, 0], [10, 10, 10]], dtype=float)
    tgt = np.array([[1, 1, 1], [11, 11, 11]], dtype=float)
    trace = create_match_lines(src, tgt, color="green", name="matches", offset_x=50)
    assert isinstance(trace, go.Scatter3d)
    assert len(trace.x) == 6
    # Axis remap: data[1] -> Plotly x. src[0]=[0,0,0] -> x=src[1]=0; tgt[0]=[1,1,1] -> x=tgt[1]+50=51
    assert trace.x[0] == 0.0
    assert trace.x[1] == 51.0


def test_build_matching_figure():
    vol_mr = np.random.rand(32, 32, 32).astype(np.float32) * 0.5
    vol_us = np.random.rand(32, 32, 32).astype(np.float32) * 0.5
    coords = np.mgrid[:32, :32, :32].astype(float)
    sphere = np.exp(-np.sum((coords - 16) ** 2, axis=0) / 30)
    vol_mr += sphere.astype(np.float32)
    vol_us += sphere.astype(np.float32)
    pts_mr = np.array([[16, 16, 16], [10, 10, 10], [20, 20, 20]], dtype=float)
    pts_us = np.array([[16, 16, 16], [12, 12, 12], [25, 25, 25]], dtype=float)
    match_pairs = [(0, 0, 0.1), (1, 1, 0.5), (2, 2, 0.9)]
    metrics = {"num_matches": 3, "num_correct": 2, "precision": 66.7, "matching_score": 0.5}
    fig = build_matching_figure(
        volume_mr=vol_mr, volume_us=vol_us,
        points_mr=pts_mr, points_us=pts_us,
        padded_shape_mr=(32, 32, 32), padded_shape_us=(32, 32, 32),
        match_pairs=match_pairs, metrics=metrics, evaluation_threshold=5.0,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 4


def test_build_matching_figure_no_matches():
    vol = np.random.rand(32, 32, 32).astype(np.float32)
    coords = np.mgrid[:32, :32, :32].astype(float)
    vol += np.exp(-np.sum((coords - 16) ** 2, axis=0) / 30).astype(np.float32)
    pts_mr = np.array([[16, 16, 16]], dtype=float)
    pts_us = np.array([[16, 16, 16]], dtype=float)
    fig = build_matching_figure(
        volume_mr=vol, volume_us=vol,
        points_mr=pts_mr, points_us=pts_us,
        padded_shape_mr=(32, 32, 32), padded_shape_us=(32, 32, 32),
        match_pairs=[], metrics={"num_matches": 0, "num_correct": 0, "precision": 0, "matching_score": 0},
        evaluation_threshold=5.0,
    )
    assert isinstance(fig, go.Figure)
