import pytest

from render.occlusion_cull import (
    build_hiz,
    project_aabb,
    is_occluded,
    camera_changed,
    cull_objects
)


def test_hiz_levels():
    assert len(build_hiz([[1.0, 2.0], [3.0, 4.0]], 2, 2)) == 2

def test_hiz_base():
    assert build_hiz([[1.0, 2.0], [3.0, 4.0]], 2, 2)[0][0][0] == 1.0

def test_hiz_max():
    assert build_hiz([[1.0, 2.0], [3.0, 4.0]], 2, 2)[1][0][0] == 4.0

def test_project_identity():
    assert project_aabb((-1.0, -1.0, -1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == (0.0, 0.0, 100.0, 100.0)

def test_project_quadrant():
    assert project_aabb((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == (50.0, 50.0, 50.0, 50.0)

def test_project_clipped():
    assert project_aabb((-2.0, -2.0, -1.0, 2.0, 2.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == (0.0, 0.0, 100.0, 100.0)

def test_occluded_behind():
    assert is_occluded((0.0, 0.0, 1.0, 1.0, 1.0, 1.0), [[[0.5]]], 1, 1) is True

def test_visible_in_front():
    assert is_occluded((0.0, 0.0, 0.0, 1.0, 1.0, 1.0), [[[2.0]]], 1, 1) is False

def test_conservative_edge():
    assert is_occluded((0.0, 0.0, 0.5, 1.0, 1.0, 1.0), [[[0.6]]], 1, 1, conservative=True) is True

def test_camera_same():
    assert camera_changed((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.1) is False

def test_camera_move():
    assert camera_changed((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), 0.1) is True

def test_cull_remove():
    assert cull_objects([{'id': 'a', 'aabb': (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)}, {'id': 'b', 'aabb': (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)}], [[[0.5]]], (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == ['b']

def test_cull_keep():
    assert cull_objects([{'id': 'a', 'aabb': (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)}], [[[0.5]]], (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == ['a']

def test_cull_empty():
    assert cull_objects([], [[[0.5]]], (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == []

def test_cull_order():
    assert cull_objects([{'id': 'b', 'aabb': (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)}, {'id': 'a', 'aabb': (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)}], [[[2.0]]], (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), (100, 100)) == ['b', 'a']
