import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transforms import (translation, rotation_y, scale,
                        transform_point, transform_direction)
from maze import cell_center_3d, old_screen_to_world


def test_transform_example():
    p = np.array([1., 0., 0., 1.])
    M = translation(2, 0, 3) @ rotation_y(np.pi / 2)
    assert np.allclose((M @ p)[:3], [2., 0., 2.])


def test_world_cell_center():
    assert np.allclose(cell_center_3d(2, 1), [2.5, 0., 1.5])


def test_old_screen_to_world():
    assert np.allclose(old_screen_to_world((65, 470)), [0.5, 0.22, 7.5])


def test_direction_ignores_translation():
    d = transform_direction(translation(5, 3, 2), [1., 0., 0.])
    assert np.allclose(d, [1., 0., 0.])


def test_transform_order_matters():
    p = np.array([1., 1., 1.])
    a = transform_point(translation(2, 0, 0) @ scale(2, 2, 2), p)
    b = transform_point(scale(2, 2, 2) @ translation(2, 0, 0), p)
    assert np.allclose(a, [4., 2., 2.])
    assert np.allclose(b, [6., 2., 2.])
    assert not np.allclose(a, b)


if __name__ == "__main__":
    test_transform_example()
    test_world_cell_center()
    test_old_screen_to_world()
    test_direction_ignores_translation()
    test_transform_order_matters()
    print("all tests passed")
