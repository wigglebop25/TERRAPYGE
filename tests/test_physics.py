"""Unit tests for TERRAPYGE physics features.

Verifies the infinite-slope Factor of Safety, critical acceleration,
Arias intensity, Newmark displacement, and label thresholds against
hand-computed values, plus edge-case behavior (clipping, floors).
"""

import sys
sys.path.insert(0, r'D:\TERRAPYGE')

import numpy as np
import pytest

from src.terrapyge.features.physics import (
    derive_geotechnical_params,
    static_fs,
    critical_acceleration_g,
    arias_intensity,
    newmark_displacement_cm,
    physics_labels,
)


def test_derive_geotechnical_params_basic():
    gamma, c, phi = derive_geotechnical_params(30.0, 40.0, 140.0)
    assert gamma == pytest.approx(14.0, abs=1e-6)      # clamped to lower bound
    assert c == pytest.approx(5.0, abs=1e-6)           # 2 + 0.1*30
    assert phi == pytest.approx(32.0, abs=1e-6)        # 28 + 0.1*40


def test_derive_geotechnical_params_clamping():
    # extreme inputs must stay within bounds
    gamma, c, phi = derive_geotechnical_params(100.0, 100.0, 300.0)
    assert 14.0 <= gamma <= 20.0
    assert 2.0 <= c <= 10.0
    assert 25.0 <= phi <= 38.0


def test_static_fs_hand_computed():
    # slope=30deg, c'=5 kPa, phi=30deg, gamma=17, z=2, ru=0.15
    fs = static_fs(30.0, 5.0, 30.0, 17.0, 2.0, ru=0.15)
    assert fs == pytest.approx(1.1396, abs=1e-3)


def test_static_fs_clipped_lower_bound():
    # a statically unstable configuration is clipped to 1.0
    fs = static_fs(60.0, 0.1, 20.0, 17.0, 2.0, ru=0.5)
    assert fs == pytest.approx(1.0, abs=1e-6)


def test_static_fs_gentle_slope_is_high():
    fs_flat = static_fs(0.1, 5.0, 30.0, 17.0, 2.0, ru=0.15)
    fs_steep = static_fs(45.0, 5.0, 30.0, 17.0, 2.0, ru=0.15)
    assert fs_flat > fs_steep  # flatter terrain is more stable


def test_critical_acceleration_hand_computed():
    fs = static_fs(30.0, 5.0, 30.0, 17.0, 2.0, ru=0.15)
    ac = critical_acceleration_g(fs, 30.0)
    expected = (fs - 1.0) * np.sin(np.radians(30.0))
    assert ac == pytest.approx(expected, abs=1e-6)


def test_critical_acceleration_floor():
    # FS = 1 => Ac = 0, floored at 0.01
    ac = critical_acceleration_g(1.0, 30.0)
    assert ac == pytest.approx(0.01, abs=1e-6)


def test_arias_intensity_hand_computed():
    ia = arias_intensity(0.4 * 9.81, 10.0)
    assert ia == pytest.approx(8.218, abs=1e-2)


def test_newmark_displacement_hand_computed():
    dn = newmark_displacement_cm(8.218, 0.0698)
    assert dn == pytest.approx(18.0, abs=0.5)


def test_newmark_zero_when_no_yield():
    # Ac <= 0 => no displacement
    dn = newmark_displacement_cm(8.218, 0.0)
    assert dn == pytest.approx(0.0, abs=1e-9)


def test_newmark_scalar_and_array():
    scalar = newmark_displacement_cm(8.218, 0.2)
    arr = newmark_displacement_cm(8.218, np.array([0.2, 0.3, 0.5]))
    assert isinstance(arr, np.ndarray) and arr.shape == (3,)
    assert arr[0] == pytest.approx(scalar, abs=1e-9)
    assert arr[0] > arr[1] > arr[2]  # higher yield acc => less displacement


def test_physics_labels_boundaries():
    dn = np.array([1.0, 2.0, 4.9, 5.0, 14.9, 15.0, 30.0])
    labels = physics_labels(dn)
    assert list(labels) == [0, 1, 1, 2, 2, 3, 3]


def test_physics_labels_custom_thresholds():
    # thresholds (1, 2, 5): <1 -> 0, [1,2) -> 1, [2,5) -> 2, >=5 -> 3
    dn = np.array([0.5, 1.5, 3.0, 10.0])
    labels = physics_labels(dn, thresholds=(1.0, 2.0, 5.0))
    assert list(labels) == [0, 1, 2, 3]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
