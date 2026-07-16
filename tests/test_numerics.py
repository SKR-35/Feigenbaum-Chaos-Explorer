import numpy as np

from feigenbaum_explorer.maps import MAPS
from feigenbaum_explorer.numerics import iterate_orbit, scaling_estimates


def test_logistic_orbit_stays_bounded_at_r_3_2():
    spec = MAPS["Logistic map"]
    orbit = iterate_orbit(spec, 3.2, 0.2, 100)
    assert np.all((orbit >= 0) & (orbit <= 1))


def test_logistic_delta_converges_toward_feigenbaum_constant():
    rows = scaling_estimates(MAPS["Logistic map"], levels=6)
    estimates = [row.delta for row in rows if row.delta is not None]
    assert estimates
    assert abs(estimates[-1] - 4.6692016091) < 0.08


def test_alpha_has_expected_sign_and_scale():
    rows = scaling_estimates(MAPS["Logistic map"], levels=6)
    estimates = [row.alpha for row in rows if row.alpha is not None]
    assert estimates[-1] < 0
    assert 2.3 < abs(estimates[-1]) < 2.7
