
# tests_physics.py — sanity checks for Torque & Drag physics
import numpy as np
from physics_core import mincurv, soft_string_step, neutral_point

def simple_string(total_md=5000.0):
    md = np.arange(0.0, total_md+1.0, 1.0)
    # build to 30°, hold
    inc = np.minimum(30.0, np.maximum(0.0, md-1000.0)*(3.0/100.0))
    az  = np.zeros_like(md)
    return md, inc, az

def simple_components(md):
    return ["DP"]*(len(md)-1), {"DP":{"od_in":5.0,"id_in":4.0,"w_air_lbft":19.5}}

def test_min_curv_monotonic():
    md, inc, az = simple_string()
    N,E,TVD,DLS,k = mincurv(md, inc, az)
    assert (TVD[1:] >= TVD[:-1]).all()  # TVD non-decreasing
    assert DLS.max() > 0

def test_buoyancy_effect():
    md, inc, az = simple_string()
    N,E,TVD,DLS,k = mincurv(md, inc, az)
    cased = (md[:-1] <= 2000.0)
    comp_along, props = simple_components(md)
    # lower MW -> higher BF -> lower HL
    _, T_low, _ = soft_string_step(md, inc, k, cased, comp_along, props, 0.15,0.30,0.06,0.12, mw_ppg=12.0, scenario="pickup")
    _, T_high,_ = soft_string_step(md, inc, k, cased, comp_along, props, 0.15,0.30,0.06,0.12, mw_ppg=9.0,  scenario="pickup")
    assert -T_low[-1] > -T_high[-1]  # surface HL higher at lower BF (heavier pipe)

def test_mu_increase_increases_HL_Torque():
    md, inc, az = simple_string()
    N,E,TVD,DLS,k = mincurv(md, inc, az)
    cased = (md[:-1] <= 2000.0)
    comp_along, props = simple_components(md)
    _, T1, M1 = soft_string_step(md, inc, k, cased, comp_along, props, 0.10,0.20,0.05,0.08, 10.0, "pickup")
    _, T2, M2 = soft_string_step(md, inc, k, cased, comp_along, props, 0.20,0.40,0.10,0.16, 10.0, "pickup")
    assert -T2[-1] > -T1[-1]
    assert abs(M2[-1]) > abs(M1[-1])

def test_neutral_point_in_slackoff():
    md, inc, az = simple_string()
    N,E,TVD,DLS,k = mincurv(md, inc, az)
    cased = (md[:-1] <= 2000.0)
    comp_along, props = simple_components(md)
    _, Tso, _ = soft_string_step(md, inc, k, cased, comp_along, props, 0.15,0.30,0.06,0.12, 10.0, "slackoff")
    np_md = neutral_point(md, Tso)
    assert np.isfinite(np_md)

if __name__ == "__main__":
    # Run all tests
    test_min_curv_monotonic()
    test_buoyancy_effect()
    test_mu_increase_increases_HL_Torque()
    test_neutral_point_in_slackoff()
    print("All physics sanity tests passed.")
