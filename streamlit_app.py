# streamlit_app_advanced.py — Advanced torque & drag app
from __future__ import annotations
import math, numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DEG2RAD = math.pi/180.0
IN2FT   = 1.0/12.0

st.set_page_config(page_title="Torque & Drag — Advanced", layout="wide")

def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5

# Minimal tool-joint DB (example ratings)
TOOL_JOINT_DB = {
    "NC38": {"od": 4.75, "id": 2.25, "T_makeup_ftlbf": 12000, "F_tensile_lbf": 350000, "T_yield_ftlbf": 20000},
    "NC40": {"od": 5.00, "id": 2.25, "T_makeup_ftlbf": 16000, "F_tensile_lbf": 420000, "T_yield_ftlbf": 25000},
    "NC50": {"od": 6.63, "id": 3.00, "T_makeup_ftlbf": 30000, "F_tensile_lbf": 650000, "T_yield_ftlbf": 45000},
}

# ───────── Synthetic surveys (same API as before) ─────────
def synth_build_hold(kop_md, build_rate, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * (build_rate / 100.0))
    az = np.full_like(md, az_deg, dtype=float); return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate/100.0; dr = drop_rate/100.0
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * br)
    reach = np.argmax(theta >= theta_hold_deg-1e-6)
    reach_md = md[reach] if theta[reach] >= theta_hold_deg-1e-6 else 0.75*target_md
    theta = np.where(md>reach_md, np.maximum(0.0, theta - (md-reach_md)*dr), theta)
    az = np.full_like(md, az_deg, dtype=float); return md, theta, az

def synth_horizontal(kop_md, build_rate, lateral_length, target_md, az_deg, theta_max=90.0):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate / 100.0
    theta = np.minimum(theta_max, np.maximum(0.0, md - kop_md) * br)
    idx = np.where(theta >= theta_max - 1e-6)[0]
    if len(idx):
        m_h = md[idx[0]]
        md_end = max(target_md, m_h + lateral_length)
        md = np.arange(0.0, md_end + ds, ds)
        theta = np.where(md <= m_h, np.minimum(theta_max, np.maximum(0.0, md - kop_md) * br), theta_max)
    az = np.full_like(md, az_deg, dtype=float); return md, theta, az

# Minimum Curvature geometry
def mincurv_positions(md, inc_deg, az_deg):
    ds = np.diff(md)
    n = len(md)
    N = np.zeros(n); E = np.zeros(n); TVD = np.zeros(n); DLS = np.zeros(n)
    for i in range(1, n):
        I1 = inc_deg[i-1]*DEG2RAD; A1 = az_deg[i-1]*DEG2RAD
        I2 = inc_deg[i]*DEG2RAD;   A2 = az_deg[i]*DEG2RAD
        dmd = ds[i-1]
        cos_dl = clamp(math.cos(I1)*math.cos(I2) + math.sin(I1)*math.sin(I2)*math.cos(A2 - A1), -1.0, 1.0)
        dpsi = math.acos(cos_dl)
        RF = 1.0 if dpsi < 1e-12 else (2.0/dpsi)*math.tan(dpsi/2.0)
        dN = 0.5*dmd*(math.sin(I1)*math.cos(A1)+math.sin(I2)*math.cos(A2))*RF
        dE = 0.5*dmd*(math.sin(I1)*math.sin(A1)+math.sin(I2)*math.sin(A2))*RF
        dZ = 0.5*dmd*(math.cos(I1)+math.cos(I2))*RF
        N[i] = N[i-1]+dN; E[i] = E[i-1]+dE; TVD[i] = TVD[i-1]+dZ
        DLS[i] = (dpsi/DEG2RAD)/dmd*100.0 if dmd>0 else 0.0
    kappa = DLS/100.0*DEG2RAD   # 1/ft
    return N, E, TVD, DLS, kappa

# Section-wise μ scheduler
def mu_at_depth(depth, shoe_md, mu_sections, kind):
    """kind in {'slide','rot'}; mu_sections is a list of dicts with top, bottom, mu_casing_kind, mu_oh_kind"""
    for sec in mu_sections:
        if sec["top"] <= depth <= sec["bottom"]:
            if depth <= shoe_md:
                return sec[f"mu_casing_{kind}"]
            else:
                return sec[f"mu_oh_{kind}"]
    # fallback to last
    sec = mu_sections[-1]
    return sec[f"mu_casing_{kind}"] if depth <= shoe_md else sec[f"mu_oh_{kind}"]

# Advanced stepper with section-wise μ and rotate-off factor
def johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_sections, shoe_md, scenario="pickup", rot_fc=0.6, WOB_lbf=0.0, Mbit_ftlbf=0.0):
    ds = 1.0
    inc = np.deg2rad(inc_deg)
    n = len(md)
    BF = bf_from_mw(mw_ppg)

    # per-depth props
    od = np.zeros(n); w_b = np.zeros(n); mu_s = np.zeros(n); mu_r = np.zeros(n); N_side = np.zeros(n)
    for i in range(n):
        depth = md[i]
        # string section
        seg = None
        for s in segments:
            if s["top"] <= depth <= s["bottom"]:
                seg = s; break
        if seg is None: seg = segments[-1]
        od[i] = float(seg["od_in"])
        w_b[i] = float(seg["weight_lbft"]) * BF
        mu_s[i] = mu_at_depth(depth, shoe_md, mu_sections, "slide")
        mu_r[i] = mu_at_depth(depth, shoe_md, mu_sections, "rot")

    if scenario == "rotate_off":
        mu_s = mu_s * rot_fc  # reduce sliding friction when rotating off-bottom
    # integrate TD -> surface
    T = np.zeros(n); M = np.zeros(n)
    if scenario == "onbottom":
        T[-1] = -float(WOB_lbf)   # compressive at bit
        M[-1] = float(Mbit_ftlbf) # motor torque at bit

    sgn_ax = {"pickup": +1.0, "slackoff": -1.0, "rotate_off": +1.0, "onbottom": +1.0}.get(scenario, +1.0)

    for i in range(n-2, -1, -1):
        th = inc[i+1]; kap = kappa[i+1]
        N_raw = w_b[i+1]*math.sin(th) + abs(T[i+1])*kap
        N_side[i+1] = max(0.0, N_raw)
        dT = (sgn_ax*w_b[i+1]*math.cos(th) + mu_s[i+1]*N_side[i+1]) * ds
        r_eff = 0.5*od[i+1]*IN2FT
        dM = (mu_r[i+1]*N_side[i+1]*r_eff)*ds
        T[i] = T[i+1] + dT
        M[i] = M[i+1] + dM

    neutral_idx = int(np.argmin(np.abs(T)))
    return T, M, neutral_idx, od, w_b, mu_s, mu_r, N_side

# API-style combined limit utilization (simple form): U = max(T/T_lim, M/M_lim)
def combined_limit_util(T_surface, M_surface, tj_spec):
    T_lim = tj_spec["F_tensile_lbf"]
    M_lim = tj_spec["T_yield_ftlbf"]
    return max( abs(T_surface)/T_lim if T_lim>0 else 0.0,
                abs(M_surface)/M_lim if M_lim>0 else 0.0 )

# ───────────────────────────── UI ─────────────────────────────────
st.title("Torque & Drag — Advanced (Calibration, Section-wise μ, Rig Limits)")
st.caption("Johancsik soft-string (Δs=1 ft) with section-wise μ, μ calibration, rotate-off factor, motor torque, and rig-limit checks.")

# Trajectory
c1, c2, c3, c4 = st.columns(4)
profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
build   = c3.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.0, 0.1)
az_deg  = c4.number_input("Azimuth (deg)", 0.0, 360.0, 0.0, 1.0)
if profile == "Build & Hold":
    theta_hold = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 0.5)
    target_md  = st.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
    md, inc_deg, az = synth_build_hold(kop_md, build, theta_hold, target_md, az_deg)
elif profile == "Build–Hold–Drop":
    theta_hold = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 0.5)
    drop_rate  = st.number_input("Drop rate (deg/100 ft)", 0.0, 30.0, 2.0, 0.1)
    target_md  = st.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
    md, inc_deg, az = synth_build_hold_drop(kop_md, build, theta_hold, drop_rate, target_md, az_deg)
else:
    lateral    = st.number_input("Lateral length (ft)", 0.0, 30000.0, 2000.0, 100.0)
    target_md  = st.number_input("Target MD (ft)", 100.0, 100000.0, 10000.0, 100.0)
    md, inc_deg, az = synth_horizontal(kop_md, build, lateral, target_md, az_deg)

N, E, TVD, DLS, kappa = mincurv_positions(md, inc_deg, az)
VS = N*np.cos(az[0]*DEG2RAD) + E*np.sin(az[0]*DEG2RAD)
st.plotly_chart(go.Figure(data=[go.Scatter3d(x=E, y=N, z=TVD, mode="lines")]).update_layout(
    scene=dict(zaxis=dict(autorange="reversed"), xaxis_title="East", yaxis_title="North", zaxis_title="TVD"),
    height=360
), use_container_width=True)

# String
st.subheader("Drillstring sections (Top/Bottom MD, OD, weight/ft)")
n_default = max(1, int(len(md)/4000))
df_string = pd.DataFrame([
    {"top": 0.0, "bottom": float(md[-1]), "od_in": 5.0, "weight_lbft": 19.5},
])
df_string = st.data_editor(df_string, num_rows="dynamic")

# Friction masks — section-wise μ
st.subheader("Section-wise μ (casing vs open-hole)")
shoe_md = st.slider("Casing shoe MD (ft)", 0.0, float(md[-1]), min(8000.0, float(md[-1]*0.6)), 50.0)
df_mu = pd.DataFrame([
    {"top":0.0, "bottom": float(md[-1]), "mu_casing_slide":0.20, "mu_casing_rot":0.08, "mu_oh_slide":0.35, "mu_oh_rot":0.15}
])
df_mu = st.data_editor(df_mu, num_rows="dynamic")

st.subheader("Mud & rotation / motor")
mw_ppg   = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)
rot_fc   = st.slider("Rotate-off friction reduction factor", 0.3, 1.0, 0.6, 0.05)
motor_on = st.checkbox("Motor on-bottom (bit torque = K*ΔP)", value=False)
K_tbit   = st.number_input("K (ft-lbf/psi)", 0.0, 10000.0, 2.5, 0.1) if motor_on else 0.0
deltaP   = st.number_input("ΔP (psi)", 0.0, 5000.0, 0.0, 1.0) if motor_on else 0.0
WOB      = st.number_input("WOB (lbf)", 0.0, 150000.0, 6000.0, 100.0)
Mbit     = K_tbit * deltaP if motor_on else 0.0
scenario = st.selectbox("Scenario", ["Slack-off", "Pickup", "Rotate off-bottom", "Rotate on-bottom"])
scenario_key = {"Slack-off":"slackoff","Pickup":"pickup","Rotate off-bottom":"rotate_off","Rotate on-bottom":"onbottom"}[scenario]

# Rig / tool-joint limits
st.subheader("Rig & Tool-joint limits")
tj_name   = st.selectbox("Tool-joint", list(TOOL_JOINT_DB.keys()), index=1)
TJ = TOOL_JOINT_DB[tj_name]
rig_torque = st.number_input("Rig torque limit (ft-lbf)", 1000.0, 100000.0, 30000.0, 100.0)
rig_hl     = st.number_input("Hookload limit (lbf)",     10000.0, 1000000.0, 400000.0, 1000.0)

# Run the advanced stepper
segments = df_string.to_dict(orient="records")
mu_sections = df_mu.to_dict(orient="records")
T, M, neutral_idx, od, w_b, mu_s, mu_r, N_side = johancsik_step(
    md, inc_deg, kappa, mw_ppg, segments, mu_sections, shoe_md,
    scenario=scenario_key, rot_fc=rot_fc, WOB_lbf=WOB, Mbit_ftlbf=Mbit
)

# Plots
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=md, y=-T, name="Hookload (−T)"))
    fig.add_hline(y=rig_hl, line_dash="dot", annotation_text="Rig HL limit")
    fig.update_layout(xaxis_title="MD (ft)", yaxis_title="Hookload (lbf)")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=md, y=np.abs(M), name="Torque |M|"))
    fig.add_hline(y=rig_torque, line_dash="dot", annotation_text="Rig torque limit")
    fig.update_layout(xaxis_title="MD (ft)", yaxis_title="Torque (ft-lbf)")
    st.plotly_chart(fig, use_container_width=True)

# Combined limit utilization at surface
U = combined_limit_util(-T[0], M[0], TJ)  # surface values
st.markdown(f"**Combined limit util. (API-style)** at surface with {tj_name}: **{U:.2f}** (≥1.0 = exceed)")

# Export table
df = pd.DataFrame({
    "MD_ft": md, "Inc_deg": inc_deg, "Az_deg": az,
    "TVD_ft": TVD, "North_ft": N, "East_ft": E, "VS_ft": VS,
    "DLS_deg_per_100ft": DLS, "kappa_1_per_ft": kappa,
    "OD_in": od, "w_b_lbft": w_b, "mu_slide": mu_s, "mu_rot": mu_r,
    "N_side_lbf_per_ft": N_side, "T_lbf": T, "M_ftlbf": M
})
st.dataframe(df.head(20), use_container_width=True)
st.download_button("Download results CSV", df.to_csv(index=False).encode("utf-8"), "tdrag_advanced.csv", "text/csv")

# μ calibration grid (optional)
st.subheader("μ calibration grid (fit to measured)")
with st.expander("Calibrate μ to measured data"):
    depth_for_fit = st.number_input("Depth to fit (ft)", 0.0, float(md[-1]), float(md[-1]), 50.0)
    measured_pickup_hl   = st.number_input("Measured HL (Pickup) at depth (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
    measured_slackoff_hl = st.number_input("Measured HL (Slack-off) at depth (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
    measured_rotate_hl   = st.number_input("Measured HL (Rotate off) at depth (lbf)", 0.0, 1_000_000.0, 0.0, 100.0)
    measured_surface_torque = st.number_input("Measured Surface Torque (ft-lbf)", 0.0, 200_000.0, 0.0, 100.0)

    st.markdown("Define coarse ranges for μ (start, stop, step). Use small grids to keep it fast.")
    c1, c2 = st.columns(2)
    with c1:
        mu_c_s_rng = st.text_input("μ casing slide range (e.g. 0.10,0.30,0.05)", "0.10,0.30,0.05")
        mu_o_s_rng = st.text_input("μ OH slide range (e.g. 0.20,0.45,0.05)", "0.20,0.45,0.05")
    with c2:
        mu_c_r_rng = st.text_input("μ casing rot range (e.g. 0.04,0.15,0.03)", "0.04,0.15,0.03")
        mu_o_r_rng = st.text_input("μ OH rot range (e.g. 0.08,0.25,0.03)", "0.08,0.25,0.03")

    def parse_rng(txt):
        try:
            a,b,s = [float(x.strip()) for x in txt.split(",")]
            return np.arange(a, b+1e-9, s)
        except:
            return np.array([])

    if st.button("Run calibration grid"):
        idx = int(np.searchsorted(md, depth_for_fit, side="right"))
        md_fit = md[:idx+1]; inc_fit = inc_deg[:idx+1]; kappa_fit = kappa[:idx+1]

        mu_c_s_vals = parse_rng(mu_c_s_rng); mu_o_s_vals = parse_rng(mu_o_s_rng)
        mu_c_r_vals = parse_rng(mu_c_r_rng); mu_o_r_vals = parse_rng(mu_o_r_rng)

        best = None; best_err = 1e99
        for mu_c_s in mu_c_s_vals:
            for mu_o_s in mu_o_s_vals:
                for mu_c_r in mu_c_r_vals:
                    for mu_o_r in mu_o_r_vals:
                        mu_sections_try = [{"top":0.0,"bottom":float(md[-1]),
                                            "mu_casing_slide":mu_c_s,"mu_casing_rot":mu_c_r,
                                            "mu_oh_slide":mu_o_s,"mu_oh_rot":mu_o_r}]
                        # slack-off
                        T_sl, M_sl, _, *_ = johancsik_step(md_fit, inc_fit, kappa_fit, mw_ppg, segments, mu_sections_try, shoe_md, "slackoff", rot_fc)
                        HL_sl = max(0.0, -T_sl[0])
                        err2 = 0.0
                        if measured_slackoff_hl>0: err2 += (HL_sl - measured_slackoff_hl)**2
                        # pickup
                        T_pu, M_pu, _, *_ = johancsik_step(md_fit, inc_fit, kappa_fit, mw_ppg, segments, mu_sections_try, shoe_md, "pickup", rot_fc)
                        HL_pu = max(0.0, -T_pu[0])
                        if measured_pickup_hl>0: err2 += (HL_pu - measured_pickup_hl)**2
                        # rotate-off
                        if measured_rotate_hl>0:
                            T_ro, M_ro, _, *_ = johancsik_step(md_fit, inc_fit, kappa_fit, mw_ppg, segments, mu_sections_try, shoe_md, "rotate_off", rot_fc)
                            HL_ro = max(0.0, -T_ro[0]); err2 += (HL_ro - measured_rotate_hl)**2
                        # torque
                        if measured_surface_torque>0:
                            T_ro, M_ro, _, *_ = johancsik_step(md_fit, inc_fit, kappa_fit, mw_ppg, segments, mu_sections_try, shoe_md, "rotate_off", rot_fc)
                            err2 += (abs(M_ro[0]) - measured_surface_torque)**2
                        if err2 < best_err:
                            best_err = err2
                            best = dict(mu_c_s=float(mu_c_s), mu_o_s=float(mu_o_s), mu_c_r=float(mu_c_r), mu_o_r=float(mu_o_r), SSE=float(err2))
        if best:
            st.success(f"Best μ set: {best}")
        else:
            st.warning("No valid grid or targets.")

st.caption("Model: Johancsik soft-string with curvature (κ) and section-wise μ; rotate-off reduces friction; motor torque adds M at bit; combined limits per a simplified API-style check.")
