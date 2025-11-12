# streamlit_app.py — upgraded
from __future__ import annotations
import math, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEG2RAD = math.pi/180.0
IN2FT   = 1.0/12.0

st.set_page_config(page_title="Torque & Drag — Synthetic Survey (Johancsik, casing masks, multi-string)", layout="wide")

def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5

# ───────────────────── Synthetic surveys ─────────────────────
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
    # start drop after reaching theta_hold or toward end
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

# ───────────────────── Minimum curvature & DLS ─────────────────────
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
    return N, E, TVD, DLS

# ───────────────────── Johancsik soft-string stepper ─────────────────────
def johancsik_step(md, inc_deg, dls_deg_per100, mw_ppg, segments, mu_case, mu_oh, shoe_md, scenario="pickup"):
    """
    segments: list of dicts with keys:
        {"top": md_ft, "bottom": md_ft, "od_in": x, "weight_lbft": y}
      (assumes single ID isn't needed in this simplified axial calc)
    mu_case, mu_oh: dicts {"slide": μ_s, "rot": μ_r}
    scenario: "pickup" or "slackoff"
    Returns: tension array T (lb), torque array M (ft-lbf), neutral point index
    """
    ds = 1.0
    n = len(md)
    inc = np.deg2rad(inc_deg)
    kappa = (dls_deg_per100/100.0) * (math.pi/180.0)  # 1/ft
    BF = bf_from_mw(mw_ppg)
    # Precompute per-depth pipe properties
    od = np.zeros(n); w_b = np.zeros(n); mu_s = np.zeros(n); mu_r = np.zeros(n)
    for i in range(n):
        depth = md[i]
        # Pick pipe segment
        od_i, w_air = None, None
        for seg in segments:
            if seg["top"] <= depth <= seg["bottom"]:
                od_i = seg["od_in"]; w_air = seg["weight_lbft"]; break
        if od_i is None:
            # outside defined segments: assume last
            od_i = segments[-1]["od_in"]; w_air = segments[-1]["weight_lbft"]
        od[i] = od_i
        w_b[i] = w_air * BF
        # Casing vs open hole frictions
        if depth <= shoe_md:
            mu_s[i] = mu_case["slide"]; mu_r[i] = mu_case["rot"]
        else:
            mu_s[i] = mu_oh["slide"];   mu_r[i] = mu_oh["rot"]

    # Integrate from bit to surface (ascending index is toward surface if md increasing from 0)
    # We'll treat index 0 as surface and last as TD; integrate toward surface for tension accumulation.
    # For simplicity, assume tool at TD applying small weight; start with 0 at TD and accumulate upward.
    T = np.zeros(n); M = np.zeros(n)
    sgn_ax = +1.0 if scenario=="pickup" else -1.0
    for i in range(n-2, -1, -1):  # from TD-1 up to surface
        # use properties of segment i+1 (below)
        th = inc[i+1]; mu = mu_s[i+1]
        N_side = w_b[i+1]*math.sin(th) + abs(T[i+1])*kappa[i+1]    # Johancsik: side force from weight + curvature tension
        dT = (sgn_ax*w_b[i+1]*math.cos(th) + mu*N_side) * ds
        T[i] = T[i+1] + dT
        # torque estimate (rotating) using μ_rot and pipe radius
        r_eff = 0.5*od[i+1]*IN2FT
        M[i] = M[i+1] + (mu_r[i+1]*N_side*r_eff)*ds

    # neutral point: depth where T crosses zero under slack-off (compression below, tension above)
    neutral_idx = int(np.argmin(np.abs(T)))  # simple proxy
    return T, M, neutral_idx, od, w_b, mu_s, mu_r

# ─────────────────────────────── UI ─────────────────────────────────
st.title("Torque & Drag — Synthetic Survey")
st.caption("Johancsik soft-string with casing/open-hole friction masks and multi-component string. Δs = 1 ft. Educational model.")

with st.sidebar:
    st.header("Trajectory")
    profile = st.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
    kop_md  = st.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
    build   = st.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.0, 0.1)
    az_deg  = st.number_input("Azimuth (deg)", 0.0, 360.0, 0.0, 1.0)
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

    st.header("Friction Masks")
    shoe_md = st.number_input("Casing shoe MD (ft)", 0.0, float(md[-1]), min(float(md[-1])*0.5, 8000.0), 50.0)
    mu_case_slide = st.number_input("μ casing (slide)", 0.02, 0.60, 0.15, 0.01)
    mu_case_rot   = st.number_input("μ casing (rot)",   0.01, 0.60, 0.06, 0.01)
    mu_oh_slide   = st.number_input("μ open-hole (slide)", 0.05, 0.90, 0.30, 0.01)
    mu_oh_rot     = st.number_input("μ open-hole (rot)",   0.02, 0.90, 0.12, 0.01)

    st.header("Mud & String")
    mw_ppg = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)
    # Multi-component string: DC, HWDP, DP (depth intervals)
    st.caption("Define top/bottom MD and properties for each section.")
    def seg_editor(label, default_top, default_bot, od, w):
        c1, c2, c3, c4 = st.columns(4)
        with c1: top = st.number_input(f"{label} Top MD", 0.0, float(md[-1]), default_top, 100.0, key=label+"t")
        with c2: bot = st.number_input(f"{label} Bottom MD", 0.0, float(md[-1]), default_bot, 100.0, key=label+"b")
        with c3: od_in = st.number_input(f"{label} OD (in)", 2.0, 10.0, od, 0.125, key=label+"o")
        with c4: w_air = st.number_input(f"{label} weight (lb/ft)", 5.0, 120.0, w, 0.5, key=label+"w")
        return {"top": min(top, bot), "bottom": max(top, bot), "od_in": od_in, "weight_lbft": w_air}
    seg_dc   = seg_editor("DC",   0.0, min(800.0, md[-1]), 6.5, 50.0)
    seg_hwdp = seg_editor("HWDP", seg_dc["bottom"], min(2000.0, md[-1]), 5.0, 35.0)
    seg_dp   = seg_editor("DP",   seg_hwdp["bottom"], float(md[-1]), 5.0, 19.5)
    segments = [seg_dc, seg_hwdp, seg_dp]

# Geometry & curvature
N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

# Johancsik stepper: pickup/slack-off and rotating torque
mu_case = {"slide": mu_case_slide, "rot": mu_case_rot}
mu_oh   = {"slide": mu_oh_slide,   "rot": mu_oh_rot}

T_pu, M_ro, neutral_pu, od, w_b, mu_s, mu_r = johancsik_step(md, inc_deg, DLS, mw_ppg, segments, mu_case, mu_oh, shoe_md, "pickup")
T_so, _, neutral_so, _, _, _, _             = johancsik_step(md, inc_deg, DLS, mw_ppg, segments, mu_case, mu_oh, shoe_md, "slackoff")

# ────────────────────────── Plots ───────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("3D Path (split at shoe)")
    idx_shoe = int(np.clip(np.searchsorted(md, shoe_md), 0, len(md)-1))
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=E[:idx_shoe+1], y=N[:idx_shoe+1], z=TVD[:idx_shoe+1], mode="lines", name="Cased"))
    fig3d.add_trace(go.Scatter3d(x=E[idx_shoe:], y=N[idx_shoe:], z=TVD[idx_shoe:], mode="lines", name="Open hole"))
    fig3d.update_layout(height=460, scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)", zaxis=dict(autorange="reversed")),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0))
    st.plotly_chart(fig3d, use_container_width=True)
with c2:
    st.subheader("2D — TVD vs VS (split by shoe)")
    VS = N*np.cos(az[0]*DEG2RAD) + E*np.sin(az[0]*DEG2RAD)
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=VS[:idx_shoe+1], y=TVD[:idx_shoe+1], mode="lines", name="Cased"))
    fig2d.add_trace(go.Scatter(x=VS[idx_shoe:], y=TVD[idx_shoe:], mode="lines", name="Open hole"))
    fig2d.update_layout(height=460, xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2d, use_container_width=True)

st.subheader("Hookload — Pickup / Slack-off")
figHL = go.Figure()
figHL.add_trace(go.Scatter(x=md, y=-T_so, name=f"Slack-off (neutral @ {md[neutral_so]:.0f} ft)"))
figHL.add_trace(go.Scatter(x=md, y=-T_pu, name=f"Pickup (neutral @ {md[neutral_pu]::.0f} ft)"))
figHL.update_layout(xaxis_title="MD (ft)", yaxis_title="Hookload proxy (lbf)")
st.plotly_chart(figHL, use_container_width=True)

st.subheader("Surface Torque (rotating proxy)")
figT = go.Figure()
figT.add_trace(go.Scatter(x=md, y=np.abs(M_ro), name="Torque (rotate)"))
figT.update_layout(xaxis_title="MD (ft)", yaxis_title="Torque (ft-lbf, relative)")
st.plotly_chart(figT, use_container_width=True)

# Neutral-point and masks table
df = pd.DataFrame({
    "MD_ft": md, "Inc_deg": inc_deg, "Az_deg": az,
    "TVD_ft": TVD, "North_ft": N, "East_ft": E, "DLS_deg_per_100ft": DLS,
    "T_pickup_lbf": T_pu, "T_slackoff_lbf": T_so, "Torque_rot_ftlbf": M_ro,
    "OD_in": od, "w_b_lbft": w_b, "mu_slide": mu_s, "mu_rot": mu_r
})
st.markdown(f"**Neutral point (pickup)** ≈ {md[neutral_pu]:.0f} ft, **Neutral point (slack-off)** ≈ {md[neutral_so]:.0f} ft")
st.dataframe(df.head(15), use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="tdrag_output.csv", mime="text/csv")

st.caption("Note: Johancsik soft-string here uses N = w·sinθ + |T|·κ with κ from DLS. For higher fidelity, add contact in 3D and detailed tooljoint effects.")
