# streamlit_app.py — deep diagnostic version
from __future__ import annotations
import math, io, itertools, time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEG2RAD = math.pi/180.0
IN2FT   = 1.0/12.0

st.set_page_config(page_title="Torque & Drag — Deep Diagnostics", layout="wide")

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
        cos_dl = np.clip(np.cos(I1)*np.cos(I2) + np.sin(I1)*np.sin(I2)*np.cos(A2 - A1), -1.0, 1.0)
        dpsi = np.arccos(cos_dl)
        RF = np.where(dpsi < 1e-12, 1.0, (2.0/dpsi)*np.tan(dpsi/2.0))
        dN = 0.5*dmd*(np.sin(I1)*np.cos(A1)+np.sin(I2)*np.cos(A2))*RF
        dE = 0.5*dmd*(np.sin(I1)*np.sin(A1)+np.sin(I2)*np.sin(A2))*RF
        dZ = 0.5*dmd*(np.cos(I1)+np.cos(I2))*RF
        N[i] = N[i-1]+dN; E[i] = E[i-1]+dE; TVD[i] = TVD[i-1]+dZ
        DLS[i] = (dpsi/DEG2RAD)/dmd*100.0 if dmd>0 else 0.0
    kappa = DLS/100.0*DEG2RAD   # 1/ft
    return N, E, TVD, DLS, kappa

# ───────────────────── Johancsik soft-string stepper ─────────────────────
def pick_section(depth, segments):
    for seg in segments:
        if seg["top"] <= depth <= seg["bottom"]:
            return seg
    return segments[-1]

def johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case, mu_oh, shoe_md, scenario="pickup"):
    ds = 1.0
    n = len(md)
    inc = np.deg2rad(inc_deg)
    BF = (65.5 - mw_ppg)/65.5

    # Per-depth properties
    od = np.zeros(n); w_b = np.zeros(n); mu_s = np.zeros(n); mu_r = np.zeros(n); N_side = np.zeros(n)
    for i in range(n):
        depth = md[i]
        seg = pick_section(depth, segments)
        od[i] = seg["od_in"]
        w_b[i] = seg["weight_lbft"]*BF
        if depth <= shoe_md:
            mu_s[i] = mu_case["slide"]; mu_r[i] = mu_case["rot"]
        else:
            mu_s[i] = mu_oh["slide"];   mu_r[i] = mu_oh["rot"]

    # Integrate from TD up to surface
    T = np.zeros(n); M = np.zeros(n)
    sgn_ax = +1.0 if scenario=="pickup" else -1.0
    for i in range(n-2, -1, -1):
        th = inc[i+1]
        N_side[i+1] = w_b[i+1]*np.sin(th) + np.abs(T[i+1])*kappa[i+1]
        dT = (sgn_ax*w_b[i+1]*np.cos(th) + mu_s[i+1]*N_side[i+1]) * ds
        T[i] = T[i+1] + dT
        r_eff = 0.5*od[i+1]*IN2FT
        M[i] = M[i+1] + (mu_r[i+1]*N_side[i+1]*r_eff)*ds
    neutral_idx = int(np.argmin(np.abs(T)))
    return T, M, neutral_idx, od, w_b, mu_s, mu_r, N_side

# ─────────────────────────────── UI ─────────────────────────────────
st.title("Torque & Drag — Deep Diagnostics")
st.caption("Johancsik soft-string with casing/OH masks, multi-string, tortuosity, multi-μ comparisons, and trip torque. Δs=1 ft.")

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

    # Casing mask
    st.header("Friction Masks")
    shoe_md = st.number_input("Casing shoe MD (ft)", 0.0, float(md[-1]), min(float(md[-1])*0.5, 8000.0), 50.0)
    st.write("Scenario 1 μ (used for base plots below):")
    mu_case_slide_1 = st.number_input("μ casing slide (S1)", 0.02, 0.60, 0.15, 0.01)
    mu_case_rot_1   = st.number_input("μ casing rot (S1)",   0.01, 0.60, 0.06, 0.01)
    mu_oh_slide_1   = st.number_input("μ open-hole slide (S1)", 0.05, 0.90, 0.30, 0.01)
    mu_oh_rot_1     = st.number_input("μ open-hole rot (S1)",   0.02, 0.90, 0.12, 0.01)

    st.write("Optional μ Scenarios 2 & 3 for comparison:")
    enable_s2 = st.checkbox("Enable Scenario 2")
    if enable_s2:
        mu_case_slide_2 = st.number_input("μ casing slide (S2)", 0.02, 0.60, 0.20, 0.01)
        mu_case_rot_2   = st.number_input("μ casing rot (S2)",   0.01, 0.60, 0.08, 0.01)
        mu_oh_slide_2   = st.number_input("μ open-hole slide (S2)", 0.05, 0.90, 0.35, 0.01)
        mu_oh_rot_2     = st.number_input("μ open-hole rot (S2)",   0.02, 0.90, 0.15, 0.01)
    enable_s3 = st.checkbox("Enable Scenario 3")
    if enable_s3:
        mu_case_slide_3 = st.number_input("μ casing slide (S3)", 0.02, 0.60, 0.12, 0.01)
        mu_case_rot_3   = st.number_input("μ casing rot (S3)",   0.01, 0.60, 0.05, 0.01)
        mu_oh_slide_3   = st.number_input("μ open-hole slide (S3)", 0.05, 0.90, 0.25, 0.01)
        mu_oh_rot_3     = st.number_input("μ open-hole rot (S3)",   0.02, 0.90, 0.10, 0.01)

    st.header("Mud & Multi‑String")
    mw_ppg = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)

    st.caption("Define Top/Bottom MD and properties for DC, HWDP, DP.")
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
N, E, TVD, DLS, kappa = mincurv_positions(md, inc_deg, az)

# Base scenario (S1)
mu_case_1 = {"slide": mu_case_slide_1, "rot": mu_case_rot_1}
mu_oh_1   = {"slide": mu_oh_slide_1,   "rot": mu_oh_rot_1}
T_pu_1, M_ro_1, neutral_pu_1, od_1, w_b_1, mu_s_1, mu_r_1, N_side_1 = johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case_1, mu_oh_1, shoe_md, "pickup")
T_so_1, _,          neutral_so_1, _,    _,      _,      _,      _    = johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case_1, mu_oh_1, shoe_md, "slackoff")

# Optional scenarios
scenarios = [("S1", T_pu_1, T_so_1, M_ro_1, mu_case_1, mu_oh_1)]
if 'enable_s2' in globals() and enable_s2:
    mu_case_2 = {"slide": mu_case_slide_2, "rot": mu_case_rot_2}
    mu_oh_2   = {"slide": mu_oh_slide_2,   "rot": mu_oh_rot_2}
    T_pu_2, M_ro_2, _, _, _, _, _, _ = johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case_2, mu_oh_2, shoe_md, "pickup")
    T_so_2, _,   _, _, _, _, _, _     = johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case_2, mu_oh_2, shoe_md, "slackoff")
    scenarios.append(("S2", T_pu_2, T_so_2, M_ro_2, mu_case_2, mu_oh_2))
if 'enable_s3' in globals() and enable_s3:
    mu_case_3 = {"slide": mu_case_slide_3, "rot": mu_case_rot_3}
    mu_oh_3   = {"slide": mu_oh_slide_3,   "rot": mu_oh_rot_3}
    T_pu_3, M_ro_3, _, _, _, _, _, _ = johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case_3, mu_oh_3, shoe_md, "pickup")
    T_so_3, _,   _, _, _, _, _, _     = johancsik_step(md, inc_deg, kappa, mw_ppg, segments, mu_case_3, mu_oh_3, shoe_md, "slackoff")
    scenarios.append(("S3", T_pu_3, T_so_3, M_ro_3, mu_case_3, mu_oh_3))

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
    st.subheader("2D — TVD vs VS (split)")
    VS = N*np.cos(az[0]*DEG2RAD) + E*np.sin(az[0]*DEG2RAD)
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=VS[:idx_shoe+1], y=TVD[:idx_shoe+1], mode="lines", name="Cased"))
    fig2d.add_trace(go.Scatter(x=VS[idx_shoe:], y=TVD[idx_shoe:], mode="lines", name="Open hole"))
    fig2d.update_layout(height=460, xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2d, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Pickup / Slack-off Hookload")
    figHL = go.Figure()
    for name, T_pu, T_so, M_ro, mu_case, mu_oh in scenarios:
        figHL.add_trace(go.Scatter(x=md, y=-T_so, name=f"Slack-off {name}"))
        figHL.add_trace(go.Scatter(x=md, y=-T_pu, name=f"Pickup {name}", line=dict(dash="dash")))
    figHL.update_layout(xaxis_title="MD (ft)", yaxis_title="Hookload (lbf)")
    st.plotly_chart(figHL, use_container_width=True)
with c4:
    st.subheader("Rotating Torque vs MD")
    figT = go.Figure()
    for name, T_pu, T_so, M_ro, mu_case, mu_oh in scenarios:
        figT.add_trace(go.Scatter(x=md, y=np.abs(M_ro), name=f"Torque {name}"))
    figT.update_layout(xaxis_title="MD (ft)", yaxis_title="Torque (ft-lbf, proxy)")
    st.plotly_chart(figT, use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    st.subheader("Tension vs Inclination (diagnostic) — S1")
    figTI = go.Figure()
    figTI.add_trace(go.Scatter(x=inc_deg, y=T_pu_1, name="Pickup T"))
    figTI.add_trace(go.Scatter(x=inc_deg, y=T_so_1, name="Slack-off T", line=dict(dash="dash")))
    figTI.update_layout(xaxis_title="Inclination (deg)", yaxis_title="Axial Tension (lbf)")
    st.plotly_chart(figTI, use_container_width=True)
with c6:
    st.subheader("Tortuosity: DLS and Curvature κ")
    figD = go.Figure()
    figD.add_trace(go.Scatter(x=md, y=DLS, name="DLS (deg/100 ft)"))
    figD.add_trace(go.Scatter(x=md, y=kappa, name="κ (1/ft)", yaxis="y2"))
    figD.update_layout(
        xaxis=dict(title="MD (ft)"),
        yaxis=dict(title="DLS (deg/100 ft)"),
        yaxis2=dict(title="κ (1/ft)", overlaying="y", side="right"))
    st.plotly_chart(figD, use_container_width=True)

st.subheader("Trip‑log Torque (simple simulator) — S1")
speed = st.slider("Trip speed (ft/min)", 5, 200, 60, 5)
dt = 1.0  # minute
t = np.arange(0, max(1, math.ceil(md[-1]/speed))+dt, dt)
bit_md = np.clip(t*speed, 0, md[-1])
idx = np.searchsorted(md, bit_md)
trip_torque = np.abs(M_ro_1[idx])
figTrip = go.Figure()
figTrip.add_trace(go.Scatter(x=t, y=trip_torque, name="Torque during trip"))
figTrip.update_layout(xaxis_title="Time (min)", yaxis_title="Torque (ft-lbf, proxy)")
st.plotly_chart(figTrip, use_container_width=True)

# Table & CSV
df = pd.DataFrame({
    "MD_ft": md, "Inc_deg": inc_deg, "Az_deg": az, "TVD_ft": TVD, "North_ft": N, "East_ft": E,
    "DLS_deg_per_100ft": DLS, "kappa_1_per_ft": kappa, "OD_in": od_1, "w_b_lbft": w_b_1,
    "mu_slide": mu_s_1, "mu_rot": mu_r_1, "N_side_lbf_per_ft": N_side_1,
    "T_pickup_lbf": T_pu_1, "T_slackoff_lbf": T_so_1, "Torque_rot_ftlbf": M_ro_1
})
st.markdown(f"**Neutral point (pickup)** ≈ {md[int(np.argmin(np.abs(T_pu_1)))]:.0f} ft · **Neutral point (slack-off)** ≈ {md[int(np.argmin(np.abs(T_so_1)))]:.0f} ft")
st.dataframe(df.head(20), use_container_width=True)
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="tdrag_deep_output.csv", mime="text/csv")

st.caption("Physics: Johancsik soft-string with N = w·sinθ + |T|·κ, ΔM = μ_rot·N·r_eff. For production, extend with 3D contact and torsional balance.")
