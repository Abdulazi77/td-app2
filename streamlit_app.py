# streamlit_app.py
# PEGN517-style Streamlit app with synthetic survey options and simplified soft-string T&D
from __future__ import annotations
import math, io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DEG2RAD = math.pi/180.0
IN2FT   = 1.0/12.0

st.set_page_config(page_title="Torque & Drag (soft-string) — Synthetic Survey", layout="wide")

def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5

# ───────────────────── Synthetic survey builders ─────────────────────
def synth_build_hold(kop_md, build_rate_deg_per_100ft, theta_hold_deg, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * (build_rate_deg_per_100ft / 100.0))
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

def synth_build_hold_drop(kop_md, build_rate, theta_hold_deg, drop_rate, target_md, az_deg):
    ds = 1.0
    md = np.arange(0.0, target_md + ds, ds)
    br = build_rate / 100.0; dr = drop_rate / 100.0
    theta = np.minimum(theta_hold_deg, np.maximum(0.0, md - kop_md) * br)
    start_drop = 0.75 * target_md
    theta = np.maximum(0.0, theta - np.maximum(0.0, md - start_drop) * dr)
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

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
    az = np.full_like(md, az_deg, dtype=float)
    return md, theta, az

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

# ───────────────────── Simplified soft-string ───────────────────────
def soft_string(md, inc_deg, mu_slide, mu_rot, mw_ppg, w_air_lbft, od_in, scenario="pickup"):
    ds = 1.0
    nseg = len(md)-1
    inc = np.deg2rad(inc_deg[:-1])
    BF = bf_from_mw(mw_ppg)
    w_b = w_air_lbft*BF
    r_eff_ft = 0.5*od_in*IN2FT

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)
    sgn_ax = {"pickup": +1.0, "slackoff": -1.0}.get(scenario, 0.0)
    for i in range(nseg):
        N_side = w_b*np.sin(inc[i])  # normal force (simplified)
        T[i+1] = T[i] + (sgn_ax*w_b*np.cos(inc[i]) + mu_slide*N_side)*ds
        M[i+1] = M[i] + (mu_rot*N_side*r_eff_ft)*ds
    return T, M

# ─────────────────────────────── UI ─────────────────────────────────
st.title("Torque & Drag (soft-string) — Synthetic Survey")
st.caption("Three synthetic survey options with PEGN517-like inputs, plus simple T&D. Δs = 1 ft.")

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

    # T&D inputs
    st.header("T&D Inputs")
    mw_ppg = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)
    mu_slide = st.number_input("μ (sliding)", 0.05, 0.80, 0.25, 0.01)
    mu_rot   = st.number_input("μ (rotating)", 0.05, 0.80, 0.25, 0.01)
    od_in    = st.number_input("Pipe OD (in)", 2.0, 8.0, 5.0, 0.01)
    id_in    = st.number_input("Pipe ID (in)", 0.5, 7.5, 4.0, 0.01)
    w_air_lbft = st.number_input("Pipe weight (air, lb/ft)", 1.0, 80.0, 19.5, 0.1)

# Geometry
N, E, TVD, DLS = mincurv_positions(md, inc_deg, az)

# T&D for three scenarios
T_pu, M_pu = soft_string(md, inc_deg, mu_slide, mu_rot, mw_ppg, w_air_lbft, od_in, "pickup")
T_so, M_so = soft_string(md, inc_deg, mu_slide, mu_rot, mw_ppg, w_air_lbft, od_in, "slackoff")
T_ro, M_ro = soft_string(md, inc_deg, mu_slide*0.8, mu_rot*0.8, mw_ppg, w_air_lbft, od_in, "pickup")  # crude rotate-off

# ────────────────────────── Plots ───────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("3D Path")
    fig3d = go.Figure(data=[go.Scatter3d(x=E, y=N, z=TVD, mode="lines")])
    fig3d.update_layout(height=420, scene=dict(xaxis_title="East (ft)", yaxis_title="North (ft)", zaxis_title="TVD (ft)", zaxis=dict(autorange="reversed")))
    st.plotly_chart(fig3d, use_container_width=True)
with c2:
    st.subheader("2D — TVD vs Vertical Section")
    VS = N*np.cos(az[0]*DEG2RAD) + E*np.sin(az[0]*DEG2RAD)
    fig2d = go.Figure(data=[go.Scatter(x=VS, y=TVD, mode="lines")])
    fig2d.update_layout(height=420, xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2d, use_container_width=True)

st.subheader("Pickup / Slack-off Hookload (relative)")
figHL = go.Figure()
figHL.add_trace(go.Scatter(x=md, y=-T_so, name="Slack-off HL"))
figHL.add_trace(go.Scatter(x=md, y=-T_pu, name="Pickup HL"))
figHL.update_layout(xaxis_title="MD (ft)", yaxis_title="Hookload proxy (lbf)")
st.plotly_chart(figHL, use_container_width=True)

st.subheader("Surface Torque trend (rotating proxy)")
figT = go.Figure()
figT.add_trace(go.Scatter(x=md, y=abs(M_ro), name="Torque (rotate)"))
figT.update_layout(xaxis_title="MD (ft)", yaxis_title="Torque (ft-lbf, relative)")
st.plotly_chart(figT, use_container_width=True)

# Output table + download
df = pd.DataFrame({
    "MD_ft": md, "Inc_deg": inc_deg, "Az_deg": az,
    "TVD_ft": TVD, "North_ft": N, "East_ft": E, "DLS_deg_per_100ft": DLS,
    "T_pickup_lbf": T_pu, "T_slackoff_lbf": T_so, "M_rotate_ftlbf": M_ro
})
st.dataframe(df.head(12), use_container_width=True)
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", data=csv_bytes, file_name="tdrag_output.csv", mime="text/csv")

st.caption("Educational soft-string model. For production accuracy, adopt a full Johancsik implementation and detailed string/hole sectioning.")
