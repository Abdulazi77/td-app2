# streamlit_app_full.py — “all required plots & features”
from __future__ import annotations
import math, io, numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ───────────────────────── Page / styles ─────────────────────────
st.set_page_config(page_title="Torque & Drag — Full Suite (Δs=1 ft)", layout="wide")
COLOUR_CASED, COLOUR_OH, COLOUR_LIMIT, COLOUR_SAFE = "#4cc9f0", "#a97142", "#ff0000", "#d0f0d0"
LINE_DASH_LIMIT, LINE_DOT_LIMIT = "dash", "dot"

DEG2RAD = math.pi/180.0
IN2FT   = 1.0/12.0
def clamp(x, lo, hi): return max(lo, min(hi, x))
def bf_from_mw(mw_ppg): return (65.5 - mw_ppg)/65.5
def I_in4(od_in, id_in): return (math.pi/64.0)*(od_in**4 - id_in**4)
def J_in4(od_in, id_in): return (math.pi/32.0)*(od_in**4 - id_in**4)
def Z_in3(od_in, id_in): return I_in4(od_in, id_in)/(max(od_in/2.0,1e-9))
def A_in2(od_in, id_in): return (math.pi/4.0)*(od_in**2 - id_in**2)

TOOL_JOINT_DB = {
    "NC38": {"od": 4.75, "id": 2.25, "T_makeup_ftlbf": 12000, "F_tensile_lbf": 350000, "T_yield_ftlbf": 20000},
    "NC40": {"od": 5.00, "id": 2.25, "T_makeup_ftlbf": 16000, "F_tensile_lbf": 420000, "T_yield_ftlbf": 25000},
    "NC50": {"od": 6.63, "id": 3.00, "T_makeup_ftlbf": 30000, "F_tensile_lbf": 650000, "T_yield_ftlbf": 45000},
}

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

# ───────────────────── Soft-string stepper ─────────────────────
def soft_string_stepper(md, inc_deg, kappa, cased_mask, comp_along, comp_props,
                        mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
                        scenario="slackoff", WOB_lbf=0.0, Mbit_ftlbf=0.0,
                        tortuosity_mode="off", tau=0.0, mu_open_boost=0.0,
                        rot_factor=1.0):
    ds = 1.0
    md = np.asarray(md); inc_deg = np.asarray(inc_deg); kappa = np.asarray(kappa)
    nseg = len(md)-1; inc = np.deg2rad(inc_deg[:-1])
    kappa_seg = kappa[:-1] if len(kappa)==len(md) else kappa
    if len(kappa_seg)!=nseg: kappa_seg = np.resize(kappa_seg, nseg)
    cased_seg = np.asarray(cased_mask)[:nseg]
    comp_arr  = np.asarray(list(comp_along))[:nseg]

    r_eff_ft = np.zeros(nseg); w_air = np.zeros(nseg); w_b = np.zeros(nseg)
    mu_s = np.zeros(nseg); mu_r = np.zeros(nseg)
    od_in = np.zeros(nseg); id_in = np.zeros(nseg)
    BF = bf_from_mw(mw_ppg)

    for i in range(nseg):
        comp = comp_arr[i]
        od = float(comp_props[comp]['od_in']); idd = float(comp_props[comp]['id_in'])
        w_air_ft = float(comp_props[comp]['w_air_lbft'])
        od_in[i] = od; id_in[i] = idd
        w_air[i] = w_air_ft; w_b[i] = w_air_ft*BF
        r_eff_ft[i] = 0.5*od*IN2FT
        if cased_seg[i]:
            mu_s[i] = mu_c_s; mu_r[i] = mu_c_r
        else:
            mu_s[i] = (mu_o_s*(rot_factor if "rotate" in scenario else 1.0)) + mu_open_boost
            mu_r[i] = (mu_o_r*(rot_factor if "rotate" in scenario else 1.0)) + mu_open_boost
        if not cased_seg[i]:
            if tortuosity_mode == "kappa": kappa_seg[i] *= (1.0 + tau)
            elif tortuosity_mode == "mu":  mu_s[i] *= (1.0 + tau); mu_r[i] *= (1.0 + tau)

    T = np.zeros(nseg+1); M = np.zeros(nseg+1)
    dT = np.zeros(nseg);   dM = np.zeros(nseg); N_side = np.zeros(nseg)
    if scenario == "onbottom":
        T[0] = -float(WOB_lbf); M[0] = float(Mbit_ftlbf)
    sgn_ax = {"pickup": +1.0, "slackoff": -1.0, "rotate_off": +1.0, "onbottom": +1.0}.get(scenario, -1.0)

    for i in range(nseg):
        N_raw = w_b[i]*math.sin(inc[i]) + T[i]*kappa_seg[i]
        N_side[i] = max(0.0, N_raw)  # clamp
        T_next = T[i] + (sgn_ax*w_b[i]*math.cos(inc[i]) + mu_s[i]*N_side[i]) * ds
        M_next = M[i] + (mu_r[i]*N_side[i]*r_eff_ft[i]) * ds
        dT[i] = T_next - T[i]; dM[i] = M_next - M[i]
        T[i+1] = T_next;       M[i+1] = M_next

    df = pd.DataFrame({
        "md_top_ft": md[:-1], "md_bot_ft": md[1:], "ds_ft": 1.0,
        "inc_deg": inc_deg[:-1], "kappa_rad_ft": kappa_seg,
        "w_air_lbft": w_air, "w_b_lbft": w_b,
        "mu_slide": mu_s, "mu_rot": mu_r,
        "N_lbf": N_side, "dT_lbf": dT, "T_next_lbf": T[1:],
        "dM_lbf_ft": dM, "M_next_lbf_ft": M[1:],
        "cased?": cased_seg, "comp": comp_arr,
        "od_in": od_in, "id_in": id_in,
    })
    return df, T, M

def neutral_point_md(md, T_arr):
    if len(md)<2 or len(T_arr)<2: return float('nan')
    for i in range(len(T_arr)-1):
        t1,t2 = T_arr[i], T_arr[i+1]
        if t1==0: return md[i]
        if t1*t2<0:
            frac = abs(t1)/(abs(t1)+abs(t2)+1e-9)
            return md[i] + frac*(md[i+1]-md[i])
    return float('nan')

# ───────────────────────────── UI ─────────────────────────────
st.header("Wellpath + Torque & Drag — Full Suite (Δs = 1 ft)")

# Trajectory
c1,c2,c3,c4 = st.columns(4)
profile = c1.selectbox("Profile", ["Build & Hold", "Build–Hold–Drop", "Horizontal (build + lateral)"])
kop_md  = c2.number_input("KOP MD (ft)", 0.0, 50000.0, 2000.0, 50.0)
build   = c3.number_input("Build rate (deg/100 ft)", 0.0, 30.0, 3.0, 0.1)
az_deg  = c4.number_input("Azimuth (deg)", 0.0, 360.0, 0.0, 1.0)

if profile=="Build & Hold":
    theta_hold = st.number_input("Final inclination (deg)", 0.0, 90.0, 30.0, 0.5)
    target_md  = st.number_input("Target MD (ft)", 500.0, 120000.0, 10000.0, 100.0)
    md, inc_deg, az = synth_build_hold(kop_md, build, theta_hold, target_md, az_deg)
elif profile=="Build–Hold–Drop":
    theta_hold = st.number_input("Hold inclination (deg)", 0.0, 90.0, 30.0, 0.5)
    drop_rate  = st.number_input("Drop rate (deg/100 ft)", 0.0, 30.0, 2.0, 0.1)
    target_md  = st.number_input("Target MD (ft)", 500.0, 120000.0, 10000.0, 100.0)
    md, inc_deg, az = synth_build_hold_drop(kop_md, build, theta_hold, drop_rate, target_md, az_deg)
else:
    lateral    = st.number_input("Lateral length (ft)", 0.0, 40000.0, 2000.0, 100.0)
    target_md  = st.number_input("Target MD (ft)", 500.0, 120000.0, 10000.0, 100.0)
    md, inc_deg, az = synth_horizontal(kop_md, build, lateral, target_md, az_deg)

N,E,TVD,DLS,kappa = mincurv_positions(md, inc_deg, az)
VS = N*np.cos(az[0]*DEG2RAD) + E*np.sin(az[0]*DEG2RAD)

# Casing shoe & hole/casing sizes
st.subheader("Casing / Hole sizes & friction masks")
shoe_md = st.slider("Casing shoe MD (ft)", 0.0, float(md[-1]), min(8000.0,float(md[-1]*0.6)), 50.0)
c1,c2,c3 = st.columns(3)
casing_id_in = c1.number_input("Casing ID (in)", 1.0, 20.0, 8.535, 0.001)
hole_diam_in = c2.number_input("Hole diameter (in)", 4.0, 20.0, 8.5, 0.01)
overgage_rot_in = c3.number_input("Overgage while rotating (in)", 0.0, 1.0, 0.15, 0.01)

# Friction, tortuosity, rotate-off
c1,c2,c3,c4 = st.columns(4)
mu_c_s = c1.number_input("μ casing (slide)", 0.02, 0.60, 0.15, 0.01)
mu_o_s = c2.number_input("μ open-hole (slide)", 0.05, 0.90, 0.30, 0.01)
mu_c_r = c3.number_input("μ casing (rot)", 0.01, 0.60, 0.06, 0.01)
mu_o_r = c4.number_input("μ open-hole (rot)", 0.02, 0.90, 0.12, 0.01)
tort_mode = st.selectbox("Tortuosity mode", ["off","kappa","mu"], index=0)
tau = st.slider("Tortuosity factor τ", 0.0, 0.5, 0.15, 0.01) if tort_mode!="off" else 0.0
rot_fc_factor = st.slider("Rotate-off friction factor", 0.3, 1.0, 0.8, 0.05)
mu_boost = st.slider("Open-hole μ boost (hole cleaning)", 0.0, 0.3, 0.0, 0.01)

# Mud & string
st.subheader("Mud & Drillstring")
mw_ppg = st.number_input("Mud weight (ppg)", 7.0, 20.0, 10.0, 0.1)
st.caption("Edit components (ID needed for stiffness/area): comp name, length, OD, ID, weight/ft")
df_comp = pd.DataFrame([
    {"comp":"DP","length_ft":float(md[-1]),"od_in":5.0,"id_in":4.0,"w_air_lbft":19.5}
])
df_comp = st.data_editor(df_comp, num_rows="dynamic")
# Build comp_along by repeating comp names by segment proportion
seg_count = len(md)-1
# simple allocation by length ratios
lengths = df_comp["length_ft"].to_numpy()
names   = df_comp["comp"].astype(str).to_numpy()
ratios  = lengths/np.maximum(1.0, lengths.sum())
counts  = np.maximum(1,(ratios*(seg_count))).astype(int)
comp_along = np.concatenate([np.full(c, n) for n,c in zip(names, counts)])[:seg_count]
# normalize last element
if len(comp_along)<seg_count:
    comp_along = np.pad(comp_along, (0, seg_count-len(comp_along)), constant_values=names[-1])
comp_props = {row["comp"]: {"od_in":float(row["od_in"]), "id_in":float(row["id_in"]), "w_air_lbft":float(row["w_air_lbft"])} 
              for _,row in df_comp.iterrows()}

cased_mask = md[:-1] <= shoe_md

# Scenarios
st.subheader("Scenario")
scenario = st.selectbox("Mode", ["Slack-off","Pickup","Rotate off-bottom","Rotate on-bottom"])
WOB = st.number_input("WOB (lbf)", 0.0, 150000.0, 6000.0, 100.0)
motor_on = st.checkbox("Motor on-bottom (bit torque = K*ΔP)"); 
K = st.number_input("K (ft-lbf/psi)", 0.0, 10000.0, 2.5, 0.1) if motor_on else 0.0
dP= st.number_input("ΔP (psi)", 0.0, 5000.0, 0.0, 1.0) if motor_on else 0.0
Mbit = (K*dP) if motor_on else 0.0
scen_key = {"Slack-off":"slackoff","Pickup":"pickup","Rotate off-bottom":"rotate_off","Rotate on-bottom":"onbottom"}[scenario]

# Run stepper
df_itr, T_arr, M_arr = soft_string_stepper(
    md, inc_deg, kappa, cased_mask, comp_along, comp_props,
    mu_c_s, mu_o_s, mu_c_r, mu_o_r, mw_ppg,
    scenario=scen_key, WOB_lbf=WOB, Mbit_ftlbf=Mbit,
    tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost,
    rot_factor=rot_fc_factor
)

depth = df_itr["md_bot_ft"].to_numpy()
surf_hookload = max(0.0, -T_arr[-1]); surf_torque = abs(M_arr[-1])
np_pick = neutral_point_md(md, T_arr)

# ───────────────────────── Classic geometry panels ─────────────────────────
c1,c2 = st.columns(2)
with c1:
    st.markdown("### 3D path (split by shoe)")
    idx_shoe = int(np.clip(np.searchsorted(md, shoe_md), 0, len(md)-1))
    fig3d = go.Figure()
    fig3d.add_trace(go.Scatter3d(x=E[:idx_shoe+1], y=N[:idx_shoe+1], z=TVD[:idx_shoe+1], mode="lines", name="Cased"))
    fig3d.add_trace(go.Scatter3d(x=E[idx_shoe:], y=N[idx_shoe:], z=TVD[idx_shoe:], mode="lines", name="Open hole"))
    fig3d.update_layout(height=420, scene=dict(xaxis_title="East", yaxis_title="North", zaxis_title="TVD", zaxis=dict(autorange="reversed")),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0))
    st.plotly_chart(fig3d, use_container_width=True)
with c2:
    st.markdown("### 2D profile — TVD vs Vertical Section (split)")
    VS = N*np.cos(az[0]*DEG2RAD) + E*np.sin(az[0]*DEG2RAD)
    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=VS[:idx_shoe+1], y=TVD[:idx_shoe+1], mode="lines", name="Cased"))
    fig2d.add_trace(go.Scatter(x=VS[idx_shoe:], y=TVD[idx_shoe:], mode="lines", name="Open hole"))
    fig2d.update_layout(height=420, xaxis_title="Vertical Section (ft)", yaxis_title="TVD (ft)", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2d, use_container_width=True)

# ───────────────────────── Stacked “Required Plots” panel ─────────────────────────
st.markdown("### Stacked T&D Panel — Hookload, Torque μ‑sweep, Buckling/Severity")
tj_name = st.selectbox("Tool‑joint size for limits", list(TOOL_JOINT_DB.keys()), index=1)
sf_joint = st.slider("Make‑up torque SF", 1.0, 2.5, 1.5, 0.1)
rig_torque_lim = st.number_input("Rig torque limit (ft‑lbf)", 1000.0, 100000.0, 30000.0, 100.0)
rig_pull_lim   = st.number_input("Rig pull limit (lbf)",     10000.0, 1000000.0, 400000.0, 1000.0)

# μ sweep for rotate‑off torque
mu_band = st.multiselect("μ sweep for off‑bottom torque", [0.15,0.20,0.25,0.30,0.35,0.40], default=[0.20,0.25,0.30,0.35])

T_makeup_sf = TOOL_JOINT_DB[tj_name]['T_makeup_ftlbf']/sf_joint

def run_td_off_bottom(mu_slide, mu_rot):
    df_tmp, T_tmp, M_tmp = soft_string_stepper(
        md, inc_deg, kappa, (md[:-1]<=shoe_md), comp_along, comp_props,
        mu_c_s, mu_slide, mu_c_r, mu_rot, mw_ppg,
        scenario="rotate_off", tortuosity_mode=tort_mode, tau=tau, mu_open_boost=mu_boost, rot_factor=rot_fc_factor
    )
    return df_tmp["md_bot_ft"].to_numpy(), np.abs(df_tmp["M_next_lbf_ft"].to_numpy())

# Buckling & severity prerequisites
hole_d_profile = np.where(df_itr["cased?"].to_numpy(), 2.0*casing_id_in, hole_diam_in + (overgage_rot_in if "rotate" in scen_key else 0.0))
Epsi = 30.0e6
od_local = df_itr["od_in"].to_numpy()
id_local = df_itr["id_in"].to_numpy()
I_local = np.array([I_in4(o,i) for o,i in zip(od_local,id_local)])
Z_local = np.array([Z_in3(o,i) for o,i in zip(od_local,id_local)])
A_local = np.array([A_in2(o,i) for o,i in zip(od_local,id_local)])
J_local = np.array([J_in4(o,i) for o,i in zip(od_local,id_local)])
r_ft_local = np.maximum(1e-4, 0.5*(hole_d_profile - od_local) * IN2FT)
rot_factor = rot_fc_factor if "rotate" in scen_key else 1.0
lam_hel_ui = 2.83  # Menand coefficient
EI_lbf_ft2 = (Epsi * I_local) / (12.0**2)
inc_local_rad = np.deg2rad(np.maximum(df_itr['inc_deg'].to_numpy(), 1e-6))
denom = np.maximum(r_ft_local * np.sin(inc_local_rad), 1e-9)
Fs = (2.0 * EI_lbf_ft2 * df_itr['w_b_lbft'].to_numpy()) / denom
Fh = (lam_hel_ui * EI_lbf_ft2 * df_itr['w_b_lbft'].to_numpy()) / denom
Fs *= rot_factor; Fh *= rot_factor
M_b_lbf_in  = df_itr["N_lbf"].to_numpy() * r_ft_local * 12.0
sigma_b_psi = np.divide(M_b_lbf_in, np.maximum(Z_local, 1e-9))
r_in     = od_local / 2.0
T_lbf_in = np.abs(df_itr["M_next_lbf_ft"].to_numpy()) * 12.0
tau_psi  = np.divide(T_lbf_in * r_in, np.maximum(J_local, 1e-9))
sigma_ax_psi = np.divide(df_itr["T_next_lbf"].to_numpy(), np.maximum(A_local, 1e-9))
sigma_ax_wf  = sigma_ax_psi + np.sign(sigma_ax_psi)*np.abs(sigma_b_psi)
sigma_vm_psi = np.sqrt(sigma_ax_wf**2 + 3.0*tau_psi**2)
SB  = sigma_b_psi / 30000.0
SV  = sigma_vm_psi / 60000.0
SN  = np.abs(df_itr["N_lbf"].to_numpy()) / 5000.0
BSI = 1.0 + 3.0 * np.clip(0.35*SB + 0.45*SV + 0.20*SN, 0.0, 1.0)

fig = make_subplots(rows=3, cols=1, shared_yaxes=True, row_heights=[0.34,0.33,0.33], vertical_spacing=0.02,
                    subplot_titles=("Hookload (k‑lbf)", "Torque μ‑sweep (k lbf‑ft)", "Buckling & Severity"))
# Row 1: Hookload
fig.add_trace(go.Scatter(x=np.maximum(0.0, -df_itr['T_next_lbf'])/1000.0, y=depth, name="Hookload (k‑lbf)", mode="lines", line=dict(color=COLOUR_CASED)), row=1, col=1)
fig.add_shape(type="rect", x0=rig_pull_lim/1000.0, x1=(rig_pull_lim/1000.0)*1.2, y0=depth.min(), y1=depth.max(), fillcolor="red", opacity=0.08, row=1, col=1, line_width=0)
# Row 2: Torque μ-sweep
for mu in mu_band:
    dmu, tmu = run_td_off_bottom(mu, mu)
    fig.add_trace(go.Scatter(x=tmu/1000.0, y=dmu, name=f"μ={mu:.2f}", mode="lines"), row=2, col=1)
fig.add_vline(x=(TOOL_JOINT_DB[tj_name]['T_makeup_ftlbf']/sf_joint)/1000.0, line_dash=LINE_DASH_LIMIT, line_color=COLOUR_LIMIT, annotation_text="MU/SF", row=2, col=1)
fig.add_vline(x=rig_torque_lim/1000.0, line_dash=LINE_DOT_LIMIT, line_color=COLOUR_LIMIT, annotation_text="TD limit", row=2, col=1)
# Row 3: Buckling & severity
fig.add_trace(go.Scatter(x=Fs/1000.0, y=depth, name="Fs (k‑lbf)", line=dict(dash="dash", color="gray")), row=3, col=1)
fig.add_trace(go.Scatter(x=Fh/1000.0, y=depth, name="Fh (k‑lbf)", line=dict(dash="dot", color="gray")), row=3, col=1)
fig.add_trace(go.Scatter(x=df_itr['N_lbf']/1000.0, y=depth, name="Side‑force (k‑lbf)", line=dict(color="purple")), row=3, col=1)
fig.add_trace(go.Scatter(x=sigma_b_psi/1000.0, y=depth, name="Bending (ksi)", line=dict(color="brown")), row=3, col=1)
fig.add_trace(go.Scatter(x=sigma_vm_psi/1000.0, y=depth, name="von Mises (ksi)", line=dict(color="black")), row=3, col=1)
fig.add_trace(go.Scatter(x=BSI, y=depth, name="BSI (1–4)", line=dict(width=4, color="red")), row=3, col=1)
for r in (1,2,3):
    fig.update_yaxes(autorange="reversed", title_text="Depth (ft)", row=r, col=1)
fig.update_xaxes(title_text="Hookload (k‑lbf)", row=1, col=1)
fig.update_xaxes(title_text="Torque (k lbf‑ft)", row=2, col=1)
fig.update_xaxes(title_text="k‑lbf / ksi / BSI", row=3, col=1)
fig.update_layout(height=900, template="simple_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(l=10,r=10,t=40,b=10), hovermode="closest")
st.plotly_chart(fig, use_container_width=True, key="stacked-main")

# ───────────────────────── Diagnostics panel ─────────────────────────
c1,c2 = st.columns(2)
with c1:
    st.markdown("### Tension vs Inclination")
    figTI = go.Figure()
    figTI.add_trace(go.Scatter(x=inc_deg, y=T_arr[:len(inc_deg)], name="Tension (pickup/slackoff per scenario)"))
    figTI.update_layout(xaxis_title="Inclination (deg)", yaxis_title="Tension (lbf)")
    st.plotly_chart(figTI, use_container_width=True)
with c2:
    st.markdown("### Tortuosity: DLS & κ")
    figD = go.Figure()
    figD.add_trace(go.Scatter(x=md, y=DLS, name="DLS (deg/100ft)"))
    figD.add_trace(go.Scatter(x=md, y=kappa, name="κ (1/ft)", yaxis="y2"))
    figD.update_layout(xaxis=dict(title="MD (ft)"), yaxis=dict(title="DLS (deg/100ft)"),
                       yaxis2=dict(title="κ (1/ft)", overlaying="y", side="right"))
    st.plotly_chart(figD, use_container_width=True)

# ───────────────────────── Trip‑log torque (simple) ─────────────────────────
st.markdown("### Trip‑log torque (simple simulator)")
speed = st.slider("Trip speed (ft/min)", 5, 200, 60, 5); dt=1.0
t = np.arange(0, max(1, math.ceil(md[-1]/speed))+dt, dt)
bit_md = np.clip(t*speed, 0, md[-1]); idx = np.searchsorted(md, bit_md)
trip_torque = np.abs(M_arr[idx])
figTrip = go.Figure(go.Scatter(x=t, y=trip_torque, name="Torque during trip"))
figTrip.update_layout(xaxis_title="Time (min)", yaxis_title="Torque (ft‑lbf)")
st.plotly_chart(figTrip, use_container_width=True)

# Surface numbers & neutral point
st.success(f"Surface Hookload: {surf_hookload:,.0f} lbf — Surface Torque: {surf_torque:,.0f} lbf‑ft — Neutral Point: {np_pick:,.0f} ft")

# Export table & figure bundle
df_out = df_itr.copy()
df_out["VS_ft"] = VS[:len(df_out)]
st.download_button("Download results CSV", df_out.to_csv(index=False).encode("utf-8"), "tdrag_full_output.csv", "text/csv")

st.caption("Model: Johancsik soft‑string (Δs=1 ft). Includes μ masks (cased vs open‑hole), rotate‑off factor, motor bit torque, μ‑sweep, buckling & severity visuals, and all classic plots.")
