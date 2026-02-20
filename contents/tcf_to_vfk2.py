import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from utils.mpar import Mpar
from components import mpar_selector


st.set_page_config("SAW Design Utility", page_icon=":toolbox:", layout="wide")
st.title("TCF → rVf / rK2")
st.divider()


mpar_path = mpar_selector.render()

cols = st.columns(2)
with cols[0]:
    tcfs_ppm = st.number_input("TCFs [ppm/C]", value=10., step=0.1)
with cols[1]:
    tcfp_ppm = st.number_input("TCFp [ppm/C]", value=10., step=0.1)

eta = st.number_input("DF", value=0.5, step=0.01)

if mpar_path is None:
    st.stop()

mpar = Mpar(mpar_path)
Lmin, Lmax = mpar.pitch_range
Lstep = 0.2
L = np.arange(Lmin, Lmax + Lstep + 1e-9, Lstep)

tc_rvf, tc_rk2 = mpar.tc_rvf_rk2(L, eta, tcfs_ppm, tcfp_ppm)


df = pd.DataFrame({
    "L": L,
    "tc_rvf": tc_rvf,
    "tc_rk2": tc_rk2
})

# 共通設定
xaxis_layout = {}  # dict(
#    tick0=Lmin,
#    dtick=Lstep,
# )
yaxis_layout = dict(
    tickformat=".2e",
    exponentformat="e",
)
marker_size = 8
line_width = 2

# ループで重複削減
plots = [
    # (y列名, タイトル, y軸タイトル)
    ("tc_rvf", "TC_rVf", "TC_rVf [/C]"),
    ("tc_rk2", "TC_rK2", "TC_rK2 [/C]")
]

for y, title, ytitle in plots:
    fig = px.line(df, x="L", y=y, markers=True, title=title)
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=line_width),
        marker=dict(size=marker_size, symbol="circle"),
        text=df[y],                 # マーカーに値を表示
        texttemplate="%{text:.3g}",  # 表示桁は適宜変更可（例: .2e / .3f 等）
        textposition="top center",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=16, r=16, t=48, b=16),
        xaxis=xaxis_layout,
        yaxis=yaxis_layout,
    )
    fig.update_xaxes(
        title="L [um]",
        showgrid=True,
        gridcolor="rgba(255,255,255,0.1)",
        griddash="solid"
    )
    fig.update_yaxes(title=ytitle)

    st.plotly_chart(fig, use_container_width=True)
