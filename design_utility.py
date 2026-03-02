import streamlit as st


pages = [
    st.Page("contents/sp_viewer.py", title="Sp Viewer"),
    st.Page("contents/pitch_profile.py", title="Pitch Profile"),
    st.Page("contents/tcf_to_vfk2.py", title="TCF to Vf/K2"),
    st.Page("contents/FI_spec.py", title="FI Spec"),
    st.Page("contents/corner_lot.py", title="Corner Lot Sim"),
]
st.navigation(pages).run()
