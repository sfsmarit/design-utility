import streamlit as st
from utils.spec import Spec
import re

st.set_page_config("SAW Design Utility", page_icon=":toolbox:", layout="wide")
st.title("FI Spec → SNSS")
st.divider()


file = st.file_uploader("FI spec file", type="csv")
if not file:
    st.stop()

ckt = st.text_input("Circuit name", "Filter_L")

text = st.text_input("Spec words to exlude", "IR, OPB, SHB, RIPPLE")
if text:
    words_to_exclude = re.split(r"[,\s]+", text)
else:
    words_to_exclude = []

spec = Spec(file)  # type: ignore
spec.filter(words_to_exlude=words_to_exclude)

st.write("**SNSS spec format**")
st.code(spec.to_SNSS_string(ckt), language="")
