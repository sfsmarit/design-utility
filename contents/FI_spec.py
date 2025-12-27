import streamlit as st
from utils.spec import Spec
import tempfile
from pathlib import Path
import re

st.markdown(
    """
    ---
    ### FI Spec
    ---    
    """
)


file = st.file_uploader("FI spec file", type="csv")
if not file:
    st.stop()

# Load spec file
suffix = Path(file.name).suffix
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(file.getbuffer())

ckt = st.text_input("Circuit name", "Filter_L")

text = st.text_input("Spec words to exlude", "IR, OPB, SHB, RIPPLE")
if text:
    words_to_exclude = re.split(r"[,\s]+", text)
else:
    words_to_exclude = []

spec = Spec(tmp.name)
spec.filter(words_to_exlude=words_to_exclude)

st.write("**SNSS spec format**")
st.code(spec.to_SNSS_string(ckt), language="")
