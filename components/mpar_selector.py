import streamlit as st
import os
from pathlib import Path

from utils.saw_infra import MPARS


def render() -> str:
    """Renders the Mpar selector component.
    Returns:
        str: The selected Mpar filepath.
    """
    mpar_names = list(MPARS.keys())
    mpar_name = st.selectbox("Mpar", mpar_names)
    mpar = MPARS[mpar_name]
    layer_texts = [f"{k}={v}" for k, v in mpar["stack"].items()]
    st.caption(" / ".join(layer_texts))

    path = mpar.get("path", "")
    if path and os.name == "nt":
        path = f"local/{Path(path).name}"
    return path
