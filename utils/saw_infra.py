import os
import json
from pathlib import Path

if os.name == "nt":
    MPAR_LIST_FILE = "local/mpar.json"
else:
    MPAR_LIST_FILE = "/data/saw-infra/output/mpar.json"

with open(MPAR_LIST_FILE, encoding="utf-8") as f:
    MPARS: dict = json.load(f)
