from typing import Any
import re
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path

from . import com


class Mpar:
    ETA_REF = 0.5

    def __init__(self, file: str):
        self.file = file

        self.text: str = ""
        self.id: str = ""
        self.tech_ver: str = ""

        self.wafer: str = ""
        self.stack: dict[str, Any] = {}
        self.dfc_file: str = ""
        self.idt_trim_params: list[str] = []

        self.pitch_range: list[float] = [0.0, 0.0]  # [um]
        self.eta_range: list[float] = [0.0, 0.0]

        self.idt_thickness: float = 0.0  # [nm]
        self.rho: float = 0.0

        self.pld_offset: float = 0.0
        self.qc_offset: float = 0.0

        self.rk2_formula: str = ""
        self.rpld_formula: str = ""
        self.rqc_formula: str = ""

        self.vf_matrix: ArrayLike = np.array([])
        self.epr_matrix: ArrayLike = np.array([])
        self.d_matrix: ArrayLike = np.array([])
        self.qc_matrix: ArrayLike = np.array([])
        self.k11_matrix: ArrayLike = np.array([])
        self.k12_matrix: ArrayLike = np.array([])
        self.k2_matrix: ArrayLike = np.array([])
        self.dfc_matrix: ArrayLike = np.array([])
        self.df_matrix: ArrayLike = np.array([])
        self.vm_vector: ArrayLike = np.array([])
        self.dm_matrix: ArrayLike = np.array([])
        self.phkd_matrix: ArrayLike = np.array([])
        self.lcap_matrix: ArrayLike = np.array([])

        if file:
            self.load(file)

    # ----------------------------------------------------------
    # Acoustics functions (unchanged logic, typed)
    # ----------------------------------------------------------

    def fs(self,
           pitch: ArrayLike,
           eta: ArrayLike,
           rvf: ArrayLike = 1,
           rk11: ArrayLike = 1,
           rk12: ArrayLike = 1) -> ArrayLike:
        """Resonant frequency [MHz]"""
        vf = np.asarray(self.vf(pitch, eta)) * rvf
        k11 = np.asarray(self.k11(pitch, eta)) * rk11
        k12 = np.asarray(self.k12(pitch, eta)) * rk12
        return com.fs(pitch, vf, k11, k12)

    def fp(self,
           pitch: ArrayLike,
           eta: ArrayLike,
           rvf: ArrayLike = 1,
           rk11: ArrayLike = 1,
           rk12: ArrayLike = 1,
           rk2: ArrayLike = 1) -> ArrayLike:
        """Anti-resonant frequency [MHz]"""
        vf = np.asarray(self.vf(pitch, eta)) * rvf
        k11 = np.asarray(self.k11(pitch, eta)) * rk11
        k12 = np.asarray(self.k12(pitch, eta)) * rk12
        k2 = np.asarray(self.k2(pitch, eta)) * rk2
        return com.fp(pitch, vf, k11, k12, k2)

    def fr(self,
           pitch: ArrayLike,
           eta: ArrayLike,
           rvf: ArrayLike = 1,
           rk11: ArrayLike = 1,
           rk12: ArrayLike = 1) -> ArrayLike:
        """Stopband frequency [MHz]"""
        vf = np.asarray(self.vf(pitch, eta)) * rvf
        k11 = np.asarray(self.k11(pitch, eta)) * rk11
        k12 = np.asarray(self.k12(pitch, eta)) * rk12
        return com.fr(pitch, vf, k11, k12)

    def k2_eff(self,
               pitch: ArrayLike,
               eta: ArrayLike,
               rvf: ArrayLike = 1,
               rk11: ArrayLike = 1,
               rk12: ArrayLike = 1,
               rk2: ArrayLike = 1) -> ArrayLike:
        """Effective coupling coefficient K2 [%]"""
        fs = self.fs(pitch, eta, rvf, rk11, rk12)
        fp = self.fp(pitch, eta, rvf, rk11, rk12, rk2)
        return com.k2_eff(fs, fp)

    def fs_gradient(self,
                    pitch: ArrayLike,
                    eta: ArrayLike,
                    vf_grad: ArrayLike = 0,
                    k11_grad: ArrayLike = 0,
                    k12_grad: ArrayLike = 0) -> ArrayLike:
        """Resonant frequency gradient [MHz/um or MHz/unit eta]"""
        vf = np.asarray(self.vf(pitch, eta))
        k11 = np.asarray(self.k11(pitch, eta))
        k12 = np.asarray(self.k12(pitch, eta))
        return com.fs_gradient(pitch, vf, k11, k12, vf_grad, k11_grad, k12_grad)

    def vf(self, pitch: ArrayLike, eta: ArrayLike) -> ArrayLike:
        """Sound velocity [m/s]"""
        return self._sumproduct(self.vf_matrix, pitch, eta)

    def k11(self, pitch: ArrayLike, eta: ArrayLike) -> ArrayLike:
        """Coupling coefficient k11"""
        return self._sumproduct(self.k11_matrix, pitch, eta)

    def k12(self, pitch: ArrayLike, eta: ArrayLike) -> ArrayLike:
        """Coupling coefficient k12"""
        return self._sumproduct(self.k12_matrix, pitch, eta)

    def k2(self, pitch: ArrayLike, eta: ArrayLike) -> ArrayLike:
        """Coupling coefficient K2"""
        return self._sumproduct(self.k2_matrix, pitch, eta)

    def d(self, pitch: ArrayLike, eta: ArrayLike) -> ArrayLike:
        """Propagation loss d [dB/um]"""
        return self._sumproduct(self.d_matrix, pitch, eta)

    def qc(self, pitch: ArrayLike, eta: ArrayLike) -> ArrayLike:
        """Quality factor Qc"""
        return self._sumproduct(self.qc_matrix, pitch, eta)

    def tc_rvf_rk2(self, pitch, eta, tcfs_ppm, tcfp_ppm):
        """Calculate the relative change in vf and k2 from the temperature coefficients.
        Assuming k11 and k12 are temperature-independent, and that the change in fs and fp is entirely due to changes in vf and k2.

        Args:
            pitch: IDT pitch [um]
            eta: IDT duty factor
            tcfs_ppm: Temperature coefficient of fs [ppm/C]
            tcfp_ppm: Temperature coefficient of fp [ppm/C]
            t: Current temperature [C]
            t0: Reference temperature [C]

        Returns:
            tcrvf: Relative change in vf per degree C [/C]
            tcrk2: Relative change in k2 per degree C [/C]
        """
        L = pitch
        Lm = L * 1e-6

        vf = self.vf(L, eta)
        k11 = self.k11(L, eta)
        k12 = self.k12(L, eta)
        k2 = self.k2(L, eta)
        fs = self.fs(L, eta)
        fp = self.fp(L, eta)

        tcfs = tcfs_ppm*1e-6 * fs  # [MHz/C]
        tcfp = tcfp_ppm*1e-6 * fp  # [MHz/C]

        tcvf = L * tcfs  # [m/s/C]
        tcrvf = tcvf / vf  # [/C]

        a = 1 - (k11 + k12) / 2 / np.pi + 4 * k2 / np.pi**2
        tck2 = np.pi**2 * Lm / 4 / vf * (tcfp*1e6 - a*tcfs*1e6)
        tcrk2 = tck2 / k2

        return tcrvf, tcrk2

    # ----------------------------------------------------------
    # ndarray-safe sumproduct (minimal change, typed)
    # ----------------------------------------------------------
    def _sumproduct(self,
                    matrix: ArrayLike,
                    pitch: ArrayLike,
                    eta: ArrayLike) -> ArrayLike:
        """
        Polynomial evaluation supporting scalar or ndarray.
        matrix: (5,5)
        pitch, eta: float or array-like
        """
        pitch_arr = np.asarray(pitch, dtype=float)
        eta_arr = np.asarray(eta, dtype=float)

        hL = (self.idt_thickness * 1e-3) / pitch_arr
        de = eta_arr - self.ETA_REF

        px = hL[..., None] ** np.arange(5)       # shape (..., 5)
        py = de[..., None] ** np.arange(5)       # shape (..., 5)

        # einsum over last axes: result → shape (...)
        val = np.einsum("...i,ij,...j->...", py, matrix, px)

        if np.isscalar(pitch) and np.isscalar(eta):
            return float(val)
        return val

    # ----------------------------------------------------------
    # Parser (unchanged, lightly typed)
    # ----------------------------------------------------------
    def load(self, file: str) -> None:
        if Path(file).suffix.lower() != ".mpar":
            raise ValueError("File must have a .mpar extension")

        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            lines = [ln.rstrip("\n") for ln in f]

        i = 0
        current_section = ""
        current_block: str | None = None
        buf: list[str] = []

        def flush_block(block_name: str | None, rows: list[str], section: str) -> None:
            if not block_name:
                return
            mat = self._rows_to_matrix(rows)

            name = block_name.upper()
            if name == "VF":
                self.vf_matrix = mat
            elif name == "EPR":
                self.epr_matrix = mat
            elif name == "D":
                self.d_matrix = mat
            elif name == "QC":
                self.qc_matrix = mat
            elif name == "K11":
                self.k11_matrix = mat
            elif name == "K12":
                self.k12_matrix = mat
            elif name == "K2":
                self.k2_matrix = mat
            elif name in ("D_S", "DS"):
                self.dfc_matrix = mat
            elif name == "DF":
                self.df_matrix = mat
            elif name == "VM":
                self.vm_vector = mat.squeeze()
            elif name == "DM":
                self.dm_matrix = mat
            elif name == "PHKD":
                self.phkd_matrix = mat
            elif name == "L_CAP":
                self.lcap_matrix = mat

        in_data_region = False

        while i < len(lines):
            line = lines[i].strip()

            if line == "":
                i += 1
                continue

            if line.startswith("#") and line.endswith("#"):
                flush_block(current_block, buf, current_section)
                buf.clear()
                current_block = None
                current_section = line.strip("#").upper()
                in_data_region = True
                i += 1
                continue

            if line.startswith("*") and line.endswith("*"):
                flush_block(current_block, buf, current_section)
                buf.clear()
                current_block = line.strip("*").strip()
                i += 1
                continue

            if in_data_region and current_block is not None:
                if not line.startswith("#"):
                    buf.append(lines[i])
                i += 1
                continue

            self._parse_header_line(lines[i])
            i += 1

        flush_block(current_block, buf, current_section)

    # ----------------------------------------------------------
    # Header parsing
    # ----------------------------------------------------------
    def _parse_header_line(self, raw: str) -> None:
        s = raw.strip()

        m = re.search(r"Wafer for Mpar extraction:\s*ID\s*=\s*([^\s]+)", s, re.IGNORECASE)
        if m:
            self.wafer = m.group(1).strip()
            return

        if s.startswith("(") and s.endswith(")") and "=" in s:
            self.stack = self._parse_stack(s[1:-1])
            return

        if s.lower().startswith("idt trim"):
            inside = s[s.index("(")+1:s.index(")")]
            self.idt_trim_params = [param.strip() for param in inside.split(",") if param.strip()]
            return

        m = re.search(r'-dfc_fnc\s+"([^"]+)"', s, re.IGNORECASE)
        if m:
            self.dfc_file = m.group(1).strip()
            return

        m = re.search(r"IDT\s+active\s+area:\s*L\s*=\s*([0-9.]+)\s*-\s*([0-9.]+)\s*um", s, re.IGNORECASE)
        if m:
            self.pitch_range = [float(m.group(1)), float(m.group(2))]
            return

        m = re.search(r"IDT\s+Duty\s+factor\s+ac?ctive\s+area:\s*([0-9.]+)\s*-\s*([0-9.]+)", s, re.IGNORECASE)
        if m:
            self.eta_range = [float(m.group(1)), float(m.group(2))]
            return

        m = re.search(r"Metal Thickness in net file:\s*([0-9.]+)\s*nm", s, re.IGNORECASE)
        if m:
            self.idt_thickness = float(m.group(1))
            return

        m = re.search(
            r"LOSS Parameter:\s*Pld\s*=\s*([0-9.eE+-]+)\s*,\s*Resistance\s*=\s*([0-9.eE+-]+)\s*,\s*Qc\s*=\s*([0-9.eE+-]+)",
            s, re.IGNORECASE)
        if m:
            self.pld_offset = float(m.group(1))
            self.rho = float(m.group(2))
            self.qc_offset = float(m.group(3))
            return

        m = re.match(r"\s*rk2\s*=\s*(.+)$", s, re.IGNORECASE)
        if m:
            self.rk2_formula = m.group(1).strip()
            return

        m = re.match(r"\s*rPld\s*=\s*(.+)$", s, re.IGNORECASE)
        if m:
            self.rpld_formula = m.group(1).strip()
            return

        m = re.match(r"\s*rQc\s*=\s*(.+)$", s, re.IGNORECASE)
        if m:
            self.rqc_formula = m.group(1).strip()
            return

    # ----------------------------------------------------------
    # Stack parser
    # ----------------------------------------------------------
    def _parse_stack(self, s: str) -> dict[str, float]:
        out: dict[str, float] = {}
        parts = [p.strip() for p in s.split("/") if p.strip()]
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            v = v.strip()

            m = re.match(r"([0-9.]+)\s*(nm|n)?$", v, re.IGNORECASE)
            if m:
                num = float(m.group(1))
                out[k] = num
            else:
                m2 = re.search(r"([0-9.]+)", v)
                if m2:
                    out[k] = float(m2.group(1))
        return out

    # ----------------------------------------------------------
    # Numeric row parsing
    # ----------------------------------------------------------
    def _parse_float_list(self, line: str) -> list[float]:
        toks = re.split(r"[\s\t]+", line.strip())
        vals = []
        for t in toks:
            if t == "":
                continue
            vals.append(float(t))
        return vals

    def _rows_to_matrix(self, rows: list[str]) -> ArrayLike:
        if not rows:
            return np.array([])

        data = [self._parse_float_list(r) for r in rows if r.strip() and not r.strip().startswith("#")]
        lens = {len(r) for r in data}
        if len(lens) != 1:
            raise ValueError(f"Row length mismatch in matrix: lengths={sorted(lens)}")

        return np.array(data, dtype=float)


if __name__ == "__main__":
    file = ""
    mpar = Mpar(file)

    L = 5.6
    eta = 0.6
    tcfs_ppm = 23
    tcfp_ppm = -6
    t0 = 25
    t = 85

    tc_rvf, tc_rk2 = mpar.tc_rvf_rk2(L, eta, tcfs_ppm, tcfp_ppm, t, t0)
    print(f"{tc_rvf=:.1e}/C, {tc_rk2=:.1e}/C")
