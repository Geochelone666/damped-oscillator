"""Numbers quoted in the paper and READMEs must come from the saved CSVs.

The CSVs are the single source (tests/test_saved_data.py recomputes them from
the solvers). This file recomputes each quoted figure from the CSVs and checks
that the text contains it, so a regenerated CSV cannot leave a stale figure in
prose unnoticed. The .docx draft is checked the same way for the figures listed in
PaperNumbersMatchCsvTests.test_docx_*; it is read with zipfile, no extra package.
"""
import csv
from pathlib import Path
import re
import unittest
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _rows(name):
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _text(*parts):
    return (ROOT.joinpath(*parts)).read_text(encoding="utf-8")


def _orders():
    rows = _rows("metrics.csv")
    out = {}
    for method in ("euler", "verlet", "rk4"):
        out[method] = []
        for zeta in (0.1, 1.0, 2.0):
            sel = [r for r in rows if r["method"] == method and float(r["zeta"]) == zeta]
            h = np.log([float(r["h"]) for r in sel])
            e = np.log([float(r["E_inf"]) for r in sel])
            out[method].append(np.polyfit(h, e, 1)[0])
    return out


class PaperNumbersMatchCsvTests(unittest.TestCase):
    def test_convergence_order_table_in_tex(self):
        tex = _text("paper", "main.tex")
        for method, row_name in (("euler", "Explicit Euler"),
                                 ("verlet", "Velocity-Verlet-type"), ("rk4", "RK4")):
            expected = " & ".join(f"{p:.2f}" for p in _orders()[method])
            with self.subTest(method=method):
                self.assertIn(f"{row_name} & {expected} \\\\", tex)

    def test_stability_scan_ranges_in_tex_and_data_readme(self):
        scan = _rows("stability_scan.csv")
        euler = [float(r["zeta_fit_err"]) for r in scan if r["method"] == "euler"]
        verlet = [float(r["zeta_fit_err"]) for r in scan if r["method"] == "verlet"]
        rk4 = [float(r["zeta_fit_err"]) for r in scan if r["method"] == "rk4"]
        tex, readme = _text("paper", "main.tex"), _text("data", "README.md")
        self.assertIn(f"{min(euler):.6f} to {max(euler):.6f}", readme)
        self.assertIn(f"{min(verlet):.6f} to {max(verlet):.6f}", readme)
        self.assertIn(f"{min(verlet):.4f}$ to $\\;{max(verlet):.4f}".replace("\\;", ""), tex)
        self.assertLess(max(rk4), 8e-5)
        self.assertIn(f"{min(euler):.3f}$ ($h=0.16$)", tex)
        self.assertIn(f"{max(euler):.3f}$ ($h=0.25$)", tex)

    def test_euler_scan_deviation_from_amplification_theory(self):
        zeta = 0.1
        worst = 0.0
        for row in _rows("stability_scan.csv"):
            if row["method"] != "euler":
                continue
            h = float(row["h"])
            predicted = abs(1 - (-np.log(1 - 2 * zeta * h + h * h) / (2 * h)) / zeta)
            worst = max(worst, abs(float(row["zeta_fit_err"]) - predicted))
        self.assertIn(f"{worst * 1e5:.1f}\\times10^{{-5}}", _text("paper", "main.tex"))
        self.assertLess(worst, 4e-5)  # data/README.md: "within 4e-5"

    def test_energy_drift_figures_in_tex(self):
        drift = {(r["method"], float(r["h"])): float(r["max_rel_E_err"])
                 for r in _rows("energy_drift.csv")}
        tex = _text("paper", "main.tex")
        self.assertIn(f"{drift[('euler', 0.2)] / 1e8:.1f}\\times10^8", tex)
        self.assertIn(f"{drift[('rk4', 0.2)] * 1e4:.1f}\\times10^{{-4}}", tex)

    def test_cost_accuracy_example_in_tex(self):
        rows = {(r["method"], float(r["zeta"]), float(r["h"])): float(r["E_inf"])
                for r in _rows("metrics.csv")}
        tex = _text("paper", "main.tex")
        self.assertIn(f"{rows[('rk4', 0.1, 0.05)] * 1e7:.1f}\\times10^{{-7}}", tex)
        self.assertIn(f"{rows[('euler', 0.1, 0.0125)] * 1e2:.1f}\\times10^{{-2}}", tex)


def _docx_paragraphs():
    xml = zipfile.ZipFile(ROOT / "paper" / "damped-oscillator-draft-v01.docx").read(
        "word/document.xml").decode("utf-8")
    # The saved XML uses a generated namespace prefix (ns0), so match any prefix.
    paras = [m.group(0) for m in re.finditer(r"<(\w+):p[ >].*?</\1:p>", xml, flags=re.S)]
    return [re.sub(r"<[^>]+>", "", p) for p in paras]


class DocxNumbersMatchCsvTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paras = _docx_paragraphs()
        cls.text = "\n".join(cls.paras)

    def test_docx_convergence_orders(self):
        for method, label in (("euler", "Explicit Euler"), ("verlet", "Velocity-Verlet-type"),
                              ("rk4", "RK4")):
            values = [f"{p:.2f}" for p in _orders()[method]]
            with self.subTest(method=method):
                self.assertIn("/".join(values), self.text)  # abstract: 1.27/1.03/1.11
                # table row: the label paragraph followed by the three cells
                # (the same label also appears as a section heading earlier)
                self.assertTrue(any(
                    self.paras[i] == label and self.paras[i + 1:i + 4] == values
                    for i in range(len(self.paras) - 3)), (label, values))

    def test_docx_stability_scan_ranges(self):
        scan = _rows("stability_scan.csv")
        euler = [float(r["zeta_fit_err"]) for r in scan if r["method"] == "euler"]
        verlet = [float(r["zeta_fit_err"]) for r in scan if r["method"] == "verlet"]
        self.assertIn(f"from {min(euler):.3f} (h = 0.16)", self.text)
        self.assertIn(f"to {max(euler):.3f} (h = 0.25)", self.text)
        self.assertIn(f"grow smoothly from {min(verlet):.4f} to {max(verlet):.4f}", self.text)

    def test_docx_energy_drift_figures(self):
        drift = {(r["method"], float(r["h"])): float(r["max_rel_E_err"])
                 for r in _rows("energy_drift.csv")}
        self.assertIn(f"{drift[('euler', 0.2)] / 1e8:.1f}×10⁸", self.text)
        self.assertIn(f"{drift[('rk4', 0.2)] * 1e4:.1f}×10⁻⁴", self.text)


if __name__ == "__main__":
    unittest.main()
