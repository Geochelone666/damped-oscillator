import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import make_tables  # noqa: E402


class MakeTablesTests(unittest.TestCase):
    def test_rows_match_paper(self):
        self.assertEqual(make_tables.latex_rows(), [
            "Explicit Euler & 1.27 & 1.03 & 1.11 \\\\",
            "Velocity-Verlet-type & 2.02 & 2.10 & 2.20 \\\\",
            "RK4 & 4.00 & 4.07 & 4.19 \\\\"])

    def test_exact_power_law_gives_its_order(self):
        # hand case: E_inf = 3 * h**2 for every method and zeta must give order exactly 2.00
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["method", "zeta", "h", "E_inf"])
                for m, _ in make_tables.METHODS:
                    for z in make_tables.ZETAS:
                        for h in (0.2, 0.1, 0.05, 0.025, 0.0125):
                            w.writerow([m, z, h, 3 * h ** 2])
            for row in make_tables.latex_rows(path):
                self.assertTrue(row.endswith("2.00 & 2.00 & 2.00 \\\\"), row)

    def _write(self, d, mutate):
        rows = [{"method": m, "zeta": z, "h": h, "E_inf": 3 * h ** 2}
                for m, _ in make_tables.METHODS for z in make_tables.ZETAS
                for h in (0.2, 0.1, 0.05, 0.025, 0.0125)]
        mutate(rows)
        path = Path(d) / "m.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "zeta", "h", "E_inf"]); w.writeheader(); w.writerows(rows)
        return path

    def test_bad_input_raises_instead_of_printing_nan(self):
        cases = {"zero": lambda r: r[4].update(E_inf=0.0), "nan": lambda r: r[4].update(E_inf="nan"),
                 "missing point": lambda r: r.pop(0),
                 "duplicate h": lambda r: r[1].update(h=r[0]["h"])}
        for name, mutate in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    make_tables.latex_rows(self._write(d, mutate))

    def test_row_order_does_not_matter(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(make_tables.latex_rows(self._write(d, lambda r: r.reverse())),
                             make_tables.latex_rows(self._write(d, lambda r: None)))

    def test_check_flag_passes_on_repo(self):
        r = subprocess.run([sys.executable, str(ROOT / "experiments" / "make_tables.py"), "--check"], cwd="/")
        self.assertEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main()
