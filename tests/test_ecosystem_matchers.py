"""Unit, boundary, and mutation conformance test suite for all five ecosystem resolvers and OSV range parsing."""

from __future__ import annotations

import unittest
from packaging.specifiers import SpecifierSet
from packaging.version import Version as Pep440

import sys
from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from src import ecosystem_matchers as er
from experiments import run_census_replay as runner


class TestEcoResolvers(unittest.TestCase):
    def _admitted(self, eco: str, req: str, vers: list[str]) -> tuple[list[str], str]:
        pvs = [er.parse_version(eco, v) for v in vers]
        pvs = [p for p in pvs if p is not None]
        fn, fam = er.compile_constraint(eco, req)
        self.assertIsNotNone(fn, f"Failed to parse {eco}: {req}")
        mask = fn(pvs)
        return [p.raw for p, ok in zip(pvs, mask) if ok], fam

    def test_name_normalization(self):
        self.assertEqual(er.normalize_package_name("pypi", "Scikit_Learn"), "scikit-learn")
        self.assertEqual(er.normalize_package_name("pypi", "zope.interface"), "zope-interface")
        self.assertEqual(er.normalize_package_name("pypi", "my__Pkg..Name"), "my-pkg-name")
        self.assertEqual(er.normalize_package_name("cargo", "serde_json"), "serde-json")
        self.assertEqual(er.normalize_package_name("cargo", "Tokio_Util"), "tokio-util")
        self.assertEqual(er.normalize_package_name("packagist", "Symfony/Http-Kernel"), "symfony/http-kernel")
        self.assertEqual(er.normalize_package_name("npm", "@scope/Pkg"), "@scope/Pkg")
        self.assertEqual(er.normalize_package_name("rubygems", "activesupport"), "activesupport")

    def test_npm_semantics_and_boundaries(self):
        vers = ["0.0.2", "0.0.3", "0.1.0", "0.1.5", "0.2.0", "1.2.3", "1.2.9", "1.5.0", "2.0.0", "1.6.0-alpha.1"]
        adm, fam = self._admitted("npm", "^1.2.3", vers)
        self.assertEqual(adm, ["1.2.3", "1.2.9", "1.5.0"])
        self.assertEqual(fam, "minor")
        adm0, fam0 = self._admitted("npm", "^0.1.0", vers)
        self.assertEqual(adm0, ["0.1.0", "0.1.5"])
        self.assertEqual(fam0, "patch")
        adm00, fam00 = self._admitted("npm", "^0.0.2", vers)
        self.assertEqual(adm00, ["0.0.2"])
        self.assertEqual(fam00, "exact")
        adm2, fam2 = self._admitted("npm", "1.2.3", vers)
        self.assertEqual(adm2, ["1.2.3"])
        self.assertEqual(fam2, "exact")
        adm3, fam3 = self._admitted("npm", "~1.2.3", vers)
        self.assertEqual(adm3, ["1.2.3", "1.2.9"])
        self.assertEqual(fam3, "patch")
        adm_or, fam_or = self._admitted("npm", "^1.2.3 || ^2.0.0", vers)
        self.assertEqual(adm_or, ["1.2.3", "1.2.9", "1.5.0", "2.0.0"])
        self.assertEqual(fam_or, "wide")

    def test_pypi_pep440_differential_extended(self):
        vers = [
            "0.9.0", "1.0.0", "1.1.0", "1.2.0", "1.2.5", "1.2.5.post1",
            "1.3.0a1", "1.3.0b2", "1.3.0.dev1", "1.3.0", "1.9.9", "2.0.0", "2.1.0",
        ]
        reqs = [
            ">=1.2,<2.0", "==1.2.5", "==1.2.*", "==1.*", "~=1.2.0", "~=1.2",
            ">=1.0", "!=1.2.5,>=1.2", ">1.0.0,<=1.3.0", "(>=1.2,<1.3)",
            ">=1.2 ; python_version >= '3.8'", "*",
        ]
        for r in reqs:
            adm, _ = self._admitted("pypi", r, vers)
            clean_r = r.split(";")[0].strip()
            if clean_r.startswith("(") and clean_r.endswith(")"):
                clean_r = clean_r[1:-1].strip()
            if clean_r == "*":
                clean_r = ""
            spec = SpecifierSet(clean_r, prereleases=False)
            ref = [
                v for v in vers
                if not (Pep440(v).is_prerelease or Pep440(v).is_devrelease)
                and spec.contains(Pep440(v), prereleases=False)
            ]
            self.assertEqual(adm, ref, f"PyPI differential mismatch on {r}")
        _, fam_eq = self._admitted("pypi", "==1.2.5", vers)
        _, fam_tilde3 = self._admitted("pypi", "~=1.2.0", vers)
        _, fam_tilde2 = self._admitted("pypi", "~=1.2", vers)
        _, fam_wc2 = self._admitted("pypi", "==1.2.*", vers)
        _, fam_wc1 = self._admitted("pypi", "==1.*", vers)
        _, fam_open = self._admitted("pypi", ">=1.2", vers)
        self.assertEqual(
            (fam_eq, fam_tilde3, fam_tilde2, fam_wc2, fam_wc1, fam_open),
            ("exact", "patch", "minor", "patch", "minor", "wide"),
        )

    def test_cargo_semver_boundaries_and_mutations(self):
        vers = [
            "0.0.2", "0.0.3", "0.0.4", "0.1.0", "0.2.3", "0.2.9", "0.3.0",
            "1.0.0", "1.2.3", "1.2.9", "1.5.0", "2.0.0", "1.5.1-beta.1",
        ]
        # Bare 1.2.3 is CARET in Cargo!
        adm, fam = self._admitted("cargo", "1.2.3", vers)
        self.assertEqual(adm, ["1.2.3", "1.2.9", "1.5.0"])
        self.assertEqual(fam, "minor")
        # Bare 0.2 is >=0.2.0, <0.3.0
        adm02, fam02 = self._admitted("cargo", "0.2", vers)
        self.assertEqual(adm02, ["0.2.3", "0.2.9"])
        self.assertEqual(fam02, "patch")
        # Bare 0.0.3 is exact 0.0.3
        adm00, fam00 = self._admitted("cargo", "0.0.3", vers)
        self.assertEqual(adm00, ["0.0.3"])
        self.assertEqual(fam00, "exact")
        # Bare 0.0 is >=0.0.0, <0.1.0
        adm00_2, fam00_2 = self._admitted("cargo", "0.0", vers)
        self.assertEqual(adm00_2, ["0.0.2", "0.0.3", "0.0.4"])
        self.assertEqual(fam00_2, "patch")
        # Explicit =1.2.3 is exact; =1.2 is >=1.2.0 <1.3.0
        admeq, fameq = self._admitted("cargo", "=1.2.3", vers)
        self.assertEqual(admeq, ["1.2.3"])
        self.assertEqual(fameq, "exact")
        admeq2, fameq2 = self._admitted("cargo", "=1.2", vers)
        self.assertEqual(admeq2, ["1.2.3", "1.2.9"])
        self.assertEqual(fameq2, "patch")
        # Tilde ~1.2.3 is patch; ~1 is minor
        admt, famt = self._admitted("cargo", "~1.2.3", vers)
        self.assertEqual(admt, ["1.2.3", "1.2.9"])
        self.assertEqual(famt, "patch")
        admt1, famt1 = self._admitted("cargo", "~1", vers)
        self.assertEqual(admt1, ["1.0.0", "1.2.3", "1.2.9", "1.5.0"])
        self.assertEqual(famt1, "minor")
        # Wildcards and compound comma constraints
        admw, famw = self._admitted("cargo", "1.2.*", vers)
        self.assertEqual(admw, ["1.2.3", "1.2.9"])
        self.assertEqual(famw, "patch")
        admc, famc = self._admitted("cargo", ">=1.2.3, <2.0.0", vers)
        self.assertEqual(admc, ["1.2.3", "1.2.9", "1.5.0"])
        self.assertEqual(famc, "minor")

    def test_rubygems_pessimistic_and_canonical_zeros(self):
        vers = ["0.9.0", "1.1.9", "1.2", "1.2.0", "1.2.3", "1.2.9", "1.3.0", "1.9.9", "2.0.0", "1.2.4.pre"]
        adm_m, fam_m = self._admitted("rubygems", "~> 1.2", vers)
        self.assertEqual(adm_m, ["1.2", "1.2.0", "1.2.3", "1.2.9", "1.3.0", "1.9.9"])
        self.assertEqual(fam_m, "minor")
        adm_p, fam_p = self._admitted("rubygems", "~> 1.2.3", vers)
        self.assertEqual(adm_p, ["1.2.3", "1.2.9"])
        self.assertEqual(fam_p, "patch")
        adm_1, fam_1 = self._admitted("rubygems", "~> 1", vers)
        self.assertEqual(adm_1, ["1.1.9", "1.2", "1.2.0", "1.2.3", "1.2.9", "1.3.0", "1.9.9"])
        self.assertEqual(fam_1, "minor")
        adm_e, fam_e = self._admitted("rubygems", "= 1.2.3", vers)
        self.assertEqual(adm_e, ["1.2.3"])
        self.assertEqual(fam_e, "exact")
        # Canonical trailing zero equivalence: = 1.2 matches 1.2 and 1.2.0
        adm_z, _ = self._admitted("rubygems", "= 1.2", vers)
        self.assertEqual(adm_z, ["1.2", "1.2.0"])
        adm_c, _ = self._admitted("rubygems", "~> 1.2, >= 1.2.3, != 1.3.0", vers)
        self.assertEqual(adm_c, ["1.2.3", "1.2.9", "1.9.9"])

    def test_packagist_composer_boundaries_and_mutations(self):
        vers = [
            "0.0.2", "0.0.3", "0.2.1", "0.2.9", "0.3.0",
            "1.2.0", "1.2.3", "1.2.9", "1.5.0", "2.0.0", "2.1.0", "3.0.0", "1.6.0-RC1",
        ]
        # Composer ~1.2 means >=1.2.0 <2.0.0 (minor-floating, NOT patch!)
        adm_t2, fam_t2 = self._admitted("packagist", "~1.2", vers)
        self.assertEqual(adm_t2, ["1.2.0", "1.2.3", "1.2.9", "1.5.0"])
        self.assertEqual(fam_t2, "minor")
        # Composer ~1.2.3 means >=1.2.3 <1.3.0 (patch!)
        adm_t3, fam_t3 = self._admitted("packagist", "~1.2.3", vers)
        self.assertEqual(adm_t3, ["1.2.3", "1.2.9"])
        self.assertEqual(fam_t3, "patch")
        # Caret ^1.2.3, ^0.2.1, ^0.0.2
        adm_c, fam_c = self._admitted("packagist", "^1.2.3", vers)
        self.assertEqual(adm_c, ["1.2.3", "1.2.9", "1.5.0"])
        self.assertEqual(fam_c, "minor")
        adm_c0, fam_c0 = self._admitted("packagist", "^0.2.1", vers)
        self.assertEqual(adm_c0, ["0.2.1", "0.2.9"])
        self.assertEqual(fam_c0, "patch")
        adm_c00, fam_c00 = self._admitted("packagist", "^0.0.2", vers)
        self.assertEqual(adm_c00, ["0.0.2"])
        self.assertEqual(fam_c00, "exact")
        # OR branches ^1.2 || ^2.0 and single-pipe ^1.2|^2.0
        adm_or, fam_or = self._admitted("packagist", "^1.2 || ^2.0", vers)
        self.assertEqual(adm_or, ["1.2.0", "1.2.3", "1.2.9", "1.5.0", "2.0.0", "2.1.0"])
        self.assertEqual(fam_or, "wide")
        adm_or1, _ = self._admitted("packagist", "^1.2|^2.0", vers)
        self.assertEqual(adm_or1, adm_or)
        # Hyphen range and stability flag stripping
        adm_h, fam_h = self._admitted("packagist", "1.2.0 - 2.0.0", vers)
        self.assertEqual(adm_h, ["1.2.0", "1.2.3", "1.2.9", "1.5.0", "2.0.0"])
        self.assertEqual(fam_h, "wide")
        adm_st, _ = self._admitted("packagist", "^1.2@stable", vers)
        self.assertEqual(adm_st, ["1.2.0", "1.2.3", "1.2.9", "1.5.0"])
        # Exact bare 1.2.3
        adm_eq, fam_eq = self._admitted("packagist", "1.2.3", vers)
        self.assertEqual(adm_eq, ["1.2.3"])
        self.assertEqual(fam_eq, "exact")
        # Wildcard 1.2.*
        adm_wc, fam_wc = self._admitted("packagist", "1.2.*", vers)
        self.assertEqual(adm_wc, ["1.2.0", "1.2.3", "1.2.9"])
        self.assertEqual(fam_wc, "patch")

    def test_osv_multi_segment_vuln_mask(self):
        vers = ["1.0.0", "1.1.0", "1.2.0", "1.3.0", "2.0.0", "2.0.5", "2.1.0", "2.2.0"]
        pvs = [er.parse_version("npm", v) for v in vers]
        ranges = [[
            {"introduced": "1.0.0"}, {"fixed": "1.2.0"},
            {"introduced": "2.0.0"}, {"fixed": "2.1.0"},
        ]]
        vmask = runner.vuln_mask_for_ranges("npm", pvs, ranges)
        vuln = [v for v, m in zip(vers, vmask) if m]
        self.assertEqual(vuln, ["1.0.0", "1.1.0", "2.0.0", "2.0.5"])

    def test_mutation_killer_assertions(self):
        """Ensure mutation of key ecosystem semantics (Cargo bare caret, Composer tilde, RubyGems ~>) breaks assertions."""
        vers = ["1.2.0", "1.2.3", "1.2.9", "1.5.0", "2.0.0"]
        # Cargo bare 1.2.3 MUST differ from exact 1.2.3 (unlike npm/Packagist!)
        cargo_bare, _ = self._admitted("cargo", "1.2.3", vers)
        npm_bare, _ = self._admitted("npm", "1.2.3", vers)
        pack_bare, _ = self._admitted("packagist", "1.2.3", vers)
        self.assertNotEqual(cargo_bare, npm_bare)
        self.assertEqual(npm_bare, pack_bare)
        # Composer ~1.2 MUST include 1.5.0 (minor), while npm ~1.2.0 excludes 1.5.0 (patch)!
        pack_t2, _ = self._admitted("packagist", "~1.2", vers)
        npm_t3, _ = self._admitted("npm", "~1.2.0", vers)
        self.assertIn("1.5.0", pack_t2)
        self.assertNotIn("1.5.0", npm_t3)

    def test_canonical_keys_openness_and_cargo_prerelease_regression(self):
        """Regression tests for canonical version keys, openness classification, and Cargo prerelease bounds."""
        # Canonical key: Packagist 'v1.2.3' and '1.2.3' MUST share identical canonical key
        pv_v = er.parse_version("packagist", "v1.2.3")
        pv_plain = er.parse_version("packagist", "1.2.3")
        self.assertIsNotNone(pv_v)
        self.assertIsNotNone(pv_plain)
        self.assertEqual(pv_v.key, pv_plain.key)
        # Canonical key: Packagist last_affected with 'v'-prefixed version MUST match vulnerable
        pvs_pack = [er.parse_version("packagist", v) for v in ("v1.2.0", "v1.2.2", "v1.2.3")]
        vmask_p = runner.vuln_mask_for_ranges("packagist", pvs_pack, [[{"introduced": "1.2.0"}, {"last_affected": "1.2.2"}]])
        self.assertEqual([p.raw for p, m in zip(pvs_pack, vmask_p) if m], ["v1.2.0", "v1.2.2"])
        # Canonical key: RubyGems '1.2' and '1.2.0' MUST share identical canonical key
        rv_short = er.parse_version("rubygems", "1.2")
        rv_long = er.parse_version("rubygems", "1.2.0")
        self.assertEqual(rv_short.key, rv_long.key)
        # Openness/prerelease: Packagist bounded ranges classify as minor/patch rather than wide
        _, fam_pm = er.compile_constraint("packagist", ">=1.0 <2.0")
        _, fam_pp = er.compile_constraint("packagist", ">=1.2 <1.3")
        self.assertEqual(fam_pm, "minor")
        self.assertEqual(fam_pp, "patch")
        # Openness/prerelease: npm openness edge cases (^0.0 -> patch, ~1 -> minor, bare 1.2 -> patch)
        self.assertEqual(er.classify_npm_openness("^0.0"), "patch")
        self.assertEqual(er.classify_npm_openness("^0.0.2"), "exact")
        self.assertEqual(er.classify_npm_openness("~1"), "minor")
        self.assertEqual(er.classify_npm_openness("1.2"), "patch")
        # Openness/prerelease: Cargo prerelease constraint =1.0.0-alpha.1 / <=1.0.0-alpha.1 MUST NOT admit stable 1.0.0
        adm_eq_pre, _ = self._admitted("cargo", "=1.0.0-alpha.1", ["0.9.0", "1.0.0", "1.1.0"])
        adm_le_pre, _ = self._admitted("cargo", "<=1.0.0-alpha.1", ["0.9.0", "1.0.0", "1.1.0"])
        self.assertEqual(adm_eq_pre, [])
        self.assertEqual(adm_le_pre, ["0.9.0"])


if __name__ == "__main__":
    unittest.main()
