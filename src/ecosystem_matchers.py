"""Unified, audited multi-ecosystem version parser, constraint matcher, and openness classifier.

Supports five package ecosystems:
  - npm: delegates directly to the E12 node-semver 7.8.6 conformance-verified port (npm_semver.py).
  - pypi: delegates directly to official PyPA packaging.version and packaging.specifiers (PEP 440).
  - cargo: implements official Rust Cargo SemVer 2.0 requirements (bare versions are caret ^).
  - rubygems: implements official RubyGems Gem::Version and Gem::Requirement (~> pessimistic).
  - packagist: implements official Composer SemVer rules (^, Composer ~, || / |, wildcards, hyphens).

All resolvers exclude prerelease versions by default, matching standard package-manager policy.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version as Pep440Version

# Local self-contained copy of audited node-semver 7.8.6 port
from src import npm_semver  # type: ignore


# ---------------------------------------------------------------------------
# Package name normalization (official per-ecosystem rules)
# ---------------------------------------------------------------------------

_PYPI_NORM_RE = re.compile(r"[-_.]+")


def normalize_package_name(ecosystem: str, name: str) -> str:
    n = name.strip()
    if ecosystem == "pypi":
        return _PYPI_NORM_RE.sub("-", n).lower()
    if ecosystem == "cargo":
        return n.lower().replace("_", "-")
    if ecosystem == "packagist":
        return n.lower()
    if ecosystem in ("npm", "rubygems"):
        return n
    raise ValueError(f"Unsupported ecosystem: {ecosystem}")


# ---------------------------------------------------------------------------
# Generic parsed version representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedVersion:
    raw: str
    key: tuple
    is_prerelease: bool
    major: int
    minor: int
    patch: int


# ---------------------------------------------------------------------------
# 1. NPM (E12 audited port)
# ---------------------------------------------------------------------------

def parse_npm_version(raw: str) -> ParsedVersion | None:
    sv = npm_semver.parse_version(raw, loose=True)
    if sv is None:
        return None
    is_pre = bool(sv.prerelease)
    # npm_semver SemVer supports exact SemVer comparison via tuple or cmp
    key = (sv.major, sv.minor, sv.patch, 0 if is_pre else 1, tuple((0, p) if isinstance(p, int) else (1, str(p)) for p in sv.prerelease))
    return ParsedVersion(raw=raw, key=key, is_prerelease=is_pre, major=sv.major, minor=sv.minor, patch=sv.patch)


def compile_npm_matcher(req: str) -> tuple[Callable[[list[ParsedVersion]], list[bool]] | None, str]:
    s = (req or "").strip()
    if not s:
        s = "*"
    rng = npm_semver.parse_range(s, loose=True)
    if rng is None:
        return None, "unparseable"
    raw_cache: dict[str, object] = {}

    def match_all(versions: list[ParsedVersion]) -> list[bool]:
        out = []
        for pv in versions:
            if pv.is_prerelease:
                out.append(False)
                continue
            sv = raw_cache.get(pv.raw)
            if sv is None:
                sv = npm_semver.parse_version(pv.raw, loose=True)
                raw_cache[pv.raw] = sv
            out.append(bool(sv is not None and rng.test(sv)))
        return out

    return match_all, classify_npm_openness(s)


def classify_npm_openness(s: str) -> str:
    t = s.strip()
    if t in ("*", "", "x", "X", "latest"):
        return "wide"
    if "||" in t or " - " in t:
        return "wide"
    if re.fullmatch(r"=?\s*v?\d+\.\d+\.\d+", t):
        return "exact"
    if re.fullmatch(r"v?\d+\.\d+", t):
        return "patch"
    if t.startswith("~"):
        if re.fullmatch(r"~\s*v?\d+", t):
            return "minor"
        return "patch"
    if t.startswith("^"):
        m = re.match(r"^\^\s*v?(\d+)\.(\d+)(?:\.(\d+))?", t)
        if m and int(m.group(1)) == 0:
            if int(m.group(2)) > 0:
                return "patch"
            return "exact" if m.group(3) is not None else "patch"
        return "minor"
    if re.fullmatch(r"v?\d+\.\d+\.[xX*]", t):
        return "patch"
    if re.fullmatch(r"v?\d+\.[xX*](\.[xX*])?", t):
        return "minor"
    if re.search(r"[<>]", t):
        if "<" not in t:
            return "wide"
        # Check whether upper and lower share major/minor
        low = re.search(r">=\s*v?(\d+)\.(\d+)", t)
        high = re.search(r"<\s*=?\s*v?(\d+)\.(\d+)", t)
        if low and high:
            lm, ln = int(low.group(1)), int(low.group(2))
            hm, hn = int(high.group(1)), int(high.group(2))
            if hm == lm and hn == ln + 1:
                return "patch"
            if hm == lm + 1 and hn == 0:
                return "minor"
        return "wide"
    return "wide"


# ---------------------------------------------------------------------------
# 2. PYPI (Official PyPA PEP 440 packaging)
# ---------------------------------------------------------------------------

def parse_pypi_version(raw: str) -> ParsedVersion | None:
    try:
        v = Pep440Version(raw.strip())
    except InvalidVersion:
        return None
    rel = v.release
    major = rel[0] if len(rel) > 0 else 0
    minor = rel[1] if len(rel) > 1 else 0
    patch = rel[2] if len(rel) > 2 else 0
    return ParsedVersion(
        raw=raw,
        key=(v,),
        is_prerelease=bool(v.is_prerelease or v.is_devrelease),
        major=major,
        minor=minor,
        patch=patch,
    )


def compile_pypi_matcher(req: str) -> tuple[Callable[[list[ParsedVersion]], list[bool]] | None, str]:
    s = (req or "").strip()
    if s in ("", "*"):
        spec = SpecifierSet("", prereleases=False)
        return (lambda versions: [not pv.is_prerelease for pv in versions]), "wide"
    # Strip trailing environment markers if present (e.g. '; python_version >= "3.8"')
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    # Strip parentheses commonly wrapped around requirements in metadata
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s in ("", "*"):
        return (lambda versions: [not pv.is_prerelease for pv in versions]), "wide"
    try:
        spec = SpecifierSet(s, prereleases=False)
    except InvalidSpecifier:
        return None, "unparseable"

    def match_all(versions: list[ParsedVersion]) -> list[bool]:
        out = []
        for pv in versions:
            if pv.is_prerelease:
                out.append(False)
                continue
            out.append(spec.contains(pv.key[0], prereleases=False))
        return out

    return match_all, classify_pypi_openness(s)


def classify_pypi_openness(s: str) -> str:
    t = s.strip()
    if not t or t == "*":
        return "wide"
    parts = [p.strip() for p in t.split(",") if p.strip()]
    if len(parts) == 1:
        p = parts[0]
        if p.startswith(("==", "===")):
            val = p.lstrip("=").strip()
            if val.endswith(".*"):
                segs = val[:-2].split(".")
                return "patch" if len(segs) >= 2 else "minor"
            return "exact"
        if p.startswith("~="):
            segs = p[2:].strip().split(".")
            return "patch" if len(segs) >= 3 else "minor"
        return "wide"
    has_lower = any(p.startswith((">=", ">")) for p in parts)
    has_upper = any(p.startswith(("<=", "<")) for p in parts)
    if has_lower and has_upper:
        low_m = next((re.match(r">=?\s*(\d+)(?:\.(\d+))?", p) for p in parts if p.startswith(">")), None)
        high_m = next((re.match(r"<=?\s*(\d+)(?:\.(\d+))?", p) for p in parts if p.startswith("<")), None)
        if low_m and high_m:
            lm, ln = int(low_m.group(1)), int(low_m.group(2) or 0)
            hm, hn = int(high_m.group(1)), int(high_m.group(2) or 0)
            if hm == lm and hn == ln + 1:
                return "patch"
            if hm == lm + 1 and hn == 0:
                return "minor"
    if any(p.startswith("==") and not p.endswith(".*") for p in parts):
        return "exact"
    return "wide"


# ---------------------------------------------------------------------------
# 3. CARGO (Rust crates.io SemVer 2.0)
# ---------------------------------------------------------------------------

_CARGO_VER_RE = re.compile(
    r"^\s*v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?\s*$"
)
_CARGO_PARTIAL_RE = re.compile(
    r"^\s*(>=|<=|>|<|=|\^|~)?\s*v?(\d+|\*|[xX])(?:\.(\d+|\*|[xX]))?(?:\.(\d+|\*|[xX]))?(?:-([0-9A-Za-z.-]+))?(?:\+([0-9A-Za-z.-]+))?\s*$"
)


def parse_cargo_version(raw: str) -> ParsedVersion | None:
    m = _CARGO_VER_RE.match(raw)
    if not m:
        return None
    maj, mn, pat = int(m.group(1)), int(m.group(2)), int(m.group(3))
    pre = m.group(4)
    is_pre = pre is not None
    pre_key = ()
    if pre:
        pre_key = tuple((0, int(p)) if p.isdigit() else (1, p) for p in pre.split("."))
    key = (maj, mn, pat, 0 if is_pre else 1, pre_key)
    return ParsedVersion(raw=raw, key=key, is_prerelease=is_pre, major=maj, minor=mn, patch=pat)


def _cargo_pred(comp: str) -> Callable[[ParsedVersion], bool] | None:
    m = _CARGO_PARTIAL_RE.match(comp)
    if not m:
        return None
    op, s_maj, s_min, s_pat, pre, _ = m.groups()
    is_wild = lambda x: x in (None, "*", "x", "X")
    if is_wild(s_maj):
        return lambda pv: True
    maj = int(s_maj)
    mn = None if is_wild(s_min) else int(s_min)
    pat = None if is_wild(s_pat) else int(s_pat)

    # Bare version in Cargo is CARET unless wildcard was specified!
    if op is None:
        if s_min in ("*", "x", "X") or s_pat in ("*", "x", "X"):
            op = "wild"
        else:
            op = "^"

    lo_m, lo_n, lo_p = maj, mn or 0, pat or 0
    lo_t = (lo_m, lo_n, lo_p)
    req_is_pre = pre is not None
    req_pre_key: tuple = ()
    if req_is_pre:
        req_pre_key = tuple((0, int(p)) if p.isdigit() else (1, p) for p in pre.split("."))
    lo_full = (lo_m, lo_n, lo_p, 0 if req_is_pre else 1, req_pre_key)

    if op == "wild":
        if mn is None:
            hi_t = (maj + 1, 0, 0)
        else:
            hi_t = (maj, mn + 1, 0)
        return lambda pv, lo=lo_t, hi=hi_t: lo <= (pv.major, pv.minor, pv.patch) < hi

    if op == "^":
        if mn is None:
            hi_t = (maj + 1, 0, 0)
        elif pat is None:
            if maj > 0:
                hi_t = (maj + 1, 0, 0)
            elif mn > 0:
                hi_t = (0, mn + 1, 0)
            else:
                hi_t = (0, 1, 0)
        else:
            if maj > 0:
                hi_t = (maj + 1, 0, 0)
            elif mn > 0:
                hi_t = (0, mn + 1, 0)
            else:
                hi_t = (0, 0, pat + 1)
        if req_is_pre:
            hi_full = (hi_t[0], hi_t[1], hi_t[2], 0, ())
            return lambda pv, lo=lo_full, hi=hi_full: lo <= pv.key < hi
        return lambda pv, lo=lo_t, hi=hi_t: lo <= (pv.major, pv.minor, pv.patch) < hi

    if op == "~":
        if mn is None:
            hi_t = (maj + 1, 0, 0)
        else:
            hi_t = (maj, mn + 1, 0)
        if req_is_pre:
            hi_full = (hi_t[0], hi_t[1], hi_t[2], 0, ())
            return lambda pv, lo=lo_full, hi=hi_full: lo <= pv.key < hi
        return lambda pv, lo=lo_t, hi=hi_t: lo <= (pv.major, pv.minor, pv.patch) < hi

    if op == "=":
        if mn is None:
            hi_t = (maj + 1, 0, 0)
            return lambda pv, lo=lo_t, hi=hi_t: lo <= (pv.major, pv.minor, pv.patch) < hi
        if pat is None:
            hi_t = (maj, mn + 1, 0)
            return lambda pv, lo=lo_t, hi=hi_t: lo <= (pv.major, pv.minor, pv.patch) < hi
        if req_is_pre:
            return lambda pv, lo=lo_full: pv.key == lo
        return lambda pv, lo=lo_t: (pv.major, pv.minor, pv.patch) == lo

    if op == ">=":
        if req_is_pre:
            return lambda pv, lo=lo_full: pv.key >= lo
        return lambda pv, lo=lo_t: (pv.major, pv.minor, pv.patch) >= lo
    if op == "<=":
        if mn is None:
            hi_t = (maj + 1, 0, 0)
            return lambda pv, hi=hi_t: (pv.major, pv.minor, pv.patch) < hi
        if pat is None:
            hi_t = (maj, mn + 1, 0)
            return lambda pv, hi=hi_t: (pv.major, pv.minor, pv.patch) < hi
        if req_is_pre:
            return lambda pv, lo=lo_full: pv.key <= lo
        return lambda pv, lo=lo_t: (pv.major, pv.minor, pv.patch) <= lo
    if op == ">":
        if mn is None:
            lo2 = (maj + 1, 0, 0)
            return lambda pv, lo=lo2: (pv.major, pv.minor, pv.patch) >= lo
        if pat is None:
            lo2 = (maj, mn + 1, 0)
            return lambda pv, lo=lo2: (pv.major, pv.minor, pv.patch) >= lo
        if req_is_pre:
            return lambda pv, lo=lo_full: pv.key > lo
        return lambda pv, lo=lo_t: (pv.major, pv.minor, pv.patch) > lo
    if op == "<":
        if req_is_pre:
            return lambda pv, lo=lo_full: pv.key < lo
        return lambda pv, lo=lo_t: (pv.major, pv.minor, pv.patch) < lo
    return None


def compile_cargo_matcher(req: str) -> tuple[Callable[[list[ParsedVersion]], list[bool]] | None, str]:
    s = (req or "").strip()
    if s in ("", "*"):
        return (lambda versions: [not pv.is_prerelease for pv in versions]), "wide"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    preds = []
    for p in parts:
        pr = _cargo_pred(p)
        if pr is None:
            return None, "unparseable"
        preds.append(pr)

    def match_all(versions: list[ParsedVersion]) -> list[bool]:
        return [not pv.is_prerelease and all(pr(pv) for pr in preds) for pv in versions]

    return match_all, classify_cargo_openness(parts)


def classify_cargo_openness(parts: list[str]) -> str:
    if len(parts) == 1:
        p = parts[0].strip()
        if p in ("*", "x", "X"):
            return "wide"
        if p.startswith("="):
            segs = re.findall(r"\d+", p)
            if len(segs) >= 3:
                return "exact"
            return "patch" if len(segs) == 2 else "minor"
        if p.startswith("~"):
            segs = re.findall(r"\d+", p)
            return "patch" if len(segs) >= 2 else "minor"
        if p.startswith((">=", "<=", ">", "<")):
            return "wide"
        # Caret or bare version (implicit caret in Cargo!)
        segs = re.findall(r"\d+|\*|[xX]", p)
        if not segs or segs[0] in ("*", "x", "X"):
            return "wide"
        maj = int(segs[0])
        if len(segs) >= 2 and segs[1] in ("*", "x", "X"):
            return "minor"
        if len(segs) >= 3 and segs[2] in ("*", "x", "X"):
            return "patch"
        if maj == 0:
            mn = int(segs[1]) if len(segs) >= 2 else None
            pat = int(segs[2]) if len(segs) >= 3 else None
            if mn is None:
                return "minor"
            if mn > 0:
                return "patch"
            if pat is not None:
                return "exact"
            return "patch"
        return "minor"
    # Multi-comparator
    if any(p.startswith("=") for p in parts):
        return "exact"
    low = next((re.match(r">=?\s*v?(\d+)\.(\d+)", p) for p in parts if p.startswith(">")), None)
    high = next((re.match(r"<=?\s*v?(\d+)\.(\d+)", p) for p in parts if p.startswith("<")), None)
    if low and high:
        lm, ln = int(low.group(1)), int(low.group(2))
        hm, hn = int(high.group(1)), int(high.group(2))
        if hm == lm and hn == ln + 1:
            return "patch"
        if hm == lm + 1 and hn == 0:
            return "minor"
    return "wide"


# ---------------------------------------------------------------------------
# 4. RUBYGEMS (Gem::Version and Gem::Requirement)
# ---------------------------------------------------------------------------

_RUBY_VER_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9A-Za-z-]+)*)\s*$")
_RUBY_COMP_RE = re.compile(r"^\s*(~>|>=|<=|!=|>|<|=)?\s*([0-9]+(?:\.[0-9A-Za-z-]+)*)\s*$")


def _ruby_segs(s: str) -> tuple[tuple[int, ...], bool] | None:
    m = _RUBY_VER_RE.match(s)
    if not m:
        return None
    raw_segs = re.split(r"[.-]", m.group(1))
    is_pre = False
    nums: list[int] = []
    for seg in raw_segs:
        if seg.isdigit():
            nums.append(int(seg))
        else:
            is_pre = True
            break
    if not nums:
        return None
    return tuple(nums), is_pre


def _ruby_canonical(segs: tuple[int, ...]) -> tuple[int, ...]:
    lst = list(segs)
    while len(lst) > 1 and lst[-1] == 0:
        lst.pop()
    return tuple(lst)


def parse_rubygems_version(raw: str) -> ParsedVersion | None:
    res = _ruby_segs(raw)
    if res is None:
        return None
    segs, is_pre = res
    canon = _ruby_canonical(segs)
    maj = segs[0] if len(segs) > 0 else 0
    mn = segs[1] if len(segs) > 1 else 0
    pat = segs[2] if len(segs) > 2 else 0
    key = (canon, 0 if is_pre else 1)
    return ParsedVersion(raw=raw, key=key, is_prerelease=is_pre, major=maj, minor=mn, patch=pat)


def _ruby_cmp(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    la, lb = len(a), len(b)
    n = max(la, lb)
    for i in range(n):
        va = a[i] if i < la else 0
        vb = b[i] if i < lb else 0
        if va != vb:
            return -1 if va < vb else 1
    return 0


def compile_rubygems_matcher(req: str) -> tuple[Callable[[list[ParsedVersion]], list[bool]] | None, str]:
    s = (req or "").strip()
    if s in ("", "*", ">= 0", ">=0"):
        return (lambda versions: [not pv.is_prerelease for pv in versions]), "wide"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    preds = []
    for p in parts:
        m = _RUBY_COMP_RE.match(p)
        if not m:
            return None, "unparseable"
        op, ver_s = m.groups()
        op = op or "="
        parsed = _ruby_segs(ver_s)
        if parsed is None:
            return None, "unparseable"
        segs, _ = parsed
        if op == "~>":
            lo = segs
            if len(segs) == 1:
                hi = (segs[0] + 1,)
            else:
                hi = segs[:-2] + (segs[-2] + 1,)
            preds.append(lambda pv, lo=lo, hi=hi: _ruby_cmp(pv.key[0], lo) >= 0 and _ruby_cmp(pv.key[0], hi) < 0)
        elif op == "=":
            preds.append(lambda pv, t=segs: _ruby_cmp(pv.key[0], t) == 0)
        elif op == "!=":
            preds.append(lambda pv, t=segs: _ruby_cmp(pv.key[0], t) != 0)
        elif op == ">=":
            preds.append(lambda pv, t=segs: _ruby_cmp(pv.key[0], t) >= 0)
        elif op == "<=":
            preds.append(lambda pv, t=segs: _ruby_cmp(pv.key[0], t) <= 0)
        elif op == ">":
            preds.append(lambda pv, t=segs: _ruby_cmp(pv.key[0], t) > 0)
        elif op == "<":
            preds.append(lambda pv, t=segs: _ruby_cmp(pv.key[0], t) < 0)
        else:
            return None, "unparseable"

    def match_all(versions: list[ParsedVersion]) -> list[bool]:
        return [not pv.is_prerelease and all(pr(pv) for pr in preds) for pv in versions]

    return match_all, classify_rubygems_openness(parts)


def classify_rubygems_openness(parts: list[str]) -> str:
    if len(parts) == 1:
        p = parts[0].strip()
        m = _RUBY_COMP_RE.match(p)
        if not m:
            return "wide"
        op, ver_s = m.groups()
        op = op or "="
        segs = ver_s.split(".")
        if op == "=":
            return "exact"
        if op == "~>":
            return "patch" if len(segs) >= 3 else "minor"
        return "wide"
    if any(re.match(r"^\s*=\s*\d", p) for p in parts):
        return "exact"
    for p in parts:
        if p.strip().startswith("~>"):
            segs = p.strip()[2:].strip().split(".")
            return "patch" if len(segs) >= 3 else "minor"
    low = next((re.match(r">=?\s*(\d+)(?:\.(\d+))?", p) for p in parts if p.startswith(">")), None)
    high = next((re.match(r"<=?\s*(\d+)(?:\.(\d+))?", p) for p in parts if p.startswith("<")), None)
    if low and high:
        lm, ln = int(low.group(1)), int(low.group(2) or 0)
        hm, hn = int(high.group(1)), int(high.group(2) or 0)
        if hm == lm and hn == ln + 1:
            return "patch"
        if hm == lm + 1 and hn == 0:
            return "minor"
    return "wide"


# ---------------------------------------------------------------------------
# 5. PACKAGIST (Composer SemVer)
# ---------------------------------------------------------------------------

_COMP_VER_RE = re.compile(
    r"^\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?(?:[-._]?(alpha|beta|rc|dev|a|b|p|patch|pl)[0-9A-Za-z.-]*)?(?:\+.*)?\s*$",
    re.I,
)
_COMP_ATOM_RE = re.compile(
    r"^\s*(>=|<=|!=|==|>|<|=|\^|~)?\s*v?(\d+|\*|[xX])(?:\.(\d+|\*|[xX]))?(?:\.(\d+|\*|[xX]))?(?:\.(\d+|\*|[xX]))?(?:[-._][0-9A-Za-z.-]+)?\s*$"
)


def parse_packagist_version(raw: str) -> ParsedVersion | None:
    s = raw.strip()
    if s.lower().startswith("dev-") or s.lower().endswith("-dev"):
        return None
    m = _COMP_VER_RE.match(s)
    if not m:
        return None
    maj = int(m.group(1))
    mn = int(m.group(2) or 0)
    pat = int(m.group(3) or 0)
    p4 = int(m.group(4) or 0)
    stab = m.group(5)
    is_pre = stab is not None and stab.lower() not in ("p", "patch", "pl")
    key = (maj, mn, pat, p4, 0 if is_pre else 1)
    return ParsedVersion(raw=raw, key=key, is_prerelease=is_pre, major=maj, minor=mn, patch=pat)


def _comp_atom_pred(atom: str) -> Callable[[ParsedVersion], bool] | None:
    a = re.sub(r"@[A-Za-z]+", "", atom).strip()
    if not a or a in ("*", "x", "X", "*.*"):
        return lambda pv: True
    if "dev" in a.lower():
        return lambda pv: False
    m = _COMP_ATOM_RE.match(a)
    if not m:
        return None
    op, s1, s2, s3, s4 = m.groups()
    is_w = lambda x: x in (None, "*", "x", "X")
    if is_w(s1):
        return lambda pv: True
    v1 = int(s1)
    v2 = None if is_w(s2) else int(s2)
    v3 = None if is_w(s3) else int(s3)
    v4 = None if is_w(s4) else int(s4)
    lo = (v1, v2 or 0, v3 or 0, v4 or 0)

    if op is None:
        if s2 in ("*", "x", "X"):
            return lambda pv, lo=lo, hi=(v1 + 1, 0, 0, 0): lo <= pv.key[:4] < hi
        if s3 in ("*", "x", "X"):
            return lambda pv, lo=lo, hi=(v1, (v2 or 0) + 1, 0, 0): lo <= pv.key[:4] < hi
        if s4 in ("*", "x", "X"):
            return lambda pv, lo=lo, hi=(v1, v2 or 0, (v3 or 0) + 1, 0): lo <= pv.key[:4] < hi
        return lambda pv, lo=lo: pv.key[:4] == lo

    if op == "^":
        if v1 > 0:
            hi = (v1 + 1, 0, 0, 0)
        elif v2 is not None and v2 > 0:
            hi = (0, v2 + 1, 0, 0)
        elif v3 is not None:
            hi = (0, 0, v3 + 1, 0)
        elif v2 == 0:
            hi = (0, 1, 0, 0)
        else:
            hi = (1, 0, 0, 0)
        return lambda pv, lo=lo, hi=hi: lo <= pv.key[:4] < hi

    if op == "~":
        # Composer tilde: ~1.2 => >=1.2.0 <2.0.0; ~1.2.3 => >=1.2.3 <1.3.0; ~1 => >=1.0.0 <2.0.0
        if v3 is None:
            hi = (v1 + 1, 0, 0, 0)
        elif v4 is None:
            hi = (v1, (v2 or 0) + 1, 0, 0)
        else:
            hi = (v1, v2 or 0, v3 + 1, 0)
        return lambda pv, lo=lo, hi=hi: lo <= pv.key[:4] < hi

    if op in ("=", "=="):
        return lambda pv, lo=lo: pv.key[:4] == lo
    if op == "!=":
        return lambda pv, lo=lo: pv.key[:4] != lo
    if op == ">=":
        return lambda pv, lo=lo: pv.key[:4] >= lo
    if op == "<=":
        if v2 is None:
            return lambda pv, hi=(v1 + 1, 0, 0, 0): pv.key[:4] < hi
        if v3 is None:
            return lambda pv, hi=(v1, v2 + 1, 0, 0): pv.key[:4] < hi
        return lambda pv, lo=lo: pv.key[:4] <= lo
    if op == ">":
        return lambda pv, lo=lo: pv.key[:4] > lo
    if op == "<":
        return lambda pv, lo=lo: pv.key[:4] < lo
    return None


def compile_packagist_matcher(req: str) -> tuple[Callable[[list[ParsedVersion]], list[bool]] | None, str]:
    s = (req or "").strip()
    if s in ("", "*"):
        return (lambda versions: [not pv.is_prerelease for pv in versions]), "wide"
    # Strip inline alias ("... as 1.0.0")
    s = re.sub(r"\s+as\s+\S+", "", s, flags=re.I).strip()
    or_branches = [b.strip() for b in re.split(r"\|\||\|", s) if b.strip()]
    if not or_branches:
        return None, "unparseable"
    compiled_branches = []
    for br in or_branches:
        # Handle hyphen range "A - B"
        hm = re.match(r"^\s*(\S+)\s+-\s+(\S+)\s*$", br)
        if hm:
            p_lo = _comp_atom_pred(">=" + hm.group(1))
            p_hi = _comp_atom_pred("<=" + hm.group(2))
            if p_lo is None or p_hi is None:
                return None, "unparseable"
            compiled_branches.append([p_lo, p_hi])
            continue
        # Normalize space after operator (">= 1.0" -> ">=1.0")
        norm_br = re.sub(r"(>=|<=|!=|==|>|<|=|\^|~)\s+", r"\1", br)
        atoms = [a.strip() for a in re.split(r"[,\s]+", norm_br) if a.strip()]
        preds = []
        for a in atoms:
            pr = _comp_atom_pred(a)
            if pr is None:
                return None, "unparseable"
            preds.append(pr)
        compiled_branches.append(preds)

    def match_all(versions: list[ParsedVersion]) -> list[bool]:
        out = []
        for pv in versions:
            if pv.is_prerelease:
                out.append(False)
                continue
            ok = any(all(pr(pv) for pr in br) for br in compiled_branches)
            out.append(ok)
        return out

    return match_all, classify_packagist_openness(s, or_branches)


def classify_packagist_openness(s: str, or_branches: list[str]) -> str:
    if len(or_branches) > 1 or " - " in s:
        return "wide"
    t = re.sub(r"@[A-Za-z]+", "", or_branches[0]).strip()
    if t in ("*", "x", "X", "*.*"):
        return "wide"
    if re.fullmatch(r"(?:==|=)?\s*v?\d+\.\d+\.\d+(?:\.\d+)?", t):
        return "exact"
    if t.startswith("^"):
        m = re.match(r"^\^\s*v?(\d+)\.(\d+)", t)
        if m and int(m.group(1)) == 0:
            return "patch" if int(m.group(2)) > 0 else "exact"
        return "minor"
    if t.startswith("~"):
        segs = re.findall(r"\d+", t)
        return "patch" if len(segs) >= 3 else "minor"
    if re.fullmatch(r"v?\d+\.\d+\.[xX*]", t):
        return "patch"
    if re.fullmatch(r"v?\d+\.[xX*](\.[xX*])?", t):
        return "minor"
    if re.search(r"[<>]", t):
        if "<" not in t:
            return "wide"
        low = re.search(r">=\s*v?(\d+)(?:\.(\d+))?", t)
        high = re.search(r"<\s*=?\s*v?(\d+)(?:\.(\d+))?", t)
        if low and high:
            lm, ln = int(low.group(1)), int(low.group(2) or 0)
            hm, hn = int(high.group(1)), int(high.group(2) or 0)
            if hm == lm and hn == ln + 1:
                return "patch"
            if hm == lm + 1 and hn == 0:
                return "minor"
    return "wide"


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

PARSERS = {
    "npm": parse_npm_version,
    "pypi": parse_pypi_version,
    "cargo": parse_cargo_version,
    "rubygems": parse_rubygems_version,
    "packagist": parse_packagist_version,
}

COMPILERS = {
    "npm": compile_npm_matcher,
    "pypi": compile_pypi_matcher,
    "cargo": compile_cargo_matcher,
    "rubygems": compile_rubygems_matcher,
    "packagist": compile_packagist_matcher,
}


def parse_version(ecosystem: str, raw: str) -> ParsedVersion | None:
    return PARSERS[ecosystem](raw)


def compile_constraint(ecosystem: str, req: str) -> tuple[Callable[[list[ParsedVersion]], list[bool]] | None, str]:
    return COMPILERS[ecosystem](req)
