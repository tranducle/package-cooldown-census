"""Study resolver for D2 (E12): a line-by-line port of npm node-semver 7.8.6 range matching.

Scope: `satisfies(version, range)` with options {loose, includePrerelease: false}: loose false is the E02 reference
setting, loose true is what npm-pick-manifest uses during npm installation; plus helpers to list admitted versions and pick the highest. Ported from node-semver 7.8.6:
internal/re.js (safeRe sources), internal/constants.js, internal/identifiers.js,
internal/trim-comparator-whitespace.js, classes/semver.js, classes/comparator.js, classes/range.js.

JavaScript to Python differences handled explicitly:
- `\\s` in JS matches exactly the WHITESPACE_CHARACTERS of constants.js; Python's `\\s` matches more, so an explicit
  class is used everywhere, and String.prototype.trim is replaced by strip() over the same characters.
- JS `$` without the m flag matches only at the end of input; Python `$` also matches before a final newline, so
  `\\Z` is used.
- JS `\\d` is ASCII only; `[0-9]` is used.
- Unmatched groups are undefined in JS callbacks and empty in `$n` replacements; Python gives None and "".
The E12 conformance run (run_e12_conformance.py) checks this port against node-semver 7.8.6 itself.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

MAX_LENGTH = 256
MAX_SAFE_INTEGER = 9007199254740991
MAX_SAFE_BUILD_LENGTH = MAX_LENGTH - 6
WS_LIST = ["\t", "\n", "\v", "\f", "\r", " ", " ", " ", " ", " ", " ", " ", " ",
           " ", " ", " ", " ", " ", " ", " ", " ", " ", " ",
           "　", "﻿"]
WS_CHARS = "".join(WS_LIST)
WS_SET = frozenset(WS_LIST)
_WS_IN_CLASS = "".join(re.escape(c) if c in "\\]^-" else c for c in WS_LIST)
WS = "[" + _WS_IN_CLASS + "]"
LDN = "[a-zA-Z0-9-]"


def js_trim(s: str) -> str:
    return s.strip(WS_CHARS)


# ---------------------------------------------------------------- internal/re.js
_src: dict[str, str] = {}
_safe: dict[str, str] = {}


def _make_safe(value: str) -> str:
    for token, mx in (("\\s", 1), ("\\d", MAX_LENGTH), (LDN, MAX_SAFE_BUILD_LENGTH)):
        value = value.replace(token + "*", token + "{0,%d}" % mx).replace(token + "+", token + "{1,%d}" % mx)
    return value


def _to_py(js: str) -> str:
    out = js.replace("[v=\\s]", "[v=" + _WS_IN_CLASS + "]")
    out = out.replace("\\s", WS).replace("\\d", "[0-9]")
    return out.replace("$", r"\Z")


def _tok(name: str, value: str) -> None:
    _src[name] = value
    _safe[name] = _make_safe(value)


_tok("NUMERICIDENTIFIER", "0|[1-9]\\d*")
_tok("NUMERICIDENTIFIERLOOSE", "\\d+")
_tok("NONNUMERICIDENTIFIER", f"\\d*[a-zA-Z-]{LDN}*")
_tok("MAINVERSION", f"({_src['NUMERICIDENTIFIER']})\\.({_src['NUMERICIDENTIFIER']})\\.({_src['NUMERICIDENTIFIER']})")
_tok("PRERELEASEIDENTIFIER", f"(?:{_src['NONNUMERICIDENTIFIER']}|{_src['NUMERICIDENTIFIER']})")
_tok("MAINVERSIONLOOSE", f"({_src['NUMERICIDENTIFIERLOOSE']})\\.({_src['NUMERICIDENTIFIERLOOSE']})\\.({_src['NUMERICIDENTIFIERLOOSE']})")
_tok("PRERELEASEIDENTIFIERLOOSE", f"(?:{_src['NONNUMERICIDENTIFIER']}|{_src['NUMERICIDENTIFIERLOOSE']})")
_tok("PRERELEASE", f"(?:-({_src['PRERELEASEIDENTIFIER']}(?:\\.{_src['PRERELEASEIDENTIFIER']})*))")
_tok("PRERELEASELOOSE", f"(?:-?({_src['PRERELEASEIDENTIFIERLOOSE']}(?:\\.{_src['PRERELEASEIDENTIFIERLOOSE']})*))")
_tok("BUILDIDENTIFIER", f"{LDN}+")
_tok("BUILD", f"(?:\\+({_src['BUILDIDENTIFIER']}(?:\\.{_src['BUILDIDENTIFIER']})*))")
_tok("FULLPLAIN", f"v?{_src['MAINVERSION']}{_src['PRERELEASE']}?{_src['BUILD']}?")
_tok("FULL", f"^{_src['FULLPLAIN']}$")
_tok("LOOSEPLAIN", f"[v=\\s]*{_src['MAINVERSIONLOOSE']}{_src['PRERELEASELOOSE']}?{_src['BUILD']}?")
_tok("LOOSE", f"^{_src['LOOSEPLAIN']}$")
_tok("GTLT", "((?:<|>)?=?)")
_tok("XRANGEIDENTIFIERLOOSE", f"{_src['NUMERICIDENTIFIERLOOSE']}|x|X|\\*")
_tok("XRANGEIDENTIFIER", f"{_src['NUMERICIDENTIFIER']}|x|X|\\*")
_tok("XRANGEPLAIN", f"[v=\\s]*({_src['XRANGEIDENTIFIER']})(?:\\.({_src['XRANGEIDENTIFIER']})"
                    f"(?:\\.({_src['XRANGEIDENTIFIER']})(?:{_src['PRERELEASE']})?{_src['BUILD']}?)?)?")
_tok("XRANGEPLAINLOOSE", f"[v=\\s]*({_src['XRANGEIDENTIFIERLOOSE']})(?:\\.({_src['XRANGEIDENTIFIERLOOSE']})"
                         f"(?:\\.({_src['XRANGEIDENTIFIERLOOSE']})(?:{_src['PRERELEASELOOSE']})?{_src['BUILD']}?)?)?")
_tok("XRANGE", f"^{_src['GTLT']}\\s*{_src['XRANGEPLAIN']}$")
_tok("XRANGELOOSE", f"^{_src['GTLT']}\\s*{_src['XRANGEPLAINLOOSE']}$")
_tok("LONETILDE", "(?:~>?)")
_tok("TILDETRIM", f"(\\s*){_src['LONETILDE']}\\s+")
_tok("TILDE", f"^{_src['LONETILDE']}{_src['XRANGEPLAIN']}$")
_tok("TILDELOOSE", f"^{_src['LONETILDE']}{_src['XRANGEPLAINLOOSE']}$")
_tok("LONECARET", "(?:\\^)")
_tok("CARETTRIM", f"(\\s*){_src['LONECARET']}\\s+")
_tok("CARET", f"^{_src['LONECARET']}{_src['XRANGEPLAIN']}$")
_tok("CARETLOOSE", f"^{_src['LONECARET']}{_src['XRANGEPLAINLOOSE']}$")
_tok("COMPARATORLOOSE", f"^{_src['GTLT']}\\s*({_src['LOOSEPLAIN']})$|^$")
_tok("COMPARATOR", f"^{_src['GTLT']}\\s*({_src['FULLPLAIN']})$|^$")
_tok("HYPHENRANGE", f"^\\s*({_src['XRANGEPLAIN']})\\s+-\\s+({_src['XRANGEPLAIN']})\\s*$")
_tok("HYPHENRANGELOOSE", f"^\\s*({_src['XRANGEPLAINLOOSE']})\\s+-\\s+({_src['XRANGEPLAINLOOSE']})\\s*$")
_tok("STAR", "(<|>)?=?\\s*\\*")
_tok("GTE0", "^\\s*>=\\s*0\\.0\\.0\\s*$")

RE = {k: re.compile(_to_py(v)) for k, v in _safe.items()}
BUILDSTRIPRE = re.compile(_to_py(_src["BUILD"]))      # range.js uses the unbounded src form, global
SPLIT_WS = re.compile(WS + "+")
NUMERIC = re.compile(r"^[0-9]+\Z")


# ---------------------------------------------------------------- internal/identifiers.js
def compare_identifiers(a, b) -> int:
    if isinstance(a, int) and isinstance(b, int):
        return 0 if a == b else (-1 if a < b else 1)
    anum = bool(NUMERIC.match(str(a)))
    bnum = bool(NUMERIC.match(str(b)))
    if anum and bnum:
        a, b = int(str(a)), int(str(b))
    if a == b:
        return 0
    if anum and not bnum:
        return -1
    if bnum and not anum:
        return 1
    return -1 if str(a) < str(b) else 1


# ---------------------------------------------------------------- classes/semver.js
class InvalidVersion(ValueError):
    pass


class InvalidRange(ValueError):
    pass


class SemVer:
    __slots__ = ("major", "minor", "patch", "prerelease", "version")

    def __init__(self, version: str, loose: bool = False):
        if not isinstance(version, str):
            raise InvalidVersion("not a string")
        if len(version) > MAX_LENGTH:
            raise InvalidVersion("too long")
        m = RE["LOOSE" if loose else "FULL"].match(js_trim(version))
        if not m:
            raise InvalidVersion(version)
        self.major, self.minor, self.patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
        for part in (self.major, self.minor, self.patch):
            if part > MAX_SAFE_INTEGER:
                raise InvalidVersion("component too large")
        pre = []
        if m.group(4):
            for ident in m.group(4).split("."):
                if NUMERIC.match(ident) and int(ident) < MAX_SAFE_INTEGER:
                    pre.append(int(ident))
                else:
                    pre.append(ident)
        self.prerelease = tuple(pre)
        self.version = f"{self.major}.{self.minor}.{self.patch}" + (
            "-" + ".".join(str(x) for x in self.prerelease) if self.prerelease else "")

    def compare_main(self, o: "SemVer") -> int:
        a, b = (self.major, self.minor, self.patch), (o.major, o.minor, o.patch)
        return 0 if a == b else (-1 if a < b else 1)

    def compare_pre(self, o: "SemVer") -> int:
        if self.prerelease and not o.prerelease:
            return -1
        if not self.prerelease and o.prerelease:
            return 1
        if not self.prerelease and not o.prerelease:
            return 0
        i = 0
        while True:
            a = self.prerelease[i] if i < len(self.prerelease) else None
            b = o.prerelease[i] if i < len(o.prerelease) else None
            if a is None and b is None:
                return 0
            if b is None:
                return 1
            if a is None:
                return -1
            if a == b and type(a) is type(b):
                i += 1
                continue
            return compare_identifiers(a, b)

    def compare(self, o: "SemVer") -> int:
        if o.version == self.version:
            return 0
        return self.compare_main(o) or self.compare_pre(o)


@lru_cache(maxsize=1 << 20)
def parse_version(text: str, loose: bool = False) -> SemVer | None:
    try:
        return SemVer(text, loose)
    except (InvalidVersion, ValueError):
        return None


# ---------------------------------------------------------------- classes/comparator.js
ANY = object()


class Comparator:
    __slots__ = ("operator", "semver", "value")

    def __init__(self, comp: str, loose: bool = False):
        comp = " ".join(SPLIT_WS.split(js_trim(comp)))
        m = RE["COMPARATORLOOSE" if loose else "COMPARATOR"].match(comp)
        if not m:
            raise InvalidRange("Invalid comparator: " + comp)
        op = m.group(1) if m.group(1) is not None else ""
        self.operator = "" if op == "=" else op
        if not m.group(2):
            self.semver = ANY
            self.value = ""
        else:
            try:
                self.semver = SemVer(m.group(2), loose)
            except InvalidVersion as exc:
                raise InvalidRange(str(exc)) from exc
            self.value = self.operator + self.semver.version

    def test(self, v: SemVer) -> bool:
        if self.semver is ANY:
            return True
        c = v.compare(self.semver)
        op = self.operator
        if op == "":
            return c == 0
        if op == ">":
            return c > 0
        if op == ">=":
            return c >= 0
        if op == "<":
            return c < 0
        if op == "<=":
            return c <= 0
        raise InvalidRange("Invalid operator: " + op)


# ---------------------------------------------------------------- internal/trim-comparator-whitespace.js
def _is_ws(ch):
    return ch is not None and ch in WS_SET


def _is_prefix(ch):
    return ch == "v" or ch == "=" or _is_ws(ch)


def _is_digit(ch):
    return ch is not None and "0" <= ch <= "9"


def _is_letter(ch):
    return ch is not None and ("a" <= ch <= "z" or "A" <= ch <= "Z")


def _is_ident(ch):
    return _is_digit(ch) or _is_letter(ch) or ch == "-"


def _is_vstart(ch):
    return _is_digit(ch) or ch in ("x", "X", "*")


def _at(s, i):
    return s[i] if 0 <= i < len(s) else None


def _digits_end(s, start, limit):
    if not _is_digit(_at(s, start)):
        return start
    end = start + 1
    while _is_digit(_at(s, end)):
        end += 1
    return min(start + limit, end)


def _vnum_end(s, start, loose):
    if not loose and _at(s, start) == "0":
        return start + 1
    return _digits_end(s, start, MAX_LENGTH if loose else MAX_LENGTH + 1)


def _pre_ident_end(s, start, loose):
    d = _digits_end(s, start, MAX_LENGTH)
    if _is_letter(_at(s, d)) or _at(s, d) == "-":
        end = d + 1
        limit = end + MAX_SAFE_BUILD_LENGTH
        while end < limit and _is_ident(_at(s, end)):
            end += 1
        return end
    return _vnum_end(s, start, loose)


def _pre_end(s, start, loose):
    hyphen = _at(s, start) == "-"
    if not loose and not hyphen:
        return start
    istart = start + (1 if hyphen else 0)
    end = _pre_ident_end(s, istart, loose)
    if end == istart:
        if not loose or not hyphen:
            return start
        end = _pre_ident_end(s, start, loose)
    while _at(s, end) == ".":
        nxt = _pre_ident_end(s, end + 1, loose)
        if nxt == end + 1:
            break
        end = nxt
    return end


def _build_ident_end(s, start):
    end = start
    while end - start < MAX_SAFE_BUILD_LENGTH and _is_ident(_at(s, end)):
        end += 1
    return end


def _build_end(s, start):
    if _at(s, start) != "+":
        return start
    end = _build_ident_end(s, start + 1)
    if end == start + 1:
        return start
    while _at(s, end) == ".":
        nxt = _build_ident_end(s, end + 1)
        if nxt == end + 1:
            break
        end = nxt
    return end


def _loose_full_end(s, start):
    end = start
    for comp in range(3):
        nxt = _vnum_end(s, end, True)
        if nxt == end:
            return start
        end = nxt
        if comp < 2:
            if _at(s, end) != ".":
                return start
            end += 1
    return _build_end(s, _pre_end(s, end, True))


def _partial_comp_end(s, start):
    if _is_digit(_at(s, start)):
        return _vnum_end(s, start, False)
    return start + 1 if _is_vstart(_at(s, start)) else start


def _partial_end(s, start):
    end = _partial_comp_end(s, start)
    for _ in range(2):
        if _at(s, end) != ".":
            return end
        nxt = _partial_comp_end(s, end + 1)
        if nxt == end + 1:
            return end
        end = nxt
    return _build_end(s, _pre_end(s, end, False))


def _version_end(s, start):
    full = _loose_full_end(s, start)
    return full if full != start else _partial_end(s, start)


def _removable_ws(s, match_start):
    pos = match_start
    if _is_ws(_at(s, pos)):
        pos += 1
    if _at(s, pos) in ("<", ">"):
        pos += 1
    if _at(s, pos) == "=":
        pos += 1
    return pos if _is_ws(_at(s, pos)) else -1


def trim_comparator_whitespace(s: str) -> str:
    result, copy_from, cursor = [], 0, 0
    n = len(s)
    while cursor < n:
        match_start = cursor
        ch = s[cursor]
        if ch in ("<", ">"):
            cursor += 1
        elif not _is_prefix(ch) and not _is_vstart(ch):
            cursor += 1
            continue
        while cursor < n and _is_prefix(s[cursor]):
            cursor += 1
        if not _is_vstart(_at(s, cursor)):
            continue
        wp = _removable_ws(s, match_start)
        if wp != -1:
            result.append(s[copy_from:wp])
            copy_from = wp + 1
        cursor = _version_end(s, cursor)
    return "".join(result) + s[copy_from:]


# ---------------------------------------------------------------- classes/range.js
def _is_x(i):
    return (not i) or i.lower() == "x" or i == "*"


def _invalid_xorder(M, m, p):
    return (_is_x(M) and not _is_x(m)) or (_is_x(m) and p and not _is_x(p))


def _num(x) -> int:
    return int(x)


def _replace_tilde(comp: str, loose: bool = False) -> str:
    def f(mo):
        M, m, p, pr = mo.group(1), mo.group(2), mo.group(3), mo.group(4)
        if _is_x(M):
            return ""
        if _is_x(m):
            return f">={M}.0.0 <{_num(M) + 1}.0.0-0"
        if _is_x(p):
            return f">={M}.{m}.0 <{M}.{_num(m) + 1}.0-0"
        if pr:
            return f">={M}.{m}.{p}-{pr} <{M}.{_num(m) + 1}.0-0"
        return f">={M}.{m}.{p} <{M}.{_num(m) + 1}.0-0"
    return RE["TILDELOOSE" if loose else "TILDE"].sub(f, comp, count=1)


def _replace_caret(comp: str, loose: bool = False) -> str:
    def f(mo):
        M, m, p, pr = mo.group(1), mo.group(2), mo.group(3), mo.group(4)
        if _is_x(M):
            return ""
        if _is_x(m):
            return f">={M}.0.0 <{_num(M) + 1}.0.0-0"
        if _is_x(p):
            if M == "0":
                return f">={M}.{m}.0 <{M}.{_num(m) + 1}.0-0"
            return f">={M}.{m}.0 <{_num(M) + 1}.0.0-0"
        if pr:
            if M == "0":
                if m == "0":
                    return f">={M}.{m}.{p}-{pr} <{M}.{m}.{_num(p) + 1}-0"
                return f">={M}.{m}.{p}-{pr} <{M}.{_num(m) + 1}.0-0"
            return f">={M}.{m}.{p}-{pr} <{_num(M) + 1}.0.0-0"
        if M == "0":
            if m == "0":
                return f">={M}.{m}.{p} <{M}.{m}.{_num(p) + 1}-0"
            return f">={M}.{m}.{p} <{M}.{_num(m) + 1}.0-0"
        return f">={M}.{m}.{p} <{_num(M) + 1}.0.0-0"
    return RE["CARETLOOSE" if loose else "CARET"].sub(f, comp, count=1)


def _replace_xrange(comp: str, loose: bool = False) -> str:
    comp = js_trim(comp)

    def f(mo):
        ret = mo.group(0)
        gtlt, M, m, p = mo.group(1), mo.group(2), mo.group(3), mo.group(4)
        if _invalid_xorder(M, m, p):
            return comp
        xM = _is_x(M)
        xm = xM or _is_x(m)
        xp = xm or _is_x(p)
        anyX = xp
        if gtlt == "=" and anyX:
            gtlt = ""
        pr = ""
        if xM:
            ret = "<0.0.0-0" if gtlt in (">", "<") else "*"
        elif gtlt and anyX:
            if xm:
                m = 0
            p = 0
            if gtlt == ">":
                gtlt = ">="
                if xm:
                    M = _num(M) + 1
                    m = 0
                    p = 0
                else:
                    m = _num(m) + 1
                    p = 0
            elif gtlt == "<=":
                gtlt = "<"
                if xm:
                    M = _num(M) + 1
                else:
                    m = _num(m) + 1
            if gtlt == "<":
                pr = "-0"
            ret = f"{gtlt}{M}.{m}.{p}{pr}"
        elif xm:
            ret = f">={M}.0.0{pr} <{_num(M) + 1}.0.0-0"
        elif xp:
            ret = f">={M}.{m}.0{pr} <{M}.{_num(m) + 1}.0-0"
        return ret
    return RE["XRANGELOOSE" if loose else "XRANGE"].sub(f, comp, count=1)


def _split_join(comp: str, fn, trim_first: bool = True) -> str:
    base = js_trim(comp) if trim_first else comp
    return " ".join(fn(c) for c in SPLIT_WS.split(base))


def _parse_comparator(comp: str, loose: bool = False) -> str:
    comp = RE["BUILD"].sub("", comp, count=1)
    comp = _split_join(comp, lambda c: _replace_caret(c, loose))
    comp = _split_join(comp, lambda c: _replace_tilde(c, loose))
    comp = _split_join(comp, lambda c: _replace_xrange(c, loose), trim_first=False)
    comp = RE["STAR"].sub("", js_trim(comp), count=1)
    return comp


def _hyphen_replace(mo) -> str:
    frm, fM, fm, fp, fpr = mo.group(1), mo.group(2), mo.group(3), mo.group(4), mo.group(5)
    to, tM, tm, tp, tpr = mo.group(7), mo.group(8), mo.group(9), mo.group(10), mo.group(11)
    if _is_x(fM):
        frm = ""
    elif _is_x(fm):
        frm = f">={fM}.0.0"
    elif _is_x(fp):
        frm = f">={fM}.{fm}.0"
    elif fpr:
        frm = f">={frm}"
    else:
        frm = f">={frm}"
    if _is_x(tM):
        to = ""
    elif _is_x(tm):
        to = f"<{_num(tM) + 1}.0.0-0"
    elif _is_x(tp):
        to = f"<{tM}.{_num(tm) + 1}.0-0"
    elif tpr:
        to = f"<={tM}.{tm}.{tp}-{tpr}"
    else:
        to = f"<={to}"
    return js_trim(f"{frm} {to}")


def _parse_range(rng: str, loose: bool = False) -> list[Comparator]:
    rng = BUILDSTRIPRE.sub("", rng)
    rng = RE["HYPHENRANGELOOSE" if loose else "HYPHENRANGE"].sub(_hyphen_replace, rng, count=1)
    rng = trim_comparator_whitespace(rng)
    rng = RE["TILDETRIM"].sub(lambda mo: mo.group(1) + "~", rng)
    rng = RE["CARETTRIM"].sub(lambda mo: mo.group(1) + "^", rng)
    parts = " ".join(_parse_comparator(c, loose) for c in rng.split(" "))
    rlist = [RE["GTE0"].sub("", js_trim(c), count=1) for c in SPLIT_WS.split(parts)]
    if loose:  # range.js: in loose mode, comparators that do not parse are dropped
        rlist = [c for c in rlist if RE["COMPARATORLOOSE"].search(c)]
    comps = [Comparator(c, loose) for c in rlist]
    rmap: dict[str, Comparator] = {}
    for c in comps:
        if c.value == "<0.0.0-0":
            return [c]
        rmap[c.value] = c
    if len(rmap) > 1 and "" in rmap:
        del rmap[""]
    return list(rmap.values())


class Range:
    __slots__ = ("raw", "set", "loose")

    def __init__(self, text: str, loose: bool = False):
        self.loose = loose
        if not isinstance(text, str):
            raise InvalidRange("not a string")
        self.raw = " ".join(SPLIT_WS.split(js_trim(text))) if js_trim(text) else ""
        sets = [_parse_range(js_trim(r), loose) for r in self.raw.split("||")]
        sets = [s for s in sets if len(s)]
        if not sets:
            raise InvalidRange("Invalid SemVer Range: " + self.raw)
        if len(sets) > 1:
            first = sets[0]
            sets = [s for s in sets if s[0].value != "<0.0.0-0"]
            if not sets:
                sets = [first]
            elif len(sets) > 1:
                for s in sets:
                    if len(s) == 1 and s[0].value == "":
                        sets = [s]
                        break
        self.set = sets

    def test(self, v: SemVer) -> bool:
        return any(_test_set(s, v) for s in self.set)


def _test_set(cs: list[Comparator], v: SemVer) -> bool:
    for c in cs:
        if not c.test(v):
            return False
    if v.prerelease:
        for c in cs:
            if c.semver is ANY:
                continue
            if c.semver.prerelease and (c.semver.major, c.semver.minor, c.semver.patch) == (v.major, v.minor, v.patch):
                return True
        return False
    return True


@lru_cache(maxsize=1 << 18)
def parse_range(text: str, loose: bool = False) -> Range | None:
    """Range object, or None when node-semver would throw (an unresolvable constraint)."""
    try:
        return Range(text, loose)
    except (InvalidRange, InvalidVersion, ValueError):
        return None


def satisfies(version: str, range_text: str, loose: bool = False) -> bool:
    """semver.satisfies(version, range, {loose, includePrerelease: false})."""
    r = parse_range(range_text, loose)
    if r is None:
        return False
    v = parse_version(version, loose) if isinstance(version, str) else None
    if v is None:
        return False
    return r.test(v)


def admitted(versions: Iterable[str], range_text: str, loose: bool = False) -> list[str]:
    r = parse_range(range_text, loose)
    if r is None:
        return []
    out = []
    for text in versions:
        v = parse_version(text, loose)
        if v is not None and r.test(v):
            out.append(text)
    return out


def highest(versions: Iterable[str], loose: bool = False) -> str | None:
    best, best_v = None, None
    for text in versions:
        v = parse_version(text, loose)
        if v is None:
            continue
        if best_v is None or v.compare(best_v) > 0:
            best, best_v = text, v
    return best


# E02/E03 compatible interface: unsupported syntax raises ValueError, as the E02 resolver did.
def study_satisfies(version_text: str, range_text: str) -> bool:
    if parse_range(range_text) is None:
        raise ValueError(f"unsupported or invalid range: {range_text}")
    return satisfies(version_text, range_text)


def study_admitted(versions: Iterable[str], range_text: str) -> list[str]:
    if parse_range(range_text) is None:
        raise ValueError(f"unsupported or invalid range: {range_text}")
    return admitted(versions, range_text)
