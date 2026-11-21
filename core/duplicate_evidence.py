"""Content evidence for a "likely" duplicate group (#837).

A likely group is formed from tags and file names, so a wrong tag makes a wrong
group: Magic 7 #4 and #5 both tagged #4, or a "Widescreen Edition" tagged as
the issue it reformats. The files themselves settle it:

- **Shared pages.** The CRC-32 and size of every page image, read from the
  zip's central directory -- no page is decompressed. Two scans of one release
  share most of them; two different comics share none.
- **The cover.** ``core.cover_hash`` (pHash) of the first page, for two scans
  of the same issue that share no bytes.

The verdict is ``same`` (the content says duplicate), ``different`` (it says
not), or ``unknown`` (could not read, or the signals disagree). Only the zip
formats are read; anything else is ``unknown`` and stays a likely group, as
before.
"""
import os
import time
import zipfile
from functools import lru_cache

from core.app_logging import app_logger

PAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".jxl", ".bmp")

#: Share of the smaller file's pages found in the other: one release.
SAME_PAGE_SHARE = 0.5
#: ...and at least this many pages in common. A 2-page cover pack shares its
#: one cover with the issue: 50%, and not a copy of it.
MIN_SHARED_PAGES = 4
# Calibrated on 562 real pairs (copies of one issue and wrongly grouped
# different issues), labelled by a 256-bit dHash checked by eye: the 18
# same-cover pairs were 0-6 bits apart (16) or 8 and 16; the 527
# different-cover pairs were never under 8 and 505 were 22 or more. pHash's 64
# bits put unrelated covers close, so "same" keeps a 2-bit margin under the
# nearest different pair, and everything between the two stays "unknown".
#: pHash bits (of 64) at or under which two covers are the same image.
SAME_COVER_BITS = 6
#: pHash bits at or over which two covers are different images.
DIFFERENT_COVER_BITS = 22

#: Seconds one find_duplicates call may spend reading files. The rest of the
#: groups are "unknown" this time and read on a later call (results are cached).
EVIDENCE_BUDGET_SECONDS = 20


@lru_cache(maxsize=8192)
def _pages(path, size, mtime):
    """``frozenset`` of (CRC-32, size) per page image, or None when unreadable."""
    if not path.lower().endswith((".cbz", ".zip")):
        return None
    try:
        with zipfile.ZipFile(path) as z:
            return frozenset(
                (i.CRC, i.file_size) for i in z.infolist()
                if not i.is_dir() and i.filename.lower().endswith(PAGE_EXTS)
            )
    except Exception as e:
        app_logger.debug(f"Duplicate evidence: cannot list {os.path.basename(path)}: {e}")
        return None


@lru_cache(maxsize=8192)
def _cover(path, size, mtime):
    from core.cover_hash import cover_hashes
    return tuple(cover_hashes(path))


def _key(member):
    return member["path"], member.get("size") or 0, member.get("modified_at") or 0


def _compare_pair(a, b):
    """(verdict, page_share, cover_bits) for two members."""
    from core.cover_hash import hamming

    pa, pb = _pages(*_key(a)), _pages(*_key(b))
    share = None
    if pa and pb:
        shared = len(pa & pb)
        share = shared / min(len(pa), len(pb))
        if share >= SAME_PAGE_SHARE and shared >= MIN_SHARED_PAGES:
            return "same", share, None
    ca, cb = _cover(*_key(a)), _cover(*_key(b))
    bits = min(hamming(x, y) for x in ca for y in cb) if ca and cb else None
    if bits is None:
        return "unknown", share, None
    if bits <= SAME_COVER_BITS:
        # A cover pack or a collected edition carries the issue's cover too:
        # a matching cover says "same" only between two whole, similar-length comics.
        if pa and pb and (min(len(pa), len(pb)) < MIN_SHARED_PAGES
                          or max(len(pa), len(pb)) > 2 * min(len(pa), len(pb))):
            return "unknown", share, bits
        return "same", share, bits
    if bits >= DIFFERENT_COVER_BITS and not share:
        return "different", share, bits
    return "unknown", share, bits


def group_evidence(members):
    """Evidence for one group: ``{verdict, pairs, superset}``.

    ``verdict`` is ``same`` only when every member matches the first, and
    ``different`` only when every other member differs from it. ``superset``
    is the path of a member holding every page of every other member and more,
    which is the copy to keep.
    """
    first, rest = members[0], members[1:]
    pairs = []
    for other in rest:
        verdict, share, bits = _compare_pair(first, other)
        pairs.append({"path": other["path"], "verdict": verdict,
                      "page_share": None if share is None else round(share, 2),
                      "cover_bits": bits})
    verdicts = {p["verdict"] for p in pairs}
    verdict = verdicts.pop() if len(verdicts) == 1 else "unknown"

    superset = None
    sets = {m["path"]: _pages(*_key(m)) for m in members}
    if all(sets.values()):
        for path, pages in sets.items():
            others = [s for p, s in sets.items() if p != path]
            if all(pages > s for s in others):
                superset = path
                break
    return {"verdict": verdict, "pairs": pairs, "superset": superset}


def describe(evidence):
    """One line for the review page."""
    parts = []
    for p in evidence["pairs"]:
        bits = []
        if p["page_share"] is not None:
            bits.append(f"{round(p['page_share'] * 100)}% pages shared")
        if p["cover_bits"] is not None:
            bits.append(f"cover {p['cover_bits']}/64 bits apart")
        parts.append(", ".join(bits) or "content not readable")
    verdict = {"same": "content matches", "different": "content differs",
               "unknown": "content not conclusive"}[evidence["verdict"]]
    return f"{verdict} ({'; '.join(parts)})"


class Budget:
    """Wall-clock allowance for one find_duplicates call."""

    def __init__(self, seconds=EVIDENCE_BUDGET_SECONDS, clock=time.monotonic):
        self._clock = clock
        self._deadline = clock() + seconds

    def left(self):
        return self._clock() < self._deadline
