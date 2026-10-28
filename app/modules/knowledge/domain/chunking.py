"""Chunking (guest RAG lecture): recursive character, structure-aware Markdown, table rows. Pure."""

from __future__ import annotations

import re


def recursive_split(
    text: str, size: int = 500, overlap: int = 50, seps: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")
) -> list[str]:
    """Same behaviour as RecursiveCharacterTextSplitter; overlap snaps to a word boundary."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    sep = next((s for s in seps if s and s in text), "")
    parts = text.split(sep) if sep else list(text)
    rest = seps[seps.index(sep) + 1 :] if sep in seps else ("",)
    chunks: list[str] = []
    cur = ""
    for p in parts:
        cand = f"{cur}{sep}{p}" if cur else p
        if len(cand) <= size:
            cur = cand
            continue
        if cur:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1 :] if " " in tail else ""
            cur = f"{tail}{sep}{p}" if tail and len(tail) + len(sep) + len(p) <= size else p
        else:
            cur = p
        if len(cur) > size:
            chunks.extend(recursive_split(cur, size, overlap, rest or ("",)))
            cur = ""
    if cur.strip():
        chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def markdown_sections(md: str) -> list[tuple[str, str]]:
    """Split on headings; keep the heading path ('Menu > Drinks') as metadata."""
    out: list[tuple[str, str]] = []
    path: list[str] = []
    buf: list[str] = []
    for line in md.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            if buf and "\n".join(buf).strip():
                out.append((" > ".join(path), "\n".join(buf).strip()))
            buf = []
            level = len(m.group(1))
            path = [*path[: level - 1], m.group(2).strip()]
        else:
            buf.append(line)
    if buf and "\n".join(buf).strip():
        out.append((" > ".join(path), "\n".join(buf).strip()))
    return out


def menu_rows(rows: list[dict[str, object]]) -> list[str]:
    """Tables are never split mid-row: one chunk per item, header repeated."""
    return [
        f"Menu item: {r['name']}. Category: {r.get('category', '')}. Price: KSh {r.get('price', '')}. "
        f"Description: {r.get('description', '')}".strip()
        for r in rows
    ]
