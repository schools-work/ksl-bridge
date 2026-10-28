"""Knowledge SQL: hybrid search through knowledge.hybrid_search (vector) or keyword-only fallback."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection


class SqlKnowledgeRepo:
    def __init__(self, conn: AsyncConnection):
        self.c = conn

    async def search(self, venue_id: UUID | None, q: str, qvec: list[float] | None, k: int = 5) -> list[dict[str, Any]]:
        if qvec is not None:
            rows = await self.c.execute(
                text(
                    "SELECT chunk_id, document_id, section_path, content, score FROM "
                    "knowledge.hybrid_search(:v, :q, CAST(:e AS vector), :k)"
                ),
                {"v": venue_id, "q": q, "e": str(qvec), "k": k},
            )
        else:
            rows = await self.c.execute(
                text(
                    "WITH kw AS (SELECT c.id AS chunk_id, c.document_id, c.section_path, c.content, "
                    "ts_rank_cd(c.tsv, websearch_to_tsquery('english', :q)) AS ks FROM knowledge.chunks c "
                    "JOIN knowledge.documents d ON d.id = c.document_id AND d.status = 'active' "
                    "WHERE (c.venue_id = :v OR c.venue_id IS NULL) AND c.tsv @@ websearch_to_tsquery('english', "
                    "regexp_replace(:q, '\\s+', ' or ', 'g'))) "
                    "SELECT chunk_id, document_id, section_path, content, (ks / NULLIF(max(ks) OVER (), 0))::real AS score "
                    "FROM kw ORDER BY score DESC NULLS LAST LIMIT :k"
                ),
                {"v": venue_id, "q": q, "k": k},
            )
        return [dict(r) for r in rows.mappings()]

    async def synonyms(self) -> dict[str, list[str]]:
        rows = await self.c.execute(
            text(
                "SELECT t1.phrase, array_agg(DISTINCT t2.phrase) FROM lexicon.translations t1 "
                "JOIN lexicon.translations t2 ON t2.gloss = t1.gloss AND t2.phrase <> t1.phrase GROUP BY t1.phrase"
            )
        )
        return {r[0]: list(r[1]) for r in rows}
