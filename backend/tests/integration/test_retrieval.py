"""Retrieval: vector, keyword, hybrid fusion, reranking and access scoping."""

from __future__ import annotations

import pytest

from app.core.exceptions import ProviderError
from app.rag.pipeline import RagPipeline
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.keyword import get_keyword_searcher, tokenize
from app.rag.retrieval.retriever import Retriever
from app.rag.retrieval.types import AccessScope, ScoredChunk
from app.rag.retrieval.vector_store import get_vector_store
from app.services.document_service import DocumentService
from tests.factories import HANDBOOK_MARKDOWN, SECURITY_POLICY_MARKDOWN

pytestmark = pytest.mark.integration


@pytest.fixture
def corpus(db, user, other_user):
    """Two documents owned by `user`, one owned by `other_user`."""
    service = DocumentService(db)
    handbook = service.ingest_upload(
        owner=user, filename="handbook.md", data=HANDBOOK_MARKDOWN.encode()
    )
    policy = service.ingest_upload(
        owner=user, filename="security_policy.md", data=SECURITY_POLICY_MARKDOWN.encode()
    )
    secret = service.ingest_upload(
        owner=other_user,
        filename="salaries.md",
        data=(
            b"# Compensation\n\n## Executive Pay\n\n"
            b"The Chief Executive receives a base salary of 450000 per year "
            b"plus an annual performance bonus of up to 40 percent."
        ),
    )
    return {"handbook": handbook, "policy": policy, "secret": secret}


# ----------------------------------------------------------------------
# Vector arm
# ----------------------------------------------------------------------


def test_vector_search_returns_relevant_chunks(db, user, corpus):
    from app.rag.embeddings.factory import get_embedding_provider

    query_vector = get_embedding_provider().embed_query(
        "How many days of annual leave do employees get?"
    )
    hits = get_vector_store(db).search(
        db, query_vector, AccessScope(user_id=user.id), limit=5
    )

    assert hits
    assert any("annual leave" in hit.content.lower() for hit in hits)
    assert all(hit.owner_id == user.id for hit in hits)
    # Scores must be ordered and carry the arm that produced them.
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all("vector" in h.rank_sources for h in hits)


def test_vector_search_never_returns_another_users_chunks(db, user, corpus):
    from app.rag.embeddings.factory import get_embedding_provider

    query_vector = get_embedding_provider().embed_query(
        "What is the Chief Executive base salary and bonus?"
    )
    hits = get_vector_store(db).search(
        db, query_vector, AccessScope(user_id=user.id), limit=20
    )

    assert all(hit.owner_id == user.id for hit in hits)
    assert all("450000" not in hit.content for hit in hits)
    assert all(hit.document_id != corpus["secret"].id for hit in hits)


def test_access_scope_requires_a_user_id():
    with pytest.raises(ValueError, match="user_id"):
        AccessScope(user_id="")


def test_scope_can_narrow_to_specific_documents(db, user, corpus):
    from app.rag.embeddings.factory import get_embedding_provider

    query_vector = get_embedding_provider().embed_query("leave policy and passwords")
    scope = AccessScope(user_id=user.id, document_ids=[corpus["policy"].id])
    hits = get_vector_store(db).search(db, query_vector, scope, limit=20)

    assert hits
    assert all(hit.document_id == corpus["policy"].id for hit in hits)


# ----------------------------------------------------------------------
# Keyword arm
# ----------------------------------------------------------------------


def test_tokenizer_drops_stopwords():
    assert tokenize("What is the leave policy?") == ["leave", "policy"]


def test_keyword_search_finds_exact_terms(db, user, corpus):
    hits = get_keyword_searcher(db).search(
        db, "multi-factor authentication", AccessScope(user_id=user.id), limit=5
    )
    assert hits
    assert any("multi-factor" in hit.content.lower() for hit in hits)
    assert all("keyword" in hit.rank_sources for hit in hits)


def test_keyword_search_is_access_scoped(db, user, corpus):
    hits = get_keyword_searcher(db).search(
        db, "Chief Executive base salary 450000", AccessScope(user_id=user.id), limit=20
    )
    assert all(hit.owner_id == user.id for hit in hits)


def test_keyword_search_with_only_stopwords_returns_nothing(db, user, corpus):
    assert (
        get_keyword_searcher(db).search(db, "the of and", AccessScope(user_id=user.id), 5)
        == []
    )


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------


def _stub(chunk_id: str, arm: str, rank: int) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        document_id="doc",
        content=f"content {chunk_id}",
        source_filename="f.md",
        chunk_index=int(chunk_id[-1]),
        rank_sources={arm: rank},
        score=1.0 / rank,
    )


def test_rrf_rewards_documents_found_by_both_arms():
    vector = [_stub("c1", "vector", 1), _stub("c2", "vector", 2)]
    keyword = [_stub("c3", "keyword", 1), _stub("c1", "keyword", 2)]

    fused = reciprocal_rank_fusion([vector, keyword], k=60)

    # c1 appears in both lists, so it must outrank the arms' individual leaders.
    assert fused[0].chunk_id == "c1"
    assert set(fused[0].rank_sources) == {"vector", "keyword"}


def test_rrf_is_deterministic_for_tied_scores():
    left = [_stub("c1", "vector", 1)]
    right = [_stub("c2", "keyword", 1)]
    first = [c.chunk_id for c in reciprocal_rank_fusion([left, right], k=60)]
    second = [c.chunk_id for c in reciprocal_rank_fusion([left, right], k=60)]
    assert first == second


def test_rrf_does_not_mutate_input_lists():
    vector = [_stub("c1", "vector", 1)]
    original = vector[0].score
    reciprocal_rank_fusion([vector], k=60)
    assert vector[0].score == original


# ----------------------------------------------------------------------
# End-to-end retriever
# ----------------------------------------------------------------------


def test_hybrid_retrieval_answers_from_the_right_document(db, user, corpus):
    result = Retriever().retrieve(
        db,
        "How many days of annual leave do full-time employees accrue?",
        AccessScope(user_id=user.id),
        top_k=3,
    )

    assert result.mode == "hybrid"
    assert not result.is_empty
    assert result.chunks[0].source_filename == "handbook.md"
    assert "leave" in result.chunks[0].content.lower()
    assert result.vector_candidates > 0
    assert result.keyword_candidates > 0
    assert set(result.timings_ms) >= {"embed_ms", "vector_ms", "keyword_ms", "rerank_ms"}


def test_retrieval_respects_top_k(db, user, corpus):
    result = Retriever().retrieve(
        db, "leave policy", AccessScope(user_id=user.id), top_k=2
    )
    assert len(result.chunks) <= 2


def test_retrieved_chunks_expose_citation_labels(db, user, corpus):
    result = Retriever().retrieve(
        db, "expense claims deadline", AccessScope(user_id=user.id), top_k=3
    )
    assert result.chunks
    for chunk in result.chunks:
        label = chunk.citation_label()
        assert chunk.source_filename in label


def test_reranker_records_explainable_features(db, user, corpus):
    result = Retriever().retrieve(
        db, "multi-factor authentication requirement", AccessScope(user_id=user.id)
    )
    assert result.chunks
    features = result.chunks[0].meta["rerank_features"]
    assert set(features) == {
        "retrieval",
        "coverage",
        "phrase",
        "length",
        "injection_penalty",
    }


def test_retrieval_for_a_user_with_no_documents_is_empty(db, other_user):
    result = Retriever().retrieve(
        db, "anything at all", AccessScope(user_id=other_user.id)
    )
    assert result.is_empty


def test_keyword_only_and_vector_only_modes_both_work(db, user, corpus):
    retriever = Retriever()
    keyword_only = retriever.retrieve(
        db, "multi-factor authentication", AccessScope(user_id=user.id), mode="keyword"
    )
    vector_only = retriever.retrieve(
        db, "multi-factor authentication", AccessScope(user_id=user.id), mode="vector"
    )

    assert keyword_only.chunks and keyword_only.keyword_candidates > 0
    assert keyword_only.vector_candidates == 0
    assert vector_only.chunks and vector_only.vector_candidates > 0
    assert vector_only.keyword_candidates == 0


# ----------------------------------------------------------------------
# What a refusal reports about retrieval
# ----------------------------------------------------------------------


def test_a_provider_failure_still_reports_what_retrieval_found(db, user, corpus):
    """A failed generation must not be indistinguishable from a failed retrieval.

    The provider-error branch used to build its response without the retrieval
    fields, so every provider outage reported zero chunks. In the UI that is
    identical to a genuine retrieval miss, and it was duly filed as a separate
    bug and investigated as one -- twice -- before the stage timings gave it
    away: `sanitise` cannot run on an empty retrieval, so its presence proved
    chunks had been found all along.

    `retrieved_documents` carries a field comment saying it exists so retrieval
    stays measurable regardless of whether the answer was refused. This is that
    promise under test.
    """

    class FailingGenerator:
        def generate(self, question, chunks):
            raise ProviderError(internal_detail="HTTP 401 from LLM provider")

    pipeline = RagPipeline(generator=FailingGenerator())

    response = pipeline.answer(
        db, "How many days of annual leave do employees get?", user=user
    )

    assert response.refused
    assert response.reason == "provider_unavailable"
    # The point of the test.
    assert response.retrieved_chunk_count > 0
    assert "handbook.md" in response.retrieved_documents
    # The stage timings that exposed the original bug must survive too.
    assert "sanitise_ms" in response.timings_ms
