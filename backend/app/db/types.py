"""Portable column types.

SecureRAG targets PostgreSQL + pgvector in production, but the test suite and
offline development run on SQLite so the project can be cloned and verified
with no infrastructure.  Rather than maintaining two sets of models, these
:class:`~sqlalchemy.types.TypeDecorator` implementations resolve to the native
PostgreSQL type when available and to a portable equivalent otherwise.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import Float, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.engine import Dialect
from sqlalchemy.types import JSON

# Sentinel bound in place of an unparseable identifier. It is a valid UUID, so
# the query executes on both dialects, and it is never generated as a real
# primary key, so it can only ever match zero rows.
NIL_UUID = "00000000-0000-0000-0000-000000000000"


class GUID(TypeDecorator):
    """UUID stored as native ``uuid`` on PostgreSQL, ``CHAR(36)`` elsewhere.

    Identifiers arrive from URL paths and request bodies, so they are
    attacker-controlled.  A malformed value is bound as :data:`NIL_UUID`
    rather than raising: raising here turns ``GET /documents/../etc/passwd``
    into a 500 with a stack trace, whereas binding a never-matching id lets
    the normal "not found" path handle it -- which is both the correct HTTP
    answer and the one that leaks nothing.
    """

    impl = String(36)
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=False))
        return dialect.type_descriptor(String(36))

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        if isinstance(value, uuid.UUID):
            return str(value)
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, AttributeError, TypeError):
            return NIL_UUID

    def process_result_value(self, value: Any, dialect: Dialect) -> str | None:
        if value is None:
            return None
        return str(value)


class JSONBType(TypeDecorator):
    """``JSONB`` on PostgreSQL, plain ``JSON`` elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())


class VectorType(TypeDecorator):
    """Embedding column: ``vector(n)`` on PostgreSQL, JSON array elsewhere.

    On PostgreSQL this is a real ``pgvector`` column, which means ANN indexes
    (HNSW / IVFFlat) and the ``<=>`` cosine-distance operator execute inside the
    database.  On SQLite the values round-trip as a JSON array and similarity is
    computed in Python by the fallback vector store -- correct, but O(n) per
    query, and therefore intended only for tests and small local corpora.
    """

    impl = JSON
    cache_ok = True

    def __init__(self, dimensions: int, *args: Any, **kwargs: Any) -> None:
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector  # imported lazily: PG only

            return dialect.type_descriptor(Vector(self.dimensions))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Sequence[float] | None, dialect: Dialect) -> Any:
        if value is None:
            return None
        return [float(x) for x in value]

    def process_result_value(self, value: Any, dialect: Dialect) -> list[float] | None:
        if value is None:
            return None
        return [float(x) for x in value]

    class comparator_factory(TypeDecorator.Comparator):  # noqa: N801
        """Expose pgvector's distance operators on the decorated type."""

        def cosine_distance(self, other: Any) -> Any:
            return self.op("<=>", return_type=Float)(other)

        def l2_distance(self, other: Any) -> Any:
            return self.op("<->", return_type=Float)(other)

        def max_inner_product(self, other: Any) -> Any:
            return self.op("<#>", return_type=Float)(other)
