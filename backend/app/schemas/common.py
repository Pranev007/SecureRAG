"""Shared API schema pieces."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class Page(BaseModel, Generic[T]):
    """A page of results plus the total, so a UI can render a count."""

    items: list[T]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    blocked: bool | None = None
    reason: str | None = None


class ErrorResponse(BaseModel):
    """The single error envelope every failing endpoint returns."""

    error: ErrorDetail


class MessageResponse(BaseModel):
    message: str
