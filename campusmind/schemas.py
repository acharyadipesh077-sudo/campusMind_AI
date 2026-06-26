import re
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, field_validator


Domain = Literal["Academic", "Mental Health", "Emergency", "Unknown"]


class SignupRequest(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    age: int = Field(ge=10, le=100)
    gender: Literal["male", "female", "other"]
    education_level: str = Field(min_length=1)
    interests: str = Field(min_length=1)
    mental_health_sensitive: bool = False

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be empty")
        if not 2 <= len(value) <= 50:
            raise ValueError("name must be 2-50 characters")
        return value

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value) < 8:
            raise ValueError("password must be at least 8 characters")
        if re.search(r"\s", value):
            raise ValueError("password must contain no spaces")
        return value

    @field_validator("education_level", "interests")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("education_level and interests must not be empty")
        return value


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(min_length=3, max_length=120)


class RetrievedDocument(BaseModel):
    doc_id: int
    question: str
    context: str
    answer: str
    domain: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    domain: Domain
    confidence: float
    rewritten_query: str
    retrieved_documents: list[RetrievedDocument]
    workflow: list[str]
    emergency: bool = False
    cached: bool = False
    latency_ms: int = 0


class EvaluationItem(BaseModel):
    prediction: str
    reference: str
    context: str | None = None
