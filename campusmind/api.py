import logging
import sqlite3

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from campusmind.auth import create_access_token, decode_access_token
from campusmind.database import analytics_for_user, authenticate_user, create_user, get_user_by_id, init_db
from campusmind.evaluation import evaluate_item
from campusmind.fast_engine import FastCampusMindEngine, result_to_jsonable
from campusmind.schemas import ChatRequest, ChatResponse, EvaluationItem, LoginRequest, SignupRequest, TokenResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="CampusMind AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: FastCampusMindEngine | None = None


def _validation_reason(exc: RequestValidationError) -> str:
    first = exc.errors()[0]
    field = str(first.get("loc", ["input"])[-1])
    message = str(first.get("msg", "invalid input")).removeprefix("Value error, ")
    custom_messages = {
        "name": "name must be 2-50 characters and not empty",
        "email": "email must be in valid email format",
        "password": "password must be at least 8 characters and contain no spaces",
        "age": "age must be an integer between 10 and 100",
        "gender": "gender must be exactly one of: male, female, other",
        "education_level": "education_level must not be empty",
        "interests": "interests must not be empty",
        "mental_health_sensitive": "mental_health_sensitive must be a boolean",
    }
    return custom_messages.get(field, message)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"valid": False, "reason": _validation_reason(exc)},
        media_type="application/json; charset=utf-8",
    )


@app.on_event("startup")
def startup() -> None:
    global engine
    init_db()
    engine = FastCampusMindEngine()


def current_user(authorization: str = Header(default="")) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_access_token(token)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc
    user = get_user_by_id(str(payload["sub"]))
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user.pop("password_hash", None)
    return user


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, media_type="application/json; charset=utf-8")


@app.post("/auth/signup", response_model=TokenResponse)
def signup(payload: SignupRequest) -> TokenResponse:
    try:
        user = create_user(payload)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=400, detail={"valid": False, "reason": "email already exists"}) from exc
    token = create_access_token(user["user_id"], {"email": user["email"]})
    return TokenResponse(access_token=token, user=user)


@app.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest) -> TokenResponse:
    user = authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token = create_access_token(user["user_id"], {"email": user["email"]})
    return TokenResponse(access_token=token, user=user)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, user: dict = Depends(current_user)) -> ChatResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Chat engine is not ready")
    result = engine.chat(user_id=user["user_id"], session_id=payload.session_id, message=payload.message)
    return ChatResponse(**result_to_jsonable(result))


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest, user: dict = Depends(current_user)) -> StreamingResponse:
    if engine is None:
        raise HTTPException(status_code=503, detail="Chat engine is not ready")
    return StreamingResponse(
        engine.stream_chat(user_id=user["user_id"], session_id=payload.session_id, message=payload.message),
        media_type="text/plain; charset=utf-8",
    )


@app.get("/analytics")
def analytics(user: dict = Depends(current_user)) -> dict:
    return analytics_for_user(user["user_id"])


@app.post("/evaluate")
def evaluate(payload: EvaluationItem) -> dict:
    return evaluate_item(payload.prediction, payload.reference, payload.context)
