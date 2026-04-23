# """
# api/main.py — FastAPI application entry point.

# Mounts:
#   /api/v1/documents  — upload, list, delete documents
#   /api/v1/qa         — ask questions, get answers
#   /api/v1/sessions   — session management, history
#   /api/v1/health     — health check
# """
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware

# from core.config import settings
# from core.logging import configure_logging, get_logger
# from api.routers import documents, qa, sessions

# configure_logging()
# logger = get_logger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Startup and shutdown lifecycle."""
#     logger.info("docsage_starting", version=settings.app_version, env=settings.environment)
#     settings.create_dirs()
#     # Warm up pipeline (lazy-load models on first request to avoid blocking startup)
#     from core.pipeline import DocSagePipeline
#     DocSagePipeline.get()
#     logger.info("docsage_ready")
#     yield
#     logger.info("docsage_shutdown")


# app = FastAPI(
#     title="DocSage API",
#     description="Smart Document Question Answering System",
#     version=settings.app_version,
#     lifespan=lifespan,
#     docs_url="/api/docs",
#     redoc_url="/api/redoc",
# )

# # ── Middleware ────────────────────────────────────────────────────────────────
# app.add_middleware(GZipMiddleware, minimum_size=1000)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=settings.cors_origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── Routers ───────────────────────────────────────────────────────────────────
# app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"])
# app.include_router(qa.router, prefix=f"{settings.api_prefix}/qa", tags=["QA"])
# app.include_router(sessions.router, prefix=f"{settings.api_prefix}/sessions", tags=["Sessions"])


# @app.get(f"{settings.api_prefix}/health", tags=["Health"])
# async def health():
#     return {"status": "ok", "version": settings.app_version}
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.middleware.gzip import GZipMiddleware

# from core.config import settings
# from core.logging import configure_logging, get_logger
# from api.routers import documents, qa, sessions

# configure_logging()
# logger = get_logger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Startup and shutdown lifecycle."""
#     logger.info("docsage_starting", version=settings.app_version, env=settings.environment)
#     settings.create_dirs()

#     # ❌ REMOVE MODEL LOADING (VERY IMPORTANT)
#     # from core.pipeline import DocSagePipeline
#     # DocSagePipeline.get()

#     logger.info("docsage_ready")
#     yield
#     logger.info("docsage_shutdown")


# app = FastAPI(
#     title="DocSage API",
#     description="Smart Document Question Answering System",
#     version=settings.app_version,
#     lifespan=lifespan,
#     docs_url="/api/docs",
#     redoc_url="/api/redoc",
# )

# # ── Middleware ───────────────────────────────────────────────

# app.add_middleware(GZipMiddleware, minimum_size=1000)

# # ✅ FIX CORS (FORCE ALLOW)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # 🔥 FIXED
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ── Routers ──────────────────────────────────────────────────

# app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"])
# app.include_router(qa.router, prefix=f"{settings.api_prefix}/qa", tags=["QA"])
# app.include_router(sessions.router, prefix=f"{settings.api_prefix}/sessions", tags=["Sessions"])


# @app.get(f"{settings.api_prefix}/health", tags=["Health"])
# async def health():
#     return {"status": "ok", "version": settings.app_version}
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from core.config import settings
from core.logging import configure_logging, get_logger
from api.routers import documents, qa, sessions

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── CRITICAL: Do NOT load any models here ──────────────────────────
    # Render's health check hits /health within 5 seconds of startup.
    # Loading RoBERTa + MiniLM + FAISS takes 3-5 minutes on free tier.
    # If we load models here, health check times out → Bad Gateway.
    # Models load lazily on the FIRST actual request instead.
    # ────────────────────────────────────────────────────────────────────
    settings.create_dirs()
    logger.info("docsage_ready", note="models load on first request")
    yield
    logger.info("docsage_shutdown")


app = FastAPI(
    title="DocSage API",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────
_raw_origins = os.environ.get("CORS_ORIGINS", "*")
if _raw_origins == "*":
    cors_origins = ["*"]
    cors_credentials = False
else:
    cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    cors_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ───────────────────────────────────────────────────────────
app.include_router(
    documents.router,
    prefix=f"{settings.api_prefix}/documents",
    tags=["Documents"]
)
app.include_router(
    qa.router,
    prefix=f"{settings.api_prefix}/qa",
    tags=["QA"]
)
app.include_router(
    sessions.router,
    prefix=f"{settings.api_prefix}/sessions",
    tags=["Sessions"]
)


@app.get("/", tags=["Health"])
async def root():
    """Root health check — responds instantly, no model loading."""
    return {"status": "ok", "service": "DocSage API"}


@app.get(f"{settings.api_prefix}/health", tags=["Health"])
async def health():
    """Health check — responds instantly, no model loading."""
    return {"status": "ok", "version": settings.app_version}