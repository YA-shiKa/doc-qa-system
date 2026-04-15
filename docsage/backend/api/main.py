# """
# api/main.py — FastAPI application entry point.

# Mounts:
#   /api/v1/documents  — upload, list, delete documents
#   /api/v1/qa         — ask questions, get answers
#   /api/v1/sessions   — session management, history
#   /api/v1/health     — health check
# """
# # from contextlib import asynccontextmanager
# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.middleware.gzip import GZipMiddleware

# # from core.config import settings
# # from core.logging import configure_logging, get_logger
# # from api.routers import documents, qa, sessions

# # configure_logging()
# # logger = get_logger(__name__)


# # @asynccontextmanager
# # async def lifespan(app: FastAPI):
# #     """Startup and shutdown lifecycle."""
# #     logger.info("docsage_starting", version=settings.app_version, env=settings.environment)
# #     settings.create_dirs()
# #     # Warm up pipeline (lazy-load models on first request to avoid blocking startup)
# #     from core.pipeline import DocSagePipeline
# #     DocSagePipeline.get()
# #     logger.info("docsage_ready")
# #     yield
# #     logger.info("docsage_shutdown")


# # app = FastAPI(
# #     title="DocSage API",
# #     description="Smart Document Question Answering System",
# #     version=settings.app_version,
# #     lifespan=lifespan,
# #     docs_url="/api/docs",
# #     redoc_url="/api/redoc",
# # )

# # # ── Middleware ────────────────────────────────────────────────────────────────
# # app.add_middleware(GZipMiddleware, minimum_size=1000)
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=settings.cors_origins,
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # ── Routers ───────────────────────────────────────────────────────────────────
# # app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"])
# # app.include_router(qa.router, prefix=f"{settings.api_prefix}/qa", tags=["QA"])
# # app.include_router(sessions.router, prefix=f"{settings.api_prefix}/sessions", tags=["Sessions"])


# # @app.get(f"{settings.api_prefix}/health", tags=["Health"])
# # async def health():
# #     return {"status": "ok", "version": settings.app_version}
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


# # ── Lifespan (NO heavy loading) ───────────────────────────────
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     logger.info("docsage_starting", version=settings.app_version, env=settings.environment)
#     settings.create_dirs()

#     # ❌ DO NOT load pipeline here (prevents HuggingFace timeout)

#     logger.info("docsage_ready")
#     yield
#     logger.info("docsage_shutdown")


# # ── App Init ──────────────────────────────────────────────────
# app = FastAPI(
#     title="DocSage API",
#     description="Smart Document Question Answering System",
#     version=settings.app_version,
#     lifespan=lifespan,
#     docs_url="/api/docs",
#     redoc_url="/api/redoc",
# )


# # ── Middleware (ORDER MATTERS) ────────────────────────────────
# app.add_middleware(GZipMiddleware, minimum_size=1000)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],          # ✅ Allow all (fixes CORS)
#     allow_credentials=False,      # ⚠️ must be False when using "*"
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )


# # ── Routers ───────────────────────────────────────────────────
# app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"])
# app.include_router(qa.router, prefix=f"{settings.api_prefix}/qa", tags=["QA"])
# app.include_router(sessions.router, prefix=f"{settings.api_prefix}/sessions", tags=["Sessions"])


# # ── Health Check ──────────────────────────────────────────────
# @app.get(f"{settings.api_prefix}/health", tags=["Health"])
# async def health():
#     return {"status": "ok", "version": settings.app_version}


# # ── CORS PREFLIGHT HANDLER (CRITICAL FOR FILE UPLOAD) ─────────
# @app.options("/{rest_of_path:path}")
# async def preflight_handler():
#     return {"message": "OK"}
"""
api/main.py — FastAPI application entry point.

Render + Vercel deployment fixes:
  - CORS: allow_origins=["*"] so Vercel frontend can reach Render backend.
    In production you can tighten this to your exact Vercel URL via CORS_ORIGINS env var.
  - Startup: pipeline is NOT warmed up at startup (Render free tier has a 30s boot
    limit — warming models at startup causes the health check to fail and the
    service to be killed before it's ready). Models load lazily on first request.
  - /api/v1/health returns fast so Render's health check passes immediately.
"""
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
    logger.info("docsage_starting", version=settings.app_version, env=settings.environment)
    settings.create_dirs()
    # NOTE: Do NOT load models at startup on Render free tier.
    # Models load lazily on first request. Startup just needs to be fast
    # so Render's health check passes within the boot window.
    logger.info("docsage_ready")
    yield
    logger.info("docsage_shutdown")


app = FastAPI(
    title="DocSage API",
    description="Smart Document Question Answering System",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# CORS_ORIGINS env var: comma-separated list of allowed origins.
# Defaults to "*" which allows all origins — required for Render + Vercel setup
# where the exact Vercel preview URL changes on every deploy.
# To restrict: set CORS_ORIGINS=https://your-app.vercel.app in Render env vars.
_raw_origins = os.environ.get("CORS_ORIGINS", "*")
if _raw_origins == "*":
    cors_origins = ["*"]
    cors_allow_credentials = False   # credentials not allowed with wildcard origin
else:
    cors_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
    cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents.router, prefix=f"{settings.api_prefix}/documents", tags=["Documents"])
app.include_router(qa.router, prefix=f"{settings.api_prefix}/qa", tags=["QA"])
app.include_router(sessions.router, prefix=f"{settings.api_prefix}/sessions", tags=["Sessions"])


@app.get(f"{settings.api_prefix}/health", tags=["Health"])
async def health():
    """Fast health check — Render pings this every 30s to keep the service alive."""
    return {"status": "ok", "version": settings.app_version}


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint so Render's default health check path also works."""
    return {"status": "ok", "service": "DocSage API", "docs": "/api/docs"}
