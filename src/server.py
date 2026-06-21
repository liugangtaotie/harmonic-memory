"""Harmonic Memory Server — FastAPI application entry point."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from .config import load_config
from .db.sqlite import MemoryDB
from .db.qdrant_client import MemoryQdrant
from .api import routes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # Startup
    logger.info("Starting Harmonic Memory server...")

    # Initialize database
    db = MemoryDB()
    db.init_schema()
    routes.db = db
    logger.info(f"SQLite initialized at {db.db_path}")

    # Initialize Qdrant
    qdrant = MemoryQdrant()
    try:
        qdrant.ensure_collection()
        routes.qdrant = qdrant
        info = qdrant.get_collection_info()
        logger.info(f"Qdrant connected: {info}")
    except Exception as e:
        logger.warning(f"Qdrant not available: {e}")
        routes.qdrant = None

    logger.info("Harmonic Memory server ready")
    yield

    # Shutdown
    logger.info("Shutting down...")
    if routes.db:
        routes.db.close()
    logger.info("Harmonic Memory server stopped")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Harmonic Memory",
        description="World-class AI memory system — unified, LLM-powered, real-time",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins for mobile access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(routes.router)

    # Dashboard route
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Serve the web dashboard."""
        dash_path = Path(__file__).parent / "dashboard.html"
        if dash_path.exists():
            return dash_path.read_text(encoding="utf-8")
        return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)

    # Root redirect to dashboard
    @app.get("/")
    async def root():
        dash_path = Path(__file__).parent / "dashboard.html"
        if dash_path.exists():
            content = dash_path.read_text(encoding="utf-8")
            return HTMLResponse(content=content, media_type="text/html; charset=utf-8")
        return {
            "name": "Harmonic Memory",
            "version": "0.1.0",
            "docs": "/docs",
            "dashboard": "/dashboard",
            "health": "/api/v1/health",
        }

    return app


def main():
    """CLI entry point."""
    import uvicorn

    config = load_config()
    log_level = config.logging.level.lower()

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    uvicorn.run(
        "src.server:create_app",
        host=config.server.host,
        port=config.server.port,
        log_level=log_level,
        reload=False,
        factory=True,
    )


# Module-level app for uvicorn
app = create_app()

if __name__ == "__main__":
    main()
