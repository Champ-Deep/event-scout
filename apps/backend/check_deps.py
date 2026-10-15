import sys

try:
    import fastapi
    print("FastAPI installed")
except ImportError:
    print("FastAPI not installed")

try:
    import uvicorn
    print("Uvicorn installed")
except ImportError:
    print("Uvicorn not installed")

try:
    import sqlalchemy
    print("SQLAlchemy installed")
except ImportError:
    print("SQLAlchemy not installed")

try:
    import asyncpg
    print("Asyncpg installed")
except ImportError:
    print("Asyncpg not installed")