import os
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from pgvector.psycopg2 import register_vector

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:devpass@localhost:5432/opsmind")


def get_connection():
    """Open a fresh connection with pgvector's Python adapter registered,
    so we can pass/receive Python lists as VECTOR columns directly."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    register_vector(conn)
    return conn
