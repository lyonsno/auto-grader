"""Database entrypoints for schema initialization and connections.

The initial schema is intentionally not implemented yet. The test suite defines the
contract we want the first database slice to satisfy.
"""

from __future__ import annotations

import sqlite3


def create_connection(path: str = ":memory:") -> sqlite3.Connection:
    """Create a SQLite connection with foreign key enforcement enabled."""

    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_schema(connection: sqlite3.Connection) -> None:
    """Initialize the project schema.

    The first implementation will land after the fail-first tests are reviewed.
    """

    del connection

