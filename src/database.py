
import sqlite3
import json

from pathlib import Path
from datetime import datetime
from filesystem import get_now_timestamp

# Database path
# The path is relative to the project root, not the current terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "curious_ai.db"


def get_connection():
    # Check that the database already exists.
    # sqlite3.connect() would otherwise silently create a new database.
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Return database rows as dictionaries-like objects.
    conn.row_factory = sqlite3.Row

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    return conn


def init_db():
    """Create the observations table if it does not exist."""

    conn = get_connection()

    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (

                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,
                image_path TEXT,

                camera_settings TEXT,
                analysis TEXT,

                analysis_version TEXT,

                status TEXT NOT NULL DEFAULT 'ok'
                    CHECK (status IN ('ok', 'failed', 'missed')),

                error TEXT
            )
        """)

        # An index makes chronological queries faster.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_timestamp
            ON observations(timestamp)
        """)

        conn.commit()

    finally:
        conn.close()

    print(f"Database initialized: {DB_PATH}")


def save_observation(
    image_path,
    camera_settings,
    analysis=None,
    analysis_version=None,
    status="ok",
    error=None,
    timestamp=None
):
    """Save one observation and return its ID."""

    if status not in ("ok", "failed", "missed"):
        raise ValueError(f"Invalid status: {status}")

    # Generate a local timestamp if none was provided.
    if timestamp is None:
        timestamp = get_now_timestamp()

    # Convert Python dictionaries into JSON strings for SQLite.
    camera_json = json.dumps(camera_settings) if camera_settings is not None else None
    analysis_json = json.dumps(analysis) if analysis is not None else None

    conn = get_connection()

    try:
        cursor = conn.execute("""
            INSERT INTO observations (
                timestamp,
                image_path,
                camera_settings,
                analysis,
                analysis_version,
                status,
                error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp,
            image_path,
            camera_json,
            analysis_json,
            analysis_version,
            status,
            error
        ))

        conn.commit()

        # SQLite generates the ID automatically.
        observation_id = cursor.lastrowid

    finally:
        conn.close()

    # Retrieve the complete saved observation.
    observation = get_observation(observation_id)
    
    print ("saved observation: ", observation)

    return observation


def get_observation(observation_id):
    """Retrieve one observation by its ID."""

    conn = get_connection()

    try:
        row = conn.execute("""
            SELECT *
            FROM observations
            WHERE observation_id = ?
        """, (observation_id,)).fetchone()

    finally:
        conn.close()

    if row is None:
        return None

    observation = dict(row)

    # Convert stored JSON strings back into Python dictionaries.
    if observation["camera_settings"] is not None:
        observation["camera_settings"] = json.loads(
            observation["camera_settings"]
        )

    if observation["analysis"] is not None:
        observation["analysis"] = json.loads(
            observation["analysis"]
        )

    return observation


if __name__ == "__main__":
    init_db()