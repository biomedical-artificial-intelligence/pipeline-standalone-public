import aiosqlite
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "pipeline.db")


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS pipeline (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pdb_name    TEXT,
                length      INTEGER,
                max_iter    INTEGER,
                alpha       REAL,
                batch_size  INTEGER,
                status      TEXT DEFAULT 'working',
                created_at  TEXT,
                finished_at TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS result (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_id INTEGER,
                iter        INTEGER,
                sequence    TEXT,
                score       REAL,
                ranking     INTEGER,
                created_at  TEXT
            )
        """)
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_result_pipeline_seq
            ON result (pipeline_id, sequence)
        """)
        await db.commit()
    logger.info(f"DB initialized: {DB_PATH}")


async def create_pipeline(
    pdb_name: str,
    length: int,
    max_iter: int,
    alpha: float,
    batch_size: int,
) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            """INSERT INTO pipeline (pdb_name, length, max_iter, alpha, batch_size, status, created_at)
               VALUES (?, ?, ?, ?, ?, 'working', ?)""",
            (pdb_name, length, max_iter, alpha, batch_size, datetime.now().isoformat()),
        )
        await db.commit()
        return cursor.lastrowid


async def finish_pipeline(pipeline_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE pipeline SET status='finished', finished_at=? WHERE id=?",
            (datetime.now().isoformat(), pipeline_id),
        )
        await db.commit()


async def create_result(pipeline_id: int, iter: int, sequence: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO result (pipeline_id, iter, sequence, created_at)
               VALUES (?, ?, ?, ?)""",
            (pipeline_id, iter, sequence, datetime.now().isoformat()),
        )
        await db.commit()


async def update_score(pipeline_id: int, iter: int, score: float, ranking: Optional[int]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """UPDATE result SET score=?, ranking=?
               WHERE pipeline_id=? AND iter=?""",
            (score, ranking, pipeline_id, iter),
        )
        await db.commit()


async def get_score_by_sequence(pipeline_id: int, sequence: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(
            """SELECT score, ranking, iter FROM result
               WHERE pipeline_id=? AND sequence=? AND score IS NOT NULL
               LIMIT 1""",
            (pipeline_id, sequence),
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {"score": row[0], "ranking": row[1], "iter": row[2]}
            return None
