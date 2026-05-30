"""RAG system for personality database"""
import sqlite3
import numpy as np
import logging
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from pathlib import Path
from src.config import Config

logger = logging.getLogger(__name__)

class PersonalityDatabase:
    """RAG system for storing and searching message examples"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or Config.database.path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        logger.info(f"Initializing embedding model: {Config.database.embedding_model}")
        self.embedding_model = SentenceTransformer(Config.database.embedding_model)
        self._setup_database()
        logger.info(f"Personality database initialized: {self.db_path}")

    def _setup_database(self):
        """Create tables if they don't exist"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS message_examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                context TEXT,
                embedding BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                chat_id INTEGER,
                user_id INTEGER
            )
        ''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON message_examples(category)')
        self.conn.commit()

        # Migrate existing database (add columns if missing)
        self._migrate_schema()
        logger.debug("Database schema initialized")

    def _migrate_schema(self):
        """Add new columns to existing database if needed"""
        cursor = self.conn.execute("PRAGMA table_info(message_examples)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'chat_id' not in columns:
            self.conn.execute('ALTER TABLE message_examples ADD COLUMN chat_id INTEGER')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_chat_id ON message_examples(chat_id)')
            logger.info("✅ Added chat_id column for personalization")

        if 'user_id' not in columns:
            self.conn.execute('ALTER TABLE message_examples ADD COLUMN user_id INTEGER')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON message_examples(user_id)')
            logger.info("✅ Added user_id column for personalization")

        self.conn.commit()

    def add_example(
        self,
        category: str,
        message: str,
        context: str = "",
        metadata: Optional[dict] = None,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None
    ):
        """
        Add message example to database

        Args:
            category: Category (joke, empathy, casual, etc.)
            message: Message text
            context: Optional context
            metadata: Additional metadata (JSON)
            chat_id: Telegram chat ID (for personalization)
            user_id: Telegram user ID of conversation partner (for personalization)
        """
        embedding = self.embedding_model.encode(message)
        embedding_blob = embedding.astype(np.float32).tobytes()

        import json
        metadata_json = json.dumps(metadata) if metadata else None

        self.conn.execute(
            '''INSERT INTO message_examples
               (category, message, context, embedding, metadata, chat_id, user_id)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (category, message, context, embedding_blob, metadata_json, chat_id, user_id)
        )
        self.conn.commit()
        logger.debug(f"Added example: category={category}, chat_id={chat_id}, user_id={user_id}, message={message[:50]}...")

    def clear_user_history(self, user_id: int) -> int:
        """
        Delete all message examples for a specific user

        Args:
            user_id: Telegram User ID

        Returns:
            Number of deleted records
        """
        cursor = self.conn.execute(
            'DELETE FROM message_examples WHERE user_id = ?',
            (user_id,)
        )
        count = cursor.rowcount
        self.conn.commit()
        logger.info(f"Cleared {count} message examples for user {user_id}")
        return count

    def find_similar(
        self,
        query: str,
        category: Optional[str] = None,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
        top_k: int = None
    ) -> List[Tuple[float, str, str]]:
        """
        Find similar examples using cosine similarity

        Prioritization:
        1. Messages from this specific user (highest priority)
        2. Messages from this chat
        3. Generic messages (chat_id and user_id are NULL)

        Args:
            query: Search query
            category: Optional category filter
            chat_id: Filter by chat ID (for personalization)
            user_id: Filter by user ID (for personalization)
            top_k: Number of results

        Returns:
            List[(similarity_score, message, context)]
        """
        if top_k is None:
            top_k = Config.database.similar_examples_count

        query_embedding = self.embedding_model.encode(query)

        # Build SQL with prioritization
        sql = '''SELECT message, context, embedding,
                    CASE
                        WHEN user_id = ? THEN 1
                        WHEN chat_id = ? THEN 2
                        WHEN user_id IS NULL AND chat_id IS NULL THEN 3
                        ELSE 4
                    END as priority
                 FROM message_examples'''
        params = [user_id, chat_id]

        conditions = []
        if category:
            conditions.append('category = ?')
            params.append(category)

        # Only show relevant messages: from this user/chat OR generic
        if user_id or chat_id:
            conditions.append('(user_id = ? OR chat_id = ? OR (user_id IS NULL AND chat_id IS NULL))')
            params.extend([user_id, chat_id])

        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)

        cursor = self.conn.execute(sql, params)

        results = []
        for row in cursor:
            message, context, embedding_blob, priority = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)

            # Cosine similarity
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )

            # Boost score for personalized messages
            if priority == 1:  # Same user
                similarity *= 1.3
            elif priority == 2:  # Same chat
                similarity *= 1.15

            results.append((float(similarity), message, context or ""))

        results.sort(reverse=True, key=lambda x: x[0])
        # Log personalization stats
        personalized = sum(1 for score, _, _ in results[:top_k] if score > 0.8)  # High similarity = likely personalized
        if personalized > 0:
            logger.debug(f"Found {personalized}/{len(results[:top_k])} highly personalized examples (user_id={user_id}, chat_id={chat_id})")

        return results[:top_k]

    def get_category_stats(self) -> dict:
        """Get statistics by category"""
        cursor = self.conn.execute(
            'SELECT category, COUNT(*) FROM message_examples GROUP BY category'
        )
        stats = dict(cursor.fetchall())
        logger.info(f"Database stats: {stats}")
        return stats

    def get_total_count(self) -> int:
        """Get total number of examples"""
        cursor = self.conn.execute('SELECT COUNT(*) FROM message_examples')
        return cursor.fetchone()[0]

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Database connection closed")
