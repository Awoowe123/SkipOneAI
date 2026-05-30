"""Contacts manager for storing and retrieving information about people"""
import sqlite3
import logging
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ContactsManager:
    """Manage contacts database with bio and metadata"""

    def __init__(self, db_path: str = "data/contacts.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._setup_database()
        self._migrate_add_style_bio()  # Add missing column if needed
        logger.info(f"Contacts manager initialized: {db_path}")

    def _setup_database(self):
        """Create contacts table"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS contacts (
                user_id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                username TEXT,
                bio TEXT,
                style_bio TEXT,
                bio_analyzed BOOLEAN DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                last_message TEXT,
                last_seen DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_username ON contacts(username)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_first_name ON contacts(first_name)')
        self.conn.commit()
        logger.debug("Contacts database schema initialized")

    def get_or_create(
        self,
        user_id: int,
        first_name: str = None,
        last_name: str = None,
        username: str = None
    ) -> Dict:
        """
        Get existing contact or create new one

        Args:
            user_id: Telegram user ID
            first_name: User's first name
            last_name: User's last name
            username: User's @username

        Returns:
            Contact dictionary
        """
        cursor = self.conn.execute(
            'SELECT * FROM contacts WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()

        if row:
            # Update metadata if changed
            self.conn.execute('''
                UPDATE contacts
                SET first_name = COALESCE(?, first_name),
                    last_name = COALESCE(?, last_name),
                    username = COALESCE(?, username),
                    last_seen = ?
                WHERE user_id = ?
            ''', (first_name, last_name, username, datetime.now(), user_id))
            self.conn.commit()

            return self._row_to_dict(row)
        else:
            # Create new contact
            self.conn.execute('''
                INSERT INTO contacts (user_id, first_name, last_name, username, last_seen)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, first_name, last_name, username, datetime.now()))
            self.conn.commit()

            logger.info(f"Created new contact: {first_name} (@{username}, ID: {user_id})")

            return {
                'user_id': user_id,
                'first_name': first_name,
                'last_name': last_name,
                'username': username,
                'bio': None,
                'last_message': None,
                'last_seen': datetime.now().isoformat()
            }

    def update_bio(self, user_id: int, bio: str):
        """Update bio for a contact"""
        self.conn.execute(
            'UPDATE contacts SET bio = ?, bio_analyzed = 1 WHERE user_id = ?',
            (bio, user_id)
        )
        self.conn.commit()
        logger.info(f"Updated bio for user {user_id}")

    def clear_bio(self, user_id: int):
        """Clear bio for a specific user"""
        self.conn.execute('UPDATE contacts SET bio = NULL, bio_analyzed = 0 WHERE user_id = ?', (user_id,))
        self.conn.commit()
        logger.info(f"Cleared bio for user {user_id}")

    def get_style_bio(self, user_id: int) -> Optional[str]:
        """Get style bio for user (typically bot owner)"""
        cursor = self.conn.execute(
            'SELECT style_bio FROM contacts WHERE user_id = ?',
            (user_id,)
        )
        result = cursor.fetchone()
        return result[0] if result and result[0] else None

    def set_style_bio(self, user_id: int, style_bio: str):
        """Set style bio for user"""
        # Ensure contact exists first
        self.get_or_create(user_id=user_id)

        self.conn.execute(
            'UPDATE contacts SET style_bio = ? WHERE user_id = ?',
            (style_bio, user_id)
        )
        self.conn.commit()
        logger.info(f"Updated style bio for user {user_id}")

    def clear_style_bio(self, user_id: int):
        """Clear style bio"""
        self.conn.execute(
            'UPDATE contacts SET style_bio = NULL WHERE user_id = ?',
            (user_id,)
        )
        self.conn.commit()

    def _migrate_add_style_bio(self):
        """Migration: Add style_bio column to existing databases"""
        try:
            # Check if column exists
            cursor = self.conn.execute("PRAGMA table_info(contacts)")
            columns = [row[1] for row in cursor.fetchall()]

            if 'style_bio' not in columns:
                logger.info("🔄 Migration: Adding style_bio column...")
                self.conn.execute('ALTER TABLE contacts ADD COLUMN style_bio TEXT')
                self.conn.commit()
                logger.info("✅ Migration complete: style_bio added")
        except Exception as e:
            # If error, column probably already exists
            logger.debug(f"Migration check: {e}")



    def update_last_message(self, user_id: int, message: str):
        """Update last message from contact and increment message counter"""
        self.conn.execute(
            '''UPDATE contacts
               SET last_message = ?,
                   last_seen = ?,
                   message_count = message_count + 1
               WHERE user_id = ?''',
            (message[:200], datetime.now(), user_id)
        )
        self.conn.commit()

    def find_by_name(self, name: str) -> Optional[Dict]:
        """
        Find contact by name (fuzzy search)

        Args:
            name: Name to search for (first_name, last_name, or username)

        Returns:
            Contact dict or None
        """
        name_lower = name.lower().strip('@')

        # Try exact username match first
        cursor = self.conn.execute(
            'SELECT * FROM contacts WHERE LOWER(username) = ?',
            (name_lower,)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)

        # Try first_name match
        cursor = self.conn.execute(
            'SELECT * FROM contacts WHERE LOWER(first_name) LIKE ?',
            (f'%{name_lower}%',)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)

        # Try last_name match
        cursor = self.conn.execute(
            'SELECT * FROM contacts WHERE LOWER(last_name) LIKE ?',
            (f'%{name_lower}%',)
        )
        row = cursor.fetchone()
        if row:
            return self._row_to_dict(row)

        logger.debug(f"Contact not found: {name}")
        return None

    def get_contact(self, user_id: int) -> Optional[Dict]:
        """Get contact by user_id"""
        cursor = self.conn.execute(
            'SELECT * FROM contacts WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def get_all_contacts(self) -> List[Dict]:
        """Get all contacts"""
        cursor = self.conn.execute(
            'SELECT * FROM contacts ORDER BY last_seen DESC'
        )
        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_bio(self, user_id: int) -> Optional[str]:
        """Get bio for a contact"""
        cursor = self.conn.execute(
            'SELECT bio FROM contacts WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        return row[0] if row and row[0] else None

    def get_message_count(self, user_id: int) -> int:
        """Get number of messages exchanged with contact"""
        cursor = self.conn.execute(
            'SELECT message_count FROM contacts WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def mark_bio_analyzed(self, user_id: int):
        """Mark bio as analyzed"""
        self.conn.execute(
            'UPDATE contacts SET bio_analyzed = 1 WHERE user_id = ?',
            (user_id,)
        )
        self.conn.commit()

    def needs_bio_analysis(self, user_id: int, threshold: int = 50) -> bool:
        """Check if contact needs bio analysis"""
        cursor = self.conn.execute(
            'SELECT message_count, bio_analyzed FROM contacts WHERE user_id = ?',
            (user_id,)
        )
        row = cursor.fetchone()
        if not row:
            return False

        message_count, bio_analyzed = row
        return message_count >= threshold and not bio_analyzed

    def _row_to_dict(self, row) -> Dict:
        """Convert database row to dictionary"""
        return {
            'user_id': row[0],
            'first_name': row[1],
            'last_name': row[2],
            'username': row[3],
            'bio': row[4],
            'bio_analyzed': row[5] if len(row) > 5 else False,
            'message_count': row[6] if len(row) > 6 else 0,
            'last_message': row[7] if len(row) > 7 else None,
            'last_seen': row[8] if len(row) > 8 else None,
            'created_at': row[9] if len(row) > 9 else None
        }

    def close(self):
        """Close database connection"""
        self.conn.close()
        logger.info("Contacts database closed")
