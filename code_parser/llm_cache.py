import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("llm-cache")

class LLMCache:
    """Handles a separate SQLite database for caching LLM prompts and responses."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self):
        """Initialize the cache database with the logic table structure."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS llm_cache (
                    prompt TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def get(self, prompt: str) -> Optional[str]:
        """Retrieve a cached response for the given prompt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT response FROM llm_cache WHERE prompt = ?', (prompt,))
                row = cursor.fetchone()
                return row[0] if row else None
        except sqlite3.Error as e:
            logger.error(f"Cache read error: {e}")
            return None
            
    def set(self, prompt: str, response: str):
        """Store a prompt and its response in the cache."""
        if not response:
            return
            
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO llm_cache (prompt, response)
                    VALUES (?, ?)
                ''', (prompt, response))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Cache write error: {e}")
