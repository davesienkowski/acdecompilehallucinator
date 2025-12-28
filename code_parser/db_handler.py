import sqlite3
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from .enum import Enum
from .struct import Struct
from .method import Method


class DatabaseHandler:
    """Handles SQLite database operations for storing enums, structs, and processed output"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def init_db(self) -> None:
        """Initialize the database with the required table structure."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            cursor = conn.cursor()
            
            # Create the types table (raw parsed types)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    namespace TEXT,
                    parent TEXT,
                    code TEXT NOT NULL,
                    fields TEXT,
                    is_ignored BOOLEAN DEFAULT 0,
                    comment TEXT,
                    is_generic BOOLEAN DEFAULT 0,
                    vtable_code TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create the methods table (raw parsed methods)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS methods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    namespace TEXT,
                    parent TEXT,
                    is_generic BOOLEAN DEFAULT 0,
                    is_ignored BOOLEAN DEFAULT 0,
                    offset TEXT,
                    return_type TEXT,
                    is_global BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create processed_types table (LLM-processed types)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_types (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    type TEXT NOT NULL,
                    original_code TEXT,
                    processed_header TEXT,
                    processed_source TEXT,
                    dependencies TEXT,
                    is_processed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            ''')
            
            # Create processed_methods table (LLM-processed methods)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processed_methods (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    full_name TEXT NOT NULL UNIQUE,
                    parent_class TEXT,
                    original_code TEXT,
                    processed_code TEXT,
                    dependencies TEXT,
                    offset TEXT,
                    is_processed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP
                )
            ''')
            
            # Migration: Add offset column if it doesn't exist
            try:
                cursor.execute("ALTER TABLE processed_methods ADD COLUMN offset TEXT")
            except sqlite3.OperationalError:
                pass  # Already exists
            
            # Migration: Add engine_used column to processed_types (defaults to lm-studio for legacy)
            try:
                cursor.execute("ALTER TABLE processed_types ADD COLUMN engine_used TEXT DEFAULT 'lm-studio'")
            except sqlite3.OperationalError:
                pass  # Already exists
            
            # Migration: Add engine_used column to processed_methods (defaults to lm-studio for legacy)
            try:
                cursor.execute("ALTER TABLE processed_methods ADD COLUMN engine_used TEXT DEFAULT 'lm-studio'")
            except sqlite3.OperationalError:
                pass  # Already exists
            
            # Create constants table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS constants (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type_id TEXT,
                    value TEXT NOT NULL,
                    name TEXT NOT NULL,
                    is_ldata BOOLEAN DEFAULT 0,
                    address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(value, name)
                )
            ''')

            # Create debug_types table (from acclient.txt)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS debug_types (
                    id TEXT PRIMARY KEY,
                    length INTEGER,
                    leaf_type TEXT,
                    description TEXT,
                    raw_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for faster queries on types table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_type_name ON types(type, name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_namespace ON types(namespace)')
            
            # Create indexes for faster queries on methods table
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_method_name ON methods(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_method_full_name ON methods(full_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_method_namespace ON methods(namespace)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_method_parent ON methods(parent)')
            
            # Create indexes for processed tables
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_type_name ON processed_types(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_type_is_processed ON processed_types(is_processed)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_method_name ON processed_methods(full_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_method_parent ON processed_methods(parent_class)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed_method_is_processed ON processed_methods(is_processed)')
            
            conn.commit()
    
    def store_enum(self, enum: Enum) -> None:
        """Store an enum in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO types (type, name, namespace, code, is_ignored, comment)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                'enum',
                enum.full_name,
                enum.namespace,
                enum.definition,
                enum.is_ignored,
                enum.comment
            ))
            
            conn.commit()
    
    def store_struct(self, struct: Struct, vtable_code: Optional[str] = None) -> None:
        """Store a struct in the database, with optional vtable code."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO types (type, name, namespace, parent, code, 
                                 is_ignored, comment, is_generic, vtable_code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                'struct',
                struct.full_name,
                struct.namespace,
                struct.parent,
                struct.definition,
                struct.is_ignored,
                struct.comment,
                struct.is_generic,
                vtable_code
            ))
            
            conn.commit()
    
    def store_method(self, method: Method) -> None:
        """Store a method in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Determine if method is global
            is_global = 1 if "::" not in method.full_name else 0
            
            cursor.execute('''
                INSERT INTO methods (name, full_name, definition, namespace, parent,
                                   is_generic, is_ignored, offset, return_type, is_global)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                method.name,
                method.full_name,
                method.definition,
                method.namespace,
                method.parent,
                method.is_generic,
                method.is_ignored,
                method.offset,
                method.return_type,
                is_global
            ))
            
            conn.commit()

    def get_all_types(self) -> List[Tuple]:
        """Retrieve all stored types from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM types ORDER BY type, name')
            return cursor.fetchall()
    
    def get_enums(self) -> List[Tuple]:
        """Retrieve all stored enums from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM types WHERE type = "enum" ORDER BY name')
            return cursor.fetchall()
    
    def get_structs(self) -> List[Tuple]:
        """Retrieve all stored structs from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM types WHERE type = "struct" ORDER BY name')
            return cursor.fetchall()
    
    def get_methods(self) -> List[Tuple]:
        """Retrieve all stored methods from the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM methods ORDER BY name')
            return cursor.fetchall()
    
    def get_type_by_name(self, name: str, type_filter: Optional[str] = None):
        """Retrieve a specific type by name, optionally filtered by type"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if type_filter:
                cursor.execute('SELECT * FROM types WHERE name = ? AND type = ? ORDER BY type', (name, type_filter))
            else:
                cursor.execute('SELECT * FROM types WHERE name = ? ORDER BY type', (name,))
            
            return cursor.fetchall()
    
    def get_method_by_name(self, name: str) -> List[Tuple]:
        """Retrieve a specific method by name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM methods WHERE name = ? ORDER BY name', (name,))
            return cursor.fetchall()
    
    def get_method_by_full_name(self, full_name: str) -> List[Tuple]:
        """Retrieve a specific method by full name."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM methods WHERE full_name = ? ORDER BY name', (full_name,))
            return cursor.fetchall()

    def get_methods_by_parent(self, parent: str) -> List[Tuple]:
        """Retrieve all methods belonging to a specific parent class"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if "::" in parent:
                parts = parent.split("::")
                simple_parent = parts[-1]
                namespace = "::".join(parts[:-1])
                cursor.execute('SELECT * FROM methods WHERE parent = ? AND namespace = ? AND is_ignored = 0 ORDER BY name', 
                             (simple_parent, namespace))
            else:
                cursor.execute('SELECT * FROM methods WHERE parent = ? AND (namespace IS NULL OR namespace = "") AND is_ignored = 0 ORDER BY name', 
                             (parent,))
            
            return cursor.fetchall()

    def get_nested_types(self, parent_name: str) -> List[Tuple]:
        """Retrieve all types nested within a specific parent class/namespace"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Find types where namespace matches parent_name
            cursor.execute('SELECT * FROM types WHERE namespace = ? AND is_ignored = 0 ORDER BY name', (parent_name,))
            return cursor.fetchall()

    def get_all_parent_classes(self) -> List[str]:
        """Get all unique parent class names from methods"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT parent FROM methods WHERE parent IS NOT NULL AND is_ignored = 0 ORDER BY parent')
            return [row[0] for row in cursor.fetchall()]

    # ───────────────────────────────────────────────────────────────────────
    # Processed Types Operations
    # ───────────────────────────────────────────────────────────────────────
    
    def store_processed_type(self, name: str, type_kind: str, original_code: str,
                             processed_header: str, processed_source: str = None,
                             dependencies: List[str] = None,
                             engine_used: str = "lm-studio"):
        """Store a processed type in the database.
        
        Args:
            name: Fully qualified type name
            type_kind: Type kind (struct, enum, etc.)
            original_code: Original decompiled code
            processed_header: Modernized header code
            processed_source: Modernized source code (optional)
            dependencies: List of type dependencies
            engine_used: Name of LLM engine that processed this type
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            deps_json = json.dumps(dependencies) if dependencies else None
            
            cursor.execute('''
                INSERT OR REPLACE INTO processed_types 
                (name, type, original_code, processed_header, processed_source, 
                 dependencies, is_processed, processed_at, engine_used)
                VALUES (?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?)
            ''', (name, type_kind, original_code, processed_header, 
                  processed_source, deps_json, engine_used))
            conn.commit()

    def get_processed_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a processed type by name"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM processed_types WHERE name = ?', (name,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('dependencies'):
                    result['dependencies'] = json.loads(result['dependencies'])
                return result
            return None

    def get_type_with_fallback(self, name: str) -> Tuple[Optional[Dict], bool]:
        """
        Get type definition, checking processed first, then raw.
        Returns (type_dict, is_processed) tuple.
        """
        # Check processed first
        processed = self.get_processed_type(name)
        if processed and processed.get('is_processed'):
            return processed, True
        
        # Fall back to raw types
        raw = self.get_type_by_name(name)
        if raw:
            return {'name': name, 'code': raw[0][5] if len(raw[0]) > 5 else None}, False
        
        return None, False

    def get_unprocessed_types(self, type_filter: Optional[str] = None) -> List[Tuple]:
        """Get all types that haven't been processed yet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if type_filter:
                cursor.execute('''
                    SELECT t.* FROM types t
                    LEFT JOIN processed_types p ON t.name = p.name
                    WHERE (p.id IS NULL OR p.is_processed = 0)
                    AND t.type = ? AND t.is_ignored = 0
                    ORDER BY t.name
                ''', (type_filter,))
            else:
                cursor.execute('''
                    SELECT t.* FROM types t
                    LEFT JOIN processed_types p ON t.name = p.name
                    WHERE (p.id IS NULL OR p.is_processed = 0)
                    AND t.is_ignored = 0
                    ORDER BY t.name
                ''')
            return cursor.fetchall()

    def is_type_processed(self, name: str) -> bool:
        """Check if a type has been processed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT is_processed FROM processed_types WHERE name = ?', (name,))
            row = cursor.fetchone()
            return bool(row and row[0])

    # ───────────────────────────────────────────────────────────────────────
    # Processed Methods Operations
    # ───────────────────────────────────────────────────────────────────────
    
    def store_processed_method(self, method: Method, processed_code: str,
                               dependencies: List[str] = None,
                               engine_used: str = "lm-studio"):
        """Store a processed method in the database.
        
        Args:
            method: Method dataclass with metadata
            processed_code: Modernized method code
            dependencies: List of type dependencies
            engine_used: Name of LLM engine that processed this method
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            deps_json = json.dumps(dependencies) if dependencies else None
            
            # Use fully qualified parent name for tracking
            parent_name = method.parent
            if method.namespace:
                parent_name = f"{method.namespace}::{method.parent}"

            cursor.execute('''
                INSERT OR REPLACE INTO processed_methods 
                (name, full_name, parent_class, original_code, processed_code,
                 dependencies, offset, is_processed, processed_at, engine_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?)
            ''', (method.name, method.full_name, parent_name,
                  method.definition, processed_code, deps_json, method.offset, engine_used))
            conn.commit()

    def get_processed_method(self, full_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a processed method by full name"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM processed_methods WHERE full_name = ?', (full_name,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                if result.get('dependencies'):
                    result['dependencies'] = json.loads(result['dependencies'])
                return result
            return None

    def get_processed_methods_by_parent(self, parent_class: str) -> List[Dict[str, Any]]:
        """Get all processed methods for a specific parent class"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # parent_class in processed_methods is now the fully qualified name
            cursor.execute('''
                SELECT * FROM processed_methods 
                WHERE parent_class = ? AND is_processed = 1
                ORDER BY name
            ''', (parent_class,))
            return [dict(row) for row in cursor.fetchall()]

    def get_unprocessed_methods(self, parent_class: Optional[str] = None) -> List[Tuple]:
        """Get all methods that haven't been processed yet"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if parent_class:
                if "::" in parent_class:
                    parts = parent_class.split("::")
                    simple_parent = parts[-1]
                    namespace = "::".join(parts[:-1])
                    cursor.execute('''
                        SELECT m.* FROM methods m
                        LEFT JOIN processed_methods p ON m.full_name = p.full_name
                        WHERE (p.id IS NULL OR p.is_processed = 0)
                        AND m.parent = ? AND m.namespace = ? AND m.is_ignored = 0
                        ORDER BY m.name
                    ''', (simple_parent, namespace))
                else:
                    cursor.execute('''
                        SELECT m.* FROM methods m
                        LEFT JOIN processed_methods p ON m.full_name = p.full_name
                        WHERE (p.id IS NULL OR p.is_processed = 0)
                        AND m.parent = ? AND (m.namespace IS NULL OR m.namespace = "") AND m.is_ignored = 0
                        ORDER BY m.name
                    ''', (parent_class,))
            else:
                cursor.execute('''
                    SELECT m.* FROM methods m
                    LEFT JOIN processed_methods p ON m.full_name = p.full_name
                    WHERE (p.id IS NULL OR p.is_processed = 0)
                    AND m.is_ignored = 0
                    ORDER BY m.parent, m.name
                ''')
            return cursor.fetchall()

    def is_method_processed(self, full_name: str) -> bool:
        """Check if a method has been processed"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT is_processed FROM processed_methods WHERE full_name = ?', (full_name,))
            row = cursor.fetchone()
            return bool(row and row[0])

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processing progress"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Types stats
            cursor.execute('SELECT COUNT(*) FROM types WHERE is_ignored = 0')
            total_types = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM processed_types WHERE is_processed = 1')
            processed_types = cursor.fetchone()[0]
            
            # Methods stats
            cursor.execute('SELECT COUNT(*) FROM methods WHERE is_ignored = 0')
            total_methods = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM processed_methods WHERE is_processed = 1')
            processed_methods = cursor.fetchone()[0]
            
            return {
                'total_types': total_types,
                'processed_types': processed_types,
                'total_methods': total_methods,
                'processed_methods': processed_methods,
                'types_progress': f"{processed_types}/{total_types}",
                'methods_progress': f"{processed_methods}/{total_methods}"
            }

    def clear_processed_class(self, class_name: str):
        """
        Clear processed status for a class and its methods.
        Used when --force is specified.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Clear processed type
            cursor.execute('''
                DELETE FROM processed_types WHERE name = ?
            ''', (class_name,))
            
            # Clear processed methods using fully qualified parent name
            cursor.execute('''
                DELETE FROM processed_methods WHERE parent_class = ?
            ''', (class_name,))
            
            conn.commit()

    # ───────────────────────────────────────────────────────────────────────
    # Constants & Debug Types Operations
    # ───────────────────────────────────────────────────────────────────────

    def store_constant(self, name: str, value: str, type_id: str = None, 
                      is_ldata: bool = False, address: str = None):
        """Store a constant in the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO constants 
                (name, value, type_id, is_ldata, address)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, value, type_id, is_ldata, address))
            conn.commit()

    def store_debug_type(self, type_id: str, length: int, leaf_type: str, 
                        description: str, raw_data: str = None):
        """Store a debug type definition"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO debug_types 
                (id, length, leaf_type, description, raw_data)
                VALUES (?, ?, ?, ?, ?)
            ''', (type_id, length, leaf_type, description, raw_data))
            conn.commit()

    def get_constants(self) -> List[Dict[str, Any]]:
        """Get all constants"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM constants ORDER BY name')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_constant_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get constant by name"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM constants WHERE name = ?', (name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_debug_type(self, type_id: str) -> Optional[Dict[str, Any]]:
        """Get debug type by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM debug_types WHERE id = ?', (type_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
