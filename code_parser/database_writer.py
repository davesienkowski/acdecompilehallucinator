import shutil
from pathlib import Path
from typing import List, Dict
from .struct import Struct
from .enum import Enum
from .db_handler import DatabaseHandler


class DatabaseWriter:
    """Writes parsed types to an SQLite database"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.db_path = self.output_path / 'types.db'
        self.db_handler = DatabaseHandler(self.db_path)
        
    def setup_directories(self):
        """Create output directories"""
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        self.output_path.mkdir(parents=True)
    
    def write_typedefs(self, typedefs: List[str]):
        """Write typedefs to file (keeping existing functionality)"""
        with open(self.output_path / 'typedefs.txt', 'w') as f:
            for typedef in sorted(typedefs):
                f.write(f"{typedef}\n")
    
    def write_enums(self, enums: Dict[str, Enum], structs: Dict[str, Struct] = None):
        """Write all enums to database"""
        for name in sorted(enums.keys()):
            enum = enums[name]
            if not enum.comment:
                continue
            # Store enum in database
            self.db_handler.store_enum(enum)
    
    def write_structs(self, structs: Dict[str, Struct]):
        """Write all structs to database, handling vtables when available"""
        for name in sorted(structs.keys()):
            struct = structs[name]
            if not struct.comment:
                continue
            
            # Check if there's a corresponding vtable for this struct
            vtable_code = None
            # Look for vtable with the same name pattern
            for other_name, other_struct in structs.items():
                if other_name == f"{name}_vtbl" or other_name == f"{name}Vtbl" or "_vtbl" in other_name:
                    vtable_code = other_struct.definition
                    break
            
            # Store struct in database with vtable code if available
            self.db_handler.store_struct(struct, vtable_code)
    
    def write_all_to_database(self, enums: Dict[str, Enum], structs: Dict[str, Struct], typedefs: List[str] = None):
        """Write all parsed types to the database"""
        # Write typedefs to file (keeping existing functionality)
        if typedefs:
            self.write_typedefs(typedefs)
        
        # Write enums to database
        self.write_enums(enums, structs)
        
        # Write structs to database
        self.write_structs(structs)