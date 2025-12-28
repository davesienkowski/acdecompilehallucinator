import re
from typing import List, Dict
from .struct import Struct
from .enum import Enum


class HeaderParser:
    """Parses C++ header files and extracts struct definitions"""
    
    def __init__(self, header_file: str):
        self.header_file = header_file
        self.typedefs: List[str] = []
        self.structs: Dict[str, Struct] = {}
        self.enums: Dict[str, Enum] = {}
        
        # Statistics counters
        self.stats = {
            'structs_found': 0,
            'structs_ignored': 0,
            'enums_found': 0,
            'enums_ignored': 0,
            'typedefs_found': 0,
            'unions_found': 0,
            'unions_ignored': 0,
        }
    
    def read_file_safely(self) -> str:
        """Read file with multiple encoding attempts"""
        with open(self.header_file, 'rb') as f:
            raw = f.read()
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode('utf-8', errors='replace')
    
    def parse_typedef(self, def_line: str, lines: List[str], i: int) -> int:
        """Parse typedef definition"""
        self.typedefs.append(def_line.strip())
        self.stats['typedefs_found'] += 1
        return i
    
    def parse(self) -> None:
        """Main parsing method."""
        content = self.read_file_safely()
        lines = content.splitlines()
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Skip blank lines and forward declarations
            if re.match(r"^\s*$", line) or re.match(r"^(struct|union|enum) .*;", line) or re.match('^#', line):
                i += 1
                continue
            
            # Look for type definitions starting with comment
            if re.match(r"\/\* \d+ \*\/", line):
                i += 1
                
                # Skip pragma
                if i < len(lines) and lines[i].startswith("#pragma"):
                    i += 1
                
                if i >= len(lines):
                    break
                    
                def_line = lines[i]
                
                # Strip modifiers
                for modifier in ['const', 'volatile']:
                    if def_line.startswith(modifier + ' '):
                        def_line = def_line[len(modifier + ' '):]
                
                # Parse based on type
                if def_line.startswith("enum "):
                    e = Enum()
                    (i, enum_name, enum) = e.parse_enum(def_line, lines, i)
                    if enum:
                        self.enums[enum_name] = e
                        self.stats['enums_found'] += 1
                        if e.is_ignored:
                            self.stats['enums_ignored'] += 1
                    else:
                        print("Could not parse enum?", def_line)
                elif def_line.startswith("struct "):
                    s = Struct()
                    (i, struct_name, struct) = s.parse_struct(def_line, lines, i)
                    if struct:
                        self.structs[struct_name] = s
                        self.stats['structs_found'] += 1
                        if s.is_ignored:
                            self.stats['structs_ignored'] += 1
                    else:
                        print("Could not parse struct?", def_line)
                elif def_line.startswith("typedef "):
                    i = self.parse_typedef(def_line, lines, i)
                elif def_line.startswith("union "):
                    self.stats['unions_ignored'] += 1
                    while i < len(lines) and lines[i].strip() != "};":
                        i = i + 1
                    if i < len(lines):  # Include the closing };
                        i += 1
                else:
                    print(f"Unmatched: {def_line}")
            else:
                print("Failed to match:", line)
            i += 1
    
    def print_stats(self) -> None:
        """Print statistics about parsed types."""
        print(f"Structs found: {self.stats['structs_found'] - self.stats['structs_ignored']} (ignored: {self.stats['structs_ignored']})")
        print(f"Enums found: {self.stats['enums_found'] - self.stats['enums_ignored']} (ignored: {self.stats['enums_ignored']})")
        print(f"Typedefs found: {self.stats['typedefs_found']}")
        print(f"Unions found: {self.stats['unions_found']} (ignored: {self.stats['unions_ignored']})")