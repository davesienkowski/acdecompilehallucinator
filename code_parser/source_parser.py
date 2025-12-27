import re
from typing import List, Dict
from .struct import Struct
from .enum import Enum
from .method import Method
from .offset_mapper import OffsetMapper

class SourceParser:
    """Parses C++ source files and extracts globals / methods"""
    
    def __init__(self, source_file: str, offset_mapper: OffsetMapper):
        self.source_file = source_file
        self.methods: Dict[str, Method] = {}
        self.offset_mapper: OffsetMapper = offset_mapper
        # Statistics counters
        self.stats = {
            'global_methods_found': 0,
            'global_methods_ignored': 0,
            'class_methods_found': 0,
            'class_methods_ignored': 0,
        }
    
    def read_file_safely(self) -> str:
        """Read file with multiple encoding attempts"""
        with open(self.source_file, 'rb') as f:
            raw = f.read()
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode('utf-8', errors='replace')
    
    def parse(self):
        """Main parsing method"""
        content = self.read_file_safely()
        lines = content.splitlines()
        i = 0
        found_funcs = 0

        while i < len(lines):
            line = lines[i]
            
            # Skip blank lines
            if re.match(r"^\s*$", line):
                i += 1
                continue

            if line.startswith('//----- '):
                m = Method()
                (i, func_name, func) = m.parse(line, lines, i)
                if func:
                    m.file = self.offset_mapper.get_filename("0x" + func.offset)
                    self.methods[func.full_name] = func
                    found_funcs = found_funcs + 1
                    if func.is_global:
                        self.stats['global_methods_found'] = self.stats['global_methods_found'] + 1
                        if func.is_ignored:
                            self.stats['global_methods_ignored'] = self.stats['global_methods_ignored'] + 1

                    else:
                        self.stats['class_methods_found'] = self.stats['class_methods_found'] + 1
                        if func.is_ignored:
                            self.stats['class_methods_ignored'] = self.stats['class_methods_ignored'] + 1
            else:
                pass
            
            i += 1
        print("Found", found_funcs, "funcs")
    
    def print_stats(self):
        """Print statistics about parsed types"""
        print(f"Global Methods found: {self.stats['global_methods_found'] - self.stats['global_methods_ignored']} (ignored: {self.stats['global_methods_ignored']})")
        print(f"Class Methods found: {self.stats['class_methods_found'] - self.stats['class_methods_ignored']} (ignored: {self.stats['class_methods_ignored']})")