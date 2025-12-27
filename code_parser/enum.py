from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
from .constants import should_ignore_class


@dataclass
class Enum:
    """Represents a parsed enum with its metadata"""
    name: str = ""
    definition: str = ""
    namespace: Optional[str] = None
    is_ignored: bool = False
    comment: str = ""
    
    @property
    def full_name(self) -> str:
        """Returns the fully qualified name"""
        if self.namespace:
            return f"{self.namespace}::{self.name}"
        return self.name
    
    @property
    def safe_name(self) -> str:
        """Returns a filesystem-safe name"""
        return self.full_name.replace('::', '__')
    
    @property
    def simple_name(self) -> str:
        """Returns just the enum name without namespace"""
        return self.name
    
    def get_comment_header(self) -> str:
        """Generate appropriate comment header for the enum"""
        comment = f"// Enum {self.simple_name} -- "
        if self.namespace:
            comment = comment + f"{self.namespace}::"
        comment = comment + f"{self.name} {self.comment}\n"
        return comment

    def parse_namespace(self, full_name: str) -> tuple[Optional[str], str]:
        """Split full name into namespace and simple name"""
        parts = full_name.split("::")
        if len(parts) == 1:
            return None, full_name
        return  parts[0], parts[-1]

    def write_to_file(self, enums_path: str, structs: Dict[str, any] = None):
        """Write a single enum to its appropriate file"""
        # If structs are provided and the enum's namespace has a struct defined,
        # write the enum to the struct file instead of its own file
        if self.namespace and structs and self.namespace in structs:
                out_file = structs[self.namespace].get_out_file(enums_path)
        else:
            # Default behavior if no structs provided or no namespace
            if self.namespace:
                namespace_dir = enums_path / self.namespace.split('::')[0]
                namespace_dir.mkdir(exist_ok=True)
                out_file = namespace_dir / f"{self.safe_name.split('__')[-1]}.cpp"
            else:
                out_file = enums_path / f"{self.safe_name}.cpp"
        
        with open(out_file, 'a') as f:
            f.write(f"{self.get_comment_header()}{self.definition}\n\n")

    def parse_enum(self, def_line: str, lines: List[str], i: int) -> int:
        """Parse enum definition"""
        comment = lines[i - 1] if i > 0 else ""
        # Handle regular enum
        # Extract enum name from definition line
        def_line_clean = def_line.replace('enum ', '').replace('struct ', '').strip()
        if '{' in def_line_clean:
            # Enum with inline definition
            enum_name = def_line_clean.split('{')[0].strip()
        else:
            enum_name = def_line_clean.split()[0] if def_line_clean.split() else ""
        
        if "::" in enum_name:
            namespace, simple_name = self.parse_namespace(enum_name)
            self.name = simple_name
            self.namespace = namespace
        else:
            self.name = enum_name
            self.namespace = None
        
        # Collect enum body
        enum_buffer = [def_line]
        i += 1
        while i < len(lines) and lines[i].strip() != "};":
            enum_buffer.append(lines[i])
            i += 1
        if i < len(lines):
            enum_buffer.append(lines[i])  # Add closing };
            i += 1
        
        self.definition = "\n".join(enum_buffer)
        self.comment = comment
        self.is_ignored = should_ignore_class(self.full_name)
        
        return i, self.name, self
