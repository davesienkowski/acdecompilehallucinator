from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
from .constants import should_ignore_class
from .method import Method

@dataclass
class Struct:
    """Represents a parsed struct with its metadata"""
    name: str = ""
    definition: str = ""
    namespace: Optional[str] = None
    parent: Optional[str] = None
    is_generic: bool = False
    is_ignored: bool = False
    comment: str = ""
    methods: List = None 
    
    STRUCT_MODIFIERS = [
        '__unaligned ', '__cppobj ', '/*VFT*/ ',
        '__declspec(align(1)) ', '__declspec(align(2)) ',
        '__declspec(align(4)) ', '__declspec(align(8)) '
    ]
    
    @property
    def full_name(self) -> str:
        """Returns the fully qualified name"""
        if self.namespace:
            return f"{self.namespace}::{self.name}"
        return self.name
    
    @property
    def safe_name(self) -> str:
        """Returns a filesystem-safe name"""
        # Replace problematic characters that can cause long filenames
        safe_name = self.full_name.replace('::', '__').replace('_vtbl', '')
        # Handle template characters that can cause extremely long names
        safe_name = safe_name.replace('<', '_').replace('>', '_').replace(',', '_').replace(' ', '_')
        # Truncate if too long for filesystem limits
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name
    
    @property
    def simple_name(self) -> str:
        """Returns just the struct name without namespace"""
        return self.name.replace('_vtbl', '')
    
    def get_comment_header(self) -> str:
        """Generate appropriate comment header for the struct"""
        comment = f"// Struct {self.full_name}\n"
        return comment
    
    def extract_struct_name(self, def_line: str) -> tuple[str, Optional[str]]:
        """Extract struct name and parent class if any"""
        def_line = def_line.replace('struct ', '')
        parts = def_line.split(' : ')
        struct_name = parts[0].strip()
        parent = parts[1].strip() if len(parts) > 1 else None
        return struct_name, parent
    
    def parse_namespace(self, full_name: str) -> tuple[Optional[str], str]:
        """Split full name into namespace and simple name"""
        parts = full_name.split("::")
        if len(parts) == 1:
            return None, full_name
        return "::".join(parts[:-1]), parts[-1]

    def clean_struct_definition(self, def_line: str) -> str:
        """Remove modifiers from struct definition"""
        for modifier in self.STRUCT_MODIFIERS:
            def_line = def_line.replace(modifier, '')
        return def_line
    
    def get_out_file(self, src_path: str, structs_dict: Dict[str, any] = None) -> Path:
        """Return the output file path for this struct"""
        # Clean the safe_name to handle template instantiations that result in very long names
        safe_name_clean = self.safe_name.replace('<', '_').replace('>', '_').replace(',', '_').replace(' ', '_')
        if len(safe_name_clean) > 100:
            safe_name_clean = safe_name_clean[:100]
        
        if self.namespace:
            # Check if namespace itself is a struct
            if structs_dict and self.namespace in structs_dict:
                out_file = src_path / f"{safe_name_clean.split('__')[0]}.cpp"
            else:
                namespace_dir = src_path / self.namespace.split('::')[0]
                namespace_dir.mkdir(exist_ok=True)
                out_file = namespace_dir / f"{safe_name_clean.split('__')[-1]}.cpp"
        else:
            out_file = src_path / f"{safe_name_clean}.cpp"
        
        return out_file

    def write_to_file(self, src_path: str, structs_dict: Dict[str, any]):
        """Write a single struct to its appropriate file"""
        out_file = self.get_out_file(src_path, structs_dict)

        
        with open(out_file, 'a') as f:
            f.write(f"{self.get_comment_header()}{self.definition}\n\n")

    def parse_struct(self, def_line: str, lines: List[str], i: int) -> int:
        """Parse struct definition"""
        self.methods = []
        def_line = self.clean_struct_definition(def_line)
        comment = lines[i - 1]
        
        # Handle forward declarations
        if def_line.endswith(";"):
            struct_name = def_line.replace('struct ', '').replace(';', '')
            namespace, simple_name = self.parse_namespace(struct_name)
            struct = Struct(
                name=simple_name,
                definition=f"{struct_name}\n{{\n}};",
                namespace=namespace
            )
            return i, simple_name, struct
        
        # Parse full struct definition
        struct_name, parent = self.extract_struct_name(def_line)
        namespace, simple_name = self.parse_namespace(struct_name)
        
        struct_buffer = [def_line]
        i += 1
        
        # Collect struct body
        while lines[i] != "};":
            struct_buffer.append(lines[i])
            i += 1
        struct_buffer.append(lines[i])
        i += 1
        
        # Fill Struct object
        self.name = simple_name
        self.definition = "\n".join(struct_buffer)
        self.namespace = namespace
        self.parent = parent
        self.is_generic = '<' in struct_name
        self.is_ignored = should_ignore_class(struct_name)
        self.comment = comment
        
        return i, simple_name, self