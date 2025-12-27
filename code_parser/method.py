from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
from .constants import should_ignore_global_method, should_ignore_class_method, should_ignore_class


@dataclass
class Method:
    """Represents a parsed method with its metadata"""
    name: str = ""
    full_name: str = ""
    definition: str = ""
    namespace: Optional[str] = None
    parent: Optional[str] = None
    is_generic: bool = False
    is_ignored: bool = False
    offset: str = ""
    return_type: str = ""
    file: str = ""
    
    FUNC_MODIFIERS = [
        '__cdecl', '__stdcall', '__thiscall', '__userpurge',
        '__usercall', '__fastcall', '__noreturn', '__spoils<ecx>'
    ]
    
    @property
    def safe_name(self) -> str:
        """Returns a filesystem-safe name"""
        return self.full_name.replace('::', '__').replace('_vtbl', '')
    
    @property
    def is_global(self) -> str:
        return "::" not in self.full_name
    
    @property
    def simple_name(self) -> str:
        """Returns just the func name without namespace"""
        return self.name.replace('_vtbl', '')
    
    def clean_definition(self, def_line: str) -> str:
        """Remove modifiers from struct definition"""
        for modifier in self.FUNC_MODIFIERS:
            def_line = def_line.replace(modifier, '')
        return def_line
    
    def get_out_file(self, src_path: str, structs_dict: Dict[str, any] = None) -> Path:
        """Return the output file path for this struct"""
        if self.namespace:
            # Check if namespace itself is a struct
            if self.parent and self.parent in structs_dict:
                out_file = structs_dict[self.parent].get_out_file(src_path, structs_dict)
            else:
                namespace_dir = src_path / self.namespace.split('::')[0]
                namespace_dir.mkdir(exist_ok=True)
                out_file = namespace_dir / f"{self.parent}.cpp"
        else:
            out_file = src_path / f"{self.parent}.cpp"
        
        return out_file

    def write_to_file(self, src_path: str, structs: Dict[str, any]):
        """Write a single struct to its appropriate file"""
        out_file = self.get_out_file(src_path, structs)

        if self.parent and self.parent in structs:
            structs[self.parent].methods.append(self)
        
        with open(out_file, 'a') as f:
            f.write(f"// Function Offset: 0x{self.offset}\n")
            f.write(f"{self.definition}\n\n")

    def extract_func_name(self, def_line: str):
        def_line = self.clean_definition(def_line)
        def_line = def_line.replace(' * *', '**')
        def_line = def_line.replace(' * ', '* ')
        def_line = def_line.replace("`vector deleting destructor'", 'VectorDeletingDestructor')
        def_line = def_line.replace("`scalar deleting destructor'", 'ScalarDeletingDestructor')
        def_line = def_line.replace("operator>", "operatorGreaterThan")
        def_line = def_line.replace("operator<", "operatorLessThan")
        def_line = def_line.replace("@<eax>", "")
        def_line = def_line.replace("@<al>", "")

        # Find the last opening parenthesis for arguments
        last_paren = def_line.find('(')
        if last_paren == -1:
            raise ValueError("No opening parenthesis found in signature")
        
        # Extract everything after the last '(' as arguments
        args_start = last_paren + 1
        args_string = def_line[args_start:].rstrip(')')
        
        # Now work backwards from the '(' to find the function name
        # We need to count <> pairs to handle templates
        i = last_paren - 1
        angle_depth = 0
        func_name_end = last_paren
        
        # Traverse left, counting <> pairs
        while i >= 0:
            char = def_line[i]
            
            if char == '>':
                angle_depth += 1
            elif char == '<':
                angle_depth -= 1
            elif char == ' ' and angle_depth <= 0:
                # Found a space outside of template brackets
                # This marks the end of the function name
                break
            
            i -= 1
        
        # Function name is from i+1 to last_paren
        func_name_start = i + 1
        self.name = def_line[func_name_start:func_name_end]
        
        # Return type is everything before the function name
        self.return_type = def_line[:func_name_start].strip()

        if self.name.startswith('*'):
            self.name = self.name[1:]
            self.return_type = self.return_type + '*'

        if self.return_type == "":
            print("Bad Return?", def_line)

        self.full_name = self.name

        if ("::" in self.name):
            parts = self.name.split("::")
            self.name = parts[-1]
            self.parent = "::".join(parts[:-1])

        if self.parent and "::" in self.parent:
            parts = self.parent.split("::")
            self.parent = parts[-1]
            self.namespace = "::".join(parts[:-1])

        self.name = self.name.replace("operatorGreaterThan", "operator>")
        self.name = self.name.replace("VectorDeletingDestructor", "operator<")

        return self.simple_name, None

    def parse(self, line: str, lines: List[str], i: int) -> int:
        """Parse func definition"""
        func_buffer = []
        parts = line.split(' ')
        if len(parts) < 2:
            print("Bad Len?", line)
        else:
            self.offset = line.split(' ')[1].strip('()')

        i = i + 1
        def_line = lines[i]

        while def_line.startswith('//'):
            i = i + 1
            def_line = lines[i]

        if def_line.startswith('#'):
            return i, None, None

        simple_name, parent = self.extract_func_name(def_line)

        # Collect function body
        while lines[i] != "}" and i < len(lines) - 1:
            func_buffer.append(lines[i])
            i += 1
        func_buffer.append(lines[i])
        i += 1

        self.definition = "\n".join(func_buffer)

        if self.name.startswith('`'):
            self.is_ignored = True
        elif self.is_global:
            self.is_ignored = should_ignore_global_method(self.name)
        else:
            self.is_ignored = should_ignore_class_method(self.full_name)
        
        if self.namespace and self.namespace.startswith('`'):
            self.is_ignored = True

        if self.parent and (should_ignore_class(self.parent) or self.parent.startswith('`')):
            self.is_ignored = True
        
        return i, simple_name, self