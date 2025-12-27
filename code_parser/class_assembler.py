"""
Class Assembler
===============
Assembles final C++ output files from processed components.
No LLM needed - just combines processed headers and method implementations.
"""
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProcessedMethod:
    """Represents a processed method ready for assembly"""
    name: str
    full_name: str
    parent_class: Optional[str]
    processed_code: str
    dependencies: List[str]
    offset: str


class ClassAssembler:
    """
    Assembles final C++ source files from processed components.
    
    This is a non-LLM component that simply combines:
    - Generated headers (from ClassHeaderGenerator)
    - Processed method implementations (from FunctionProcessor)
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize the assembler.
        
        Args:
            output_dir: Base output directory for generated files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def assemble_source(self, class_name: str, 
                        processed_methods: List[ProcessedMethod],
                        namespace: Optional[str] = None) -> str:
        """
        Combine processed methods into a .cpp source file.
        
        Args:
            class_name: Name of the class
            processed_methods: List of processed method objects
            namespace: Optional namespace for organization
            
        Returns:
            Generated source file content
        """
        lines = []
        if namespace:
            # Use forward slash for include path regardless of OS
            include_path = f"{namespace.replace('::', '/')}/{class_name}.h"
            lines.append(f'#include "{include_path}"')
        else:
            lines.append(f'#include "{class_name}.h"')
        lines.append('')
        
        for method in processed_methods:
            # Add the processed code
            code = method.processed_code.strip()
            if code:
                lines.append(code)
                lines.append('')
        
        return '\n'.join(lines)
    
    def write_source_file(self, class_name: str,
                          processed_methods: List[ProcessedMethod],
                          namespace: Optional[str] = None) -> Path:
        """
        Write assembled source file to disk.
        
        Args:
            class_name: Name of the class
            processed_methods: List of processed methods
            namespace: Optional namespace for subdirectory
            
        Returns:
            Path to written file
        """
        # Skip if no methods with actual code
        non_empty_methods = [m for m in processed_methods if m.processed_code.strip()]
        if not non_empty_methods:
            return None
        
        # Build output path
        if namespace:
            source_dir = self.output_dir / "src" / namespace.replace('::', '/')
        else:
            source_dir = self.output_dir / "src"
        
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"{class_name}.cpp"
        
        # Generate and write content
        content = self.assemble_source(class_name, non_empty_methods, namespace)
        source_path.write_text(content, encoding='utf-8')
        
        return source_path
    
    def write_header_file(self, class_name: str,
                          header_code: str,
                          namespace: Optional[str] = None,
                          path: Optional[Path] = None) -> Path:
        """
        Write header file to disk.
        
        Args:
            class_name: Name of the class
            header_code: Generated header content
            namespace: Optional namespace for subdirectory
            path: Optional full path to write to
            
        Returns:
            Path to written file
        """
        if not header_code or not header_code.strip():
            return None
        
        if path:
            header_path = path
            header_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Build output path
            if namespace:
                header_dir = self.output_dir / "include" / namespace.replace('::', '/')
            else:
                header_dir = self.output_dir / "include"
            
            header_dir.mkdir(parents=True, exist_ok=True)
            header_path = header_dir / f"{class_name}.h"
        
        header_path.write_text(header_code, encoding='utf-8')
        
        return header_path
    
    def write_enum_header(self, enum_name: str,
                          enum_code: str,
                          namespace: Optional[str] = None) -> Path:
        """
        Write enum header file. Enums don't need LLM processing.
        
        Args:
            enum_name: Name of the enum
            enum_code: Enum definition code
            namespace: Optional namespace for subdirectory
            
        Returns:
            Path to written file
        """
        if not enum_code:
            return None
        
        # Build output path
        if namespace:
            header_dir = self.output_dir / "include" / namespace.replace('::', '/')
        else:
            header_dir = self.output_dir / "include"
        
        header_dir.mkdir(parents=True, exist_ok=True)
        header_path = header_dir / f"{enum_name}.h"
        
        # Add pragma once if not present
        content = enum_code
        if not content.strip().startswith('#pragma once'):
            content = '#pragma once\n\n' + content
        
        header_path.write_text(content, encoding='utf-8')
        
        return header_path
