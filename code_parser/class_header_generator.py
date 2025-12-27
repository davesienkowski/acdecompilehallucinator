"""
Class Header Generator
======================
Generates modernized C++ header files for classes, combining struct definitions
with method signatures, using LLM processing.
"""
import re
import json
from typing import List, Dict, Optional, Set, Any
from pathlib import Path


class ClassHeaderGenerator:
    """
    Generates class header files by combining struct definitions with method signatures.
    
    Workflow:
    1. Gather struct definition and all methods for a class
    2. Find type references and get their definitions as context
    3. Send to LLM for modernization
    4. Save processed header to database and file
    """
    
    # Few-shot examples

    FEW_SHOT_CLASS = """
Task: Generate only the C++ header file content for the given class. Include #pragma once, necessary #includes (minimal), class declaration with members, method prototypes, and documentation comments.
- Forward declare or assume others are handled elsewhere. Use MCP tools if needed for base class or member types.

STRICT RULES:
- Output ONLY code, no explanations
- Do not rename classes or method names.
- For referenced types, use the path provided in the "// Defined in:" comment for the #include directive.
- Remove all forward declarations.
- do not change any existing types unless they are void/undefined/etc. if a type is defined leave it
- make sure to keep all existing methods in place, do not invent new methods.
- Absolutely NEVER define base or referenced types, and DO NOT forward declare them.
- You can add code comments.
- DO NOT inline function definitions.
- You can clean up destructor method signatures to become valid cpp.
- Do your best to produce valid cpp code.

Example Input:
struct __cppobj Player : GameObject
{
    int _health;
    void Update(float dt);
};

Example Output:
#pragma once
#include "GameObject.h"

/*
  Represents a player entity inheriting from GameObject, handling health management and status flags.
*/
class Player : public GameObject {
public:
    // Health of the player
    int _health;

    // Update the player
    void Update(float dt);
};
"""

    FEW_SHOT_TEMPLATE = """
Task: Generate only the C++ header file content for the given template. Include #pragma once, necessary #includes (minimal), template declaration with members, and full method definitions.
- If the class name contains template parameters (e.g., List<int>), generate a generic template instead (e.g., template <typename T> class List).
- Use `template <typename T>` for single parameter templates, or `template <typename T, typename U>` for multiple.

STRICT RULES:
- Output ONLY code, no explanations
- Do not rename classes or method names (except to parameterize them in templates).
- For referenced types, use the path provided in the "// Defined in:" comment for the #include directive.
- do not change any existing types unless they are void/undefined/etc. if a type is defined leave it
- make sure to keep all existing methods in place, do not invent new methods.
- Absolutely NEVER define base or referenced types, and DO NOT forward declare them.
- For templates, ALWAYS provide full function definitions inside the class or immediately after it in the header.
- For templates, convert the struct to a templated class.
- You can clean up destructor method signatures to become valid cpp.
- Do your best to produce valid cpp code.
Example Input:
class List<int> {
public:
    int* _items;
    void Add(int item);
};

// Method Definition:
void List<int>::Add(int item) {
    _items[_count++] = item;
}

Example Output:
#pragma once

template <typename T>
class List {
public:
    T* _items;

    void Add(T item) {
        _items[_count++] = item;
    }
};
"""
    
    def __init__(self, db_handler, llm_client=None, debug_dir: Optional[Path] = None):
        """
        Initialize the header generator.
        
        Args:
            db_handler: DatabaseHandler instance for type lookups
            llm_client: Optional LLM client for processing (can be set later)
            debug_dir: Optional directory for debug output
        """
        self.db = db_handler
        self.llm = llm_client
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.dependency_analyzer = None  # Set externally if needed
        
        # Debug tracking
        self._last_analysis_prompt: Optional[str] = None
        self._last_analysis: Optional[str] = None
        self._last_prompt: Optional[str] = None
        self._last_response: Optional[str] = None
        self._last_types: Optional[Dict[str, Any]] = None
    
    def _write_debug(self, class_name: str):
        """Write debug files for the last generated header"""
        if not self.debug_dir:
            return
        
        # Build path: namespace/class/_header/
        # Handle namespaced class names like "Namespace::ClassName"
        parts = class_name.replace('::', '/').split('/')
        
        # Create directory: debug_dir/namespace/class/_header/
        header_dir = self.debug_dir / '/'.join(parts) / '_header'
        header_dir.mkdir(parents=True, exist_ok=True)

        # Write analysis prompt/response
        if self._last_analysis_prompt:
            (header_dir / "analysis_prompt.txt").write_text(self._last_analysis_prompt, encoding='utf-8')
        if self._last_analysis:
            (header_dir / "analysis.txt").write_text(self._last_analysis, encoding='utf-8')
        
        # Write prompt
        if self._last_prompt:
            prompt_path = header_dir / "prompt.txt"
            prompt_path.write_text(self._last_prompt, encoding='utf-8')
        
        # Write response
        if self._last_response:
            response_path = header_dir / "response.txt"
            response_path.write_text(self._last_response, encoding='utf-8')
        
        # Write types
        if self._last_types:
            types_path = header_dir / "types.json"
            types_path.write_text(json.dumps(self._last_types, indent=2), encoding='utf-8')
    
    def gather_class_info(self, class_name: str) -> Dict:
        """
        Gather all information needed to generate a class header.
        
        Returns dict with:
        - struct: The struct definition from DB
        - methods: List of methods belonging to this class
        - vtable: VTable code if present
        """
        # Get struct definition
        struct_rows = self.db.get_type_by_name(class_name, 'struct')
        struct = struct_rows[0] if struct_rows else None
        
        # Get all methods for this class
        methods = self.db.get_methods_by_parent(class_name)
        
        # Get all nested types
        nested_types = self.db.get_nested_types(class_name)
        nested_info = []
        for nt in nested_types:
            nt_name = nt[2]
            nt_methods = self.db.get_methods_by_parent(nt_name)
            nested_info.append({
                'name': nt_name,
                'struct': nt,
                'methods': nt_methods
            })

        return {
            'class_name': class_name,
            'struct': struct,
            'methods': methods,
            'vtable': struct[10] if struct and len(struct) > 10 else None,  # vtable_code column
            'nested': nested_info
        }
    
    def is_template_instantiation(self, class_name: str) -> bool:
        """Check if a class name looks like a template instantiation"""
        return '<' in class_name and class_name.endswith('>')

    def get_template_base_name(self, class_name: str) -> str:
        """Extract the base name from a template instantiation (e.g. List<int> -> List)"""
        if '<' in class_name:
            return class_name.split('<')[0].strip()
        return class_name

    def extract_method_signature(self, method_row) -> str:
        """Extract a clean method signature from a method row"""
        # method_row format: (id, name, full_name, definition, namespace, parent, ...)
        definition = method_row[3] if len(method_row) > 3 else ""
        
        # Extract just the first line (signature)
        lines = definition.split('\n')
        signature = lines[0] if lines else ""
        
        # Clean up decompiler artifacts
        for modifier in ['__cdecl', '__stdcall', '__thiscall', '__userpurge',
                        '__usercall', '__fastcall', '__noreturn']:
            signature = signature.replace(modifier, '')
        
        # Remove body brace if present
        signature = signature.split('{')[0].strip()
        if not signature.endswith(';'):
            signature += ';'
            
        return signature
    
    def find_type_references(self, class_info: Dict) -> Set[str]:
        """Find all type names referenced in the class definition and methods"""
        references = set()
        
        # Use dependency analyzer if available
        if self.dependency_analyzer:
            struct = class_info.get('struct')
            if struct:
                code = struct[5] if len(struct) > 5 else ""
                references.update(self.dependency_analyzer.extract_type_references(code))
            
            # Check method signatures too
            for method in class_info.get('methods', []):
                definition = method[3] if len(method) > 3 else ""
                refs = self.dependency_analyzer.extract_type_references(definition)
                references.update(refs)
        else:
            # Fallback: simple pattern matching
            struct = class_info.get('struct')
            if struct:
                code = struct[5] if len(struct) > 5 else ""
                # Look for capitalized words that might be types
                pattern = re.compile(r'\b([A-Z][A-Za-z0-9_]*)\b')
                matches = pattern.findall(code)
                references.update(matches)
        
        # Remove the class itself
        references.discard(class_info['class_name'])
        
        return references

    def _get_type_filepath(self, full_name: str) -> str:
        """Get the expected header file path for a type"""
        if '::' in full_name:
            # Turbine::Debug::Assert -> Turbine/Assert.h
            parts = full_name.split('::')
            return f"{parts[0]}/{parts[-1]}.h"
        return f"{full_name}.h"
    
    def get_reference_context(self, type_names: Set[str], max_types: int = 10) -> str:
        """
        Get definitions for referenced types to provide as context.
        Checks processed types first, falls back to raw.
        
        Args:
            type_names: Set of type names to look up
            max_types: Maximum number of types to include (to limit context size)
        """
        context_parts = []
        included = 0
        
        for name in sorted(type_names):
            if included >= max_types:
                break
            
            type_def, is_processed = self.db.get_type_with_fallback(name)
            if type_def:
                # Calculate path
                file_path = self._get_type_filepath(name)
                path_info = f"// Defined in: \"{file_path}\"\n"

                if is_processed and type_def.get('processed_header'):
                    # Ensure we only include the core definition from the processed header
                    # (Though processed_header should already be just the header)
                    context_parts.append(f"// Reference: {name} (modernized)\n{path_info}{type_def['processed_header']}")
                    included += 1
                elif type_def.get('code'):
                    context_parts.append(f"// Reference: {name} (raw decompiled)\n{path_info}{type_def['code']}")
                    included += 1
        
        return "\n\n".join(context_parts)
    
    def analyze_class(self, class_name: str) -> Optional[str]:
        """
        Perform initial analysis of the class and extract all referenced types.
        """
        if not self.llm:
            return None

        class_info = self.gather_class_info(class_name)
        if not class_info.get('struct'):
            return None
            
        struct = class_info.get('struct')
        struct_code = struct[5] if struct and len(struct) > 5 else ""
        
        # Find and resolve type references using regex patterns
        existing_references = self.find_type_references(class_info)
        methods_str = ""
        
        vtable_code = class_info.get('vtable', "")
        vtable_str = f"\nVTable:\n{vtable_code}\n" if vtable_code else ""
        
        # Build prompt for type extraction
        prompt = f"""Extract all type references from the following C++ class: {class_name}

Struct Definition:
{struct_code}
{vtable_str}

Methods:
{methods_str}
"""
        # Add nested types to analysis
        if class_info.get('nested'):
            prompt += "\nNested Types:\n"
            for nt in class_info['nested']:
                nt_code = nt['struct'][5] if nt['struct'] and len(nt['struct']) > 5 else ""
                prompt += f"--- Nested: {nt['name']} ---\n{nt_code}\n"
                nt_sigs = [self.extract_method_signature(m) for m in nt['methods']]
                if nt_sigs:
                    prompt += "Methods:\n" + "\n".join(nt_sigs) + "\n"

        prompt += """
Task: Extract ALL type names referenced in the code. These include:
- Base class names
- Member variable types
- Method parameter types
- Method return types
- Types used in method bodies
- Template parameter types
- Nested type references

Output format: A JSON array of unique type names, one per line. Only output the JSON array, nothing else.

Example:
[
  "BaseClass",
  "MemberType",
  "ParameterType",
  "ReturnType"
]
"""
        self._last_analysis_prompt = prompt
        analysis = self._call_llm(prompt)
        
        # Try to parse the response to get the list of referenced types
        # Instead of parsing as JSON, find all quoted strings which represent type names
        import re
        extracted_references = set()
        try:
            # Find all quoted strings in the response
            quoted_strings = re.findall(r'"([^"]*)"', analysis)
            if quoted_strings:
                extracted_references = set(quoted_strings)
        except Exception as e:
            # If parsing fails, use the existing regex-based extraction
            extracted_references = existing_references

        # Combine existing references with LLM-extracted references
        combined_references = existing_references | extracted_references
        
        # Format the analysis to include the combined references
        analysis_result = {
            "analysis": analysis,
            "referenced_types": list(combined_references),
            "llm_extracted_types": list(extracted_references),
            "regex_extracted_types": list(existing_references)
        }
        
        self._last_analysis = json.dumps(analysis_result, indent=2)
        return json.dumps(analysis_result)
    def build_prompt(self, class_info: Dict, reference_context: str = "", analysis: str = "",
                     method_definitions: List[str] = None, is_template: bool = False) -> str:
        """Build the LLM prompt for header generation"""
        struct = class_info.get('struct')
        struct_code = struct[5] if struct and len(struct) > 5 else ""
        
        # Build methods list
        if method_definitions:
            methods_str = "\n".join(f"// Method Definition:\n{defn}" for defn in method_definitions)
        else:
            # Build method signatures list
            method_sigs = []
            for method in class_info.get('methods', []):
                sig = self.extract_method_signature(method)
                if sig:
                    method_sigs.append(sig)
            
            methods_str = "\n".join(f"// Method Signature: {sig}" for sig in method_sigs) if method_sigs else "// No methods"
        
        vtable_code = class_info.get('vtable', "")
        vtable_str = f"\nVTable:\n{vtable_code}\n" if vtable_code else ""
        
        prompt = f"""Generate a clean, modern C++ header for the class: {class_info['class_name']}
 
Struct Definition:
{struct_code}
{vtable_str}

Methods:
{methods_str}
"""
        # Add nested types to prompt
        if class_info.get('nested'):
            prompt += "\nNested Types to include in this SAME header file:\n"
            for nt in class_info['nested']:
                nt_code = nt['struct'][5] if nt['struct'] and len(nt['struct']) > 5 else ""
                prompt += f"\n--- Nested Type: {nt['name']} ---\n"
                prompt += f"Definition:\n{nt_code}\n"
                
                nt_sigs = []
                for m in nt['methods']:
                    sig = self.extract_method_signature(m)
                    if sig: nt_sigs.append(sig)
                
                if nt_sigs:
                    prompt += "Methods:\n" + "\n".join(nt_sigs) + "\n"
        
        if reference_context:
            prompt += f"""
Referenced Types (for context only, do not redefine):
{reference_context}
"""
        
        few_shot = self.FEW_SHOT_TEMPLATE if is_template else self.FEW_SHOT_CLASS
        prompt += f"\n{few_shot}"
        
        return prompt
    def generate_header(self, class_name: str, save_to_db: bool = True, analysis: str = None) -> Optional[str]:
        """
        Generate a modernized header for the given class.
        
        Args:
            class_name: Name of the class to process
            save_to_db: Whether to save the result to the database
            analysis: Optional analysis string from analyze_class
            
        Returns:
            Generated header code, or None if generation failed
        """
        if not self.llm:
            raise ValueError("LLM client not set. Set it via generator.llm = client")
        
        # Gather class information
        class_info = self.gather_class_info(class_name)
        
        if not class_info.get('struct'):
            print(f"Warning: No struct definition found for {class_name}")
            return None
        
        # Start with references found via regex patterns
        references = self.find_type_references(class_info)
        
        # If analysis is provided, extract types from it by finding quoted strings
        if analysis:
            import re
            try:
                # Find all quoted strings in the analysis (these represent type names)
                quoted_strings = re.findall(r'"([^"]*)"', analysis)
                if quoted_strings:
                    analysis_references = set(quoted_strings)
                    # Combine with regex-based references
                    references = references | analysis_references
            except Exception as e:
                # If parsing fails, just use the regex-based references
                pass
        
        # Combine all references - deduplicate by using set operations
        all_references = {f for f in references if "<" not in f}

        # Get context for all referenced types
        context = self.get_reference_context(all_references)
        
        # Gather full method definitions for templates
        method_definitions = None
        method_definitions = [m[3] for m in class_info.get('methods', [])]
        
        # Track types for debug
        self._last_types = {
            "referenced": list(all_references),
            "context_preview": context[:500] + "..." if len(context) > 500 else context
        }
        
        # Build prompt (no longer include analysis text, just use extracted types)
        is_template = self.is_template_instantiation(class_name)
        prompt = self.build_prompt(class_info, context, analysis=None, # No analysis text in prompt
                                   method_definitions=method_definitions,
                                   is_template=is_template)
        self._last_prompt = prompt
        
        # Call LLM
        header = self._call_llm(prompt)
        self._last_response = header
        
        # Write debug output
        self._write_debug(class_name)
        
        if header and save_to_db:
            struct = class_info['struct']
            original_code = struct[5] if struct and len(struct) > 5 else ""
            self.db.store_processed_type(
                name=class_name,
                type_kind='struct',
                original_code=original_code,
                processed_header=header,
                dependencies=list(all_references)
            )
        
        return header
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with the given prompt.
        Override this method or set self.llm_generate for custom LLM integration.
        """
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt)
        elif callable(self.llm):
            return self.llm(prompt)
        else:
            raise NotImplementedError("LLM client must have a 'generate' method or be callable")
    
    def write_header_file(self, class_name: str, header_code: str, 
                          output_dir: Path, namespace: str = None) -> Path:
        """
        Write the generated header to a file.
        
        Args:
            class_name: Name of the class
            header_code: Generated header content
            output_dir: Base output directory
            namespace: Optional namespace for subdirectory organization
            
        Returns:
            Path to the written file
        """
        # Build output path
        if namespace:
            header_dir = output_dir / "include" / namespace.replace('::', '/')
        else:
            header_dir = output_dir / "include"
        
        header_dir.mkdir(parents=True, exist_ok=True)
        header_path = header_dir / f"{class_name}.h"
        
        header_path.write_text(header_code, encoding='utf-8')
        
        return header_path
