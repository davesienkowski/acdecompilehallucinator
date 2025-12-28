"""
Function Processor
==================
Processes individual functions/methods through the LLM pipeline,
resolving type references and storing processed output.
"""
import re
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProcessedFunction:
    """Represents a processed function result"""
    name: str
    full_name: str
    parent_class: Optional[str]
    original_code: str
    processed_code: str
    dependencies: List[str]
    offset: str


class FunctionProcessor:
    """
    Processes individual functions with type context.
    
    Workflow:
    1. Extract type references from function code
    2. Look up type definitions (processed first, then raw)
    3. Send to LLM with type context
    4. Store processed result
    """
    
    # Few-shot example for function modernization
    FEW_SHOT_FUNCTION = """
Task: Modernize this decompiled C++ function to clean, idiomatic modern C++ (C++17+). Preserve ALL logic, and structure.

STRICT RULES:
- Output ONLY the function code, no explanations, other classes, forward declarations, includes, etc.
- Do not rename the function or parameters
- Renaming local variables to improve readability is encouraged
- Remove decompiler artifacts (__thiscall, explicit 'this' pointer, etc. )
- Use modern types (uint32_t, bool instead of int for booleans)
- Add comments to explain the logic / purpose of the function
- Do not add or remove logic
- Keep the function signature compatible with the class header
- If you see enum values, use the enum name instead of the value
- Do not make up constants that dont exist. 
- Do your best to produce valid cpp code.

Example Input:
void __thiscall Player::Update(Player *this, float dt)
{
    int v1 = this->_health;
    if (v1 <= 0) {
        this->_flags = this->_flags | 0x1;
    }
}

Example Output:
/*
    Updates the player's state.
*/
void Player::Update(float dt) {
    if (_health <= 0) {
        // Add dead flag
        _flags |= 0x1;
    }
}
"""

    # Verification prompt
    VERIFICATION_PROMPT = """
You are a senior code reviewer. Compare the ORIGINAL decompiled function with the MODERNIZED version.

ORIGINAL:
```cpp
{original}
```

MODERNIZED:
```cpp
{processed}
```

TASK:
Determine if the MODERNIZED version preserves the core logic and semantics of the ORIGINAL.
- Variable renaming is allowed if it improves readability (implied by "Modernize").
- Type updates are allowed (e.g. int -> bool).
- Structure changes are allowed (e.g. early returns).
- Safety checks are allowed (e.g. null checks).
- LOGIC must remain identical.

RESPONSE:
Return ONLY the JSON object:
{{
  "equivalent": true/false,
  "reason": "Brief explanation if false, otherwise empty string"
}}
"""
    
    def __init__(self, db_handler, llm_client=None, debug_dir: Optional[Path] = None):
        """
        Initialize the function processor.
        
        Args:
            db_handler: DatabaseHandler instance for type lookups
            llm_client: Optional LLM client for processing
            debug_dir: Optional directory for debug output
        """
        self.db = db_handler
        self.llm = llm_client
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.dependency_analyzer = None  # Set externally if needed
        
        # Patterns for extracting types from function code
        self.type_patterns = [
            # Pointer/reference: SomeType* or SomeType&
            re.compile(r'\b([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*[*&]'),
            # Parameter types: (SomeType param) - also include lowercase enum names like eCombatMode
            re.compile(r'\(\s*(?:[^()]*,\s*)?(const\s+)?([A-Za-z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s+\w+'),
            # Variable declarations: SomeType varname;
            re.compile(r'^\s*([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s+\w+\s*[;=]', re.MULTILINE),
            # Return type: SomeType ClassName::Method
            re.compile(r'^([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*[*&]?\s+\w+::'),
            # Cast: (SomeType*)
            re.compile(r'\(\s*([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*\*?\s*\)'),
            # Member access: ->member or .member with type context
            re.compile(r'([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)::(?:\w+)'),
            # Additional pattern for parameter types with lowercase enum names (like eCombatMode)
            re.compile(r'\b(const\s+)?([a-z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s+\w+(?=\s*[),])'),
        ]
        
        # Types to ignore
        self.ignore_types = {
            'int', 'unsigned', 'char', 'short', 'long', 'float', 'double',
            'void', 'bool', 'size_t', 'uint8_t', 'uint16_t', 'uint32_t',
            'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'BYTE', 'WORD', 'DWORD', 'QWORD', 'BOOL', 'HANDLE', 'HRESULT',
            'TRUE', 'FALSE', 'NULL', 'nullptr', 'String', 'Vector',
        }
        
    def _write_debug(self, method_name: str, parent_class: str, prompt: str, response: str):
        """Write debug files for function"""
        if not self.debug_dir:
            return
            
        # Build path: debug/namespace/class/method/
        # Handle namespaced class names
        parts = parent_class.replace('::', '/').split('/') if parent_class else ['Global']
        
        method_safe = re.sub(r'[^a-zA-Z0-9_]', '_', method_name)
        
        debug_dir = self.debug_dir / '/'.join(parts) / method_safe
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        (debug_dir / "prompt.txt").write_text(prompt, encoding='utf-8')
        (debug_dir / "response.txt").write_text(response, encoding='utf-8')

    def find_type_references(self, code: str) -> Set[str]:
        """Find all type names used in function code"""
        references = set()
        
        for pattern in self.type_patterns:
            matches = pattern.findall(code)
            for match in matches:
                # Handle both single matches and tuple matches from regex groups
                if isinstance(match, tuple):
                    # For patterns with multiple groups, the type name is in the last non-empty group
                    type_name = None
                    for group in reversed(match):
                        if group:  # Find the last non-empty group which should be the type name
                            type_name = group
                            break
                    if type_name and type_name not in self.ignore_types:
                        if type_name[0].isupper() or type_name.startswith('e') or '::' in type_name:
                            references.add(type_name)
                else:
                    # Handle single string matches
                    if match and match not in self.ignore_types:
                        if match[0].isupper() or match.startswith('e') or '::' in match:
                            references.add(match)
        
        return references
    
    def get_reference_context(self, type_names: Set[str], max_types: int = 10) -> str:
        """
        Get definitions for referenced types.
        Checks processed types first, falls back to raw.
        """
        context_parts = []
        included = 0
        
        for name in sorted(type_names):
            if included >= max_types:
                break
            
            type_def, is_processed = self.db.get_type_with_fallback(name)
            if type_def:
                # Helper: Get the expected header file path for a type
                # (Duplicate of ClassHeaderGenerator logic for now, could be unified)
                if '::' in name:
                    parts = name.split('::')
                    file_path = f"{parts[0]}/{parts[-1]}.h"
                else:
                    file_path = f"{name}.h"
                
                path_info = f"// Defined in: \"{file_path}\"\n"

                if is_processed and type_def.get('processed_header'):
                    # Use just the class declaration from header
                    header = type_def['processed_header']
                    context_parts.append(f"// Reference: {name} (modernized)\n{path_info}{header}")
                    included += 1
                elif type_def.get('code'):
                    context_parts.append(f"// Reference: {name} (raw decompiled)\n{path_info}{type_def['code']}")
                    included += 1
        
        return "\n\n".join(context_parts)
    
    def get_parent_header_context(self, parent_class: Optional[str]) -> str:
        """
        Get the processed header of the parent class to provide as context.
        """
        if not parent_class:
            return ""
        
        # Get the processed header for the parent class
        parent_type_def, is_processed = self.db.get_type_with_fallback(parent_class)
        
        if parent_type_def and is_processed and parent_type_def.get('processed_header'):
            # Extract just the class declaration part from the header for function context
            header = parent_type_def['processed_header']
            return f"// Parent class definition:\n{header}"
        elif parent_type_def and parent_type_def.get('code'):
            # Fallback to raw code if no processed header exists
            return f"// Parent class definition (raw):\n{parent_type_def['code']}"
        
        return ""

    def build_prompt(self, method_definition: str, parent_class: Optional[str],
                     reference_context: str = "", analysis: str = "") -> str:
        """Build the LLM prompt for function modernization"""
        prompt = f"""Modernize this decompiled C++ function:

```cpp
{method_definition}
```
"""
        
        if parent_class:
            prompt += f"\nThis function belongs to class: {parent_class}\n"
        
        # Add parent class header context
        parent_header_context = self.get_parent_header_context(parent_class)
        if parent_header_context:
            prompt += f"""
Parent Class Definition:
{parent_header_context}
"""
        
        if reference_context:
            prompt += f"""
Referenced Types (for context):
{reference_context}
"""
        
        prompt += f"\n{self.FEW_SHOT_FUNCTION}"
        
        return prompt
    
    
    def verify_logic(self, original: str, processed: str) -> Tuple[bool, str]:
        """
        Verify that processed code matches original logic.
        Returns (equivalent, reason)
        """
        import json
        
        prompt = self.VERIFICATION_PROMPT.format(
            original=original, 
            processed=processed
        )
        
        response = self._call_llm(prompt)
        response = self._clean_llm_output(response)
        
        # Parse JSON response
        try:
            # Try to find JSON block if wrapped
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("equivalent", False), data.get("reason", "")
        except json.JSONDecodeError:
            pass
            
        return False, f"Failed to parse verification response: {response[:50]}..."

    def process_function(self, method_row: Tuple, save_to_db: bool = True, analysis: str = None) -> Optional[ProcessedFunction]:
        """
        Process a single function through the LLM pipeline.
        
        Args:
            method_row: Method row from database (id, name, full_name, definition, ...)
            save_to_db: Whether to save the result to the database
            analysis: Optional analysis string from class analysis
            
        Returns:
            ProcessedFunction object, or None if processing failed
        """
        if not self.llm:
            raise ValueError("LLM client not set")
        
        # Extract method info from row
        # Format: (id, name, full_name, definition, namespace, parent, is_generic, is_ignored, offset, return_type, is_global)
        name = method_row[1]
        full_name = method_row[2]
        definition = method_row[3]
        namespace = method_row[4] if len(method_row) > 4 else None
        parent = method_row[5] if len(method_row) > 5 else None
        offset = method_row[8] if len(method_row) > 8 else "0"
        
        # Find type references from the function definition
        references = self.find_type_references(definition)
        
        # If analysis is provided, extract types from it
        if analysis:
            import json
            try:
                analysis_data = json.loads(analysis)
                # Get referenced types from the analysis result (which is a dict)
                analysis_references = set(analysis_data.get("referenced_types", []))
                # Combine with regex-based references
                references = references | analysis_references
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing fails or if analysis_data doesn't have get method (is not dict),
                # just use the regex-based references
                pass
        
        # Remove the parent class from references (we're defining it)
        if parent:
            references.discard(parent)
        
        # Get all types from the database and add them as potential references
        all_db_types = set()
        for struct_row in self.db.get_structs():
            all_db_types.add(struct_row[2])  # name column
        for enum_row in self.db.get_enums():
            all_db_types.add(enum_row[2])  # name column
        
        # Combine all references - deduplicate by using set operations
        all_references = references | all_db_types
        
        # Get context for referenced types
        context = self.get_reference_context(all_references)
        
        # Build prompt and call LLM - don't pass analysis text, just use extracted types
        prompt = self.build_prompt(definition, parent, context, analysis=None)  # Don't pass analysis text, just use extracted types
        processed_code = self._call_llm(prompt)
        
        if not processed_code:
            return None
        
        # Clean LLM output
        processed_code = self._clean_llm_output(processed_code)

        # ────────────────────────────────────────────────────────────────────────
        # Verification Step with retry logic
        # ────────────────────────────────────────────────────────────────────────
        import logging
        logger = logging.getLogger(__name__)
        
        is_valid = False
        reason = ""
        retry_count = 0
        max_retries = 5
        
        while not is_valid and retry_count < max_retries:
            is_valid, reason = self.verify_logic(definition, processed_code)
            
            if not is_valid and retry_count < max_retries:
                # Log the validation failure with retry count
                logger.warning(f"Function {full_name} validation failed on attempt {retry_count + 1}/{max_retries}: {reason}")
                
                # Build a feedback prompt to improve the function based on the verification failure
                feedback_prompt = f"""Original function:
```cpp
{definition}
```

Attempted modernization:
```cpp
{processed_code}
```

Verification feedback: {reason}

Please regenerate the function addressing the issues mentioned in the verification feedback. Ensure that the logic remains identical while improving the code style and structure where possible.
"""
                
                # Call the LLM again with the feedback
                processed_code = self._call_llm(feedback_prompt)
                if processed_code:
                    processed_code = self._clean_llm_output(processed_code)
                
                retry_count += 1
            else:
                # Log the successful validation
                if is_valid:
                    if retry_count == 0:
                        logger.info(f"Function {full_name} validation successful on first attempt")
                    else:
                        logger.info(f"Function {full_name} validation successful after {retry_count} retries")

        # Debug output
        if self.debug_dir and parent:
             # Use full parent name for debug path if possible
             parent_full = f"{namespace}::{parent}" if namespace else parent
             self._write_debug(name, parent_full, prompt, processed_code)
             # Also write verification debug
             method_safe = re.sub(r'[^a-zA-Z0-9_]', '_', name)
             parts = parent_full.replace('::', '/').split('/')
             debug_dir = self.debug_dir / '/'.join(parts) / method_safe
             
             (debug_dir / "verification.txt").write_text(
                 f"Equivalent: {is_valid}\nReason: {reason}\nRetries: {retry_count}",
                 encoding='utf-8'
             )
        
        if not is_valid:
            processed_code = f"// VERIFICATION FAILED after {max_retries} attempts: {reason}\n{processed_code}"

        result = ProcessedFunction(
            name=name,
            full_name=full_name,
            parent_class=parent,
            original_code=definition,
            processed_code=processed_code,
            dependencies=list(references),
            offset=offset
        )
        
        if save_to_db:
            # Create a method-like object for storing
            from .method import Method
            method = Method()
            method.name = name
            method.full_name = full_name
            method.parent = parent
            method.namespace = namespace
            method.definition = definition
            method.offset = offset
            
            engine_name = self.llm.name if self.llm and hasattr(self.llm, 'name') else "lm-studio"
            self.db.store_processed_method(method, processed_code, list(references), engine_used=engine_name)
        
        return result
    
    def process_class_methods(self, class_name: str, 
                              save_to_db: bool = True) -> List[ProcessedFunction]:
        """
        Process all unprocessed methods for a given class.
        
        Args:
            class_name: Name of the class whose methods to process
            save_to_db: Whether to save results to database
            
        Returns:
            List of ProcessedFunction objects
        """
        # Get unprocessed methods for this class
        methods = self.db.get_unprocessed_methods(parent_class=class_name)
        
        results = []
        for method in methods:
            result = self.process_function(method, save_to_db=save_to_db)
            if result:
                results.append(result)
        
        return results
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt"""
        if hasattr(self.llm, 'generate'):
            return self.llm.generate(prompt)
        elif callable(self.llm):
            return self.llm(prompt)
        else:
            raise NotImplementedError("LLM client must have 'generate' method or be callable")
    
    def _clean_llm_output(self, text: str) -> str:
        """Remove common LLM output artifacts"""
        if not text:
            return ""
        
        # Remove markdown code blocks
        text = re.sub(r'^```(?:cpp|c\+\+|c|\w+)?\s*\n?', '', text, flags=re.M | re.I)
        text = re.sub(r'\n?```$', '', text, flags=re.M)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        
        return text.strip()
    
    def write_source_file(self, class_name: str, functions: List[ProcessedFunction],
                          output_dir: Path, namespace: str = None) -> Path:
        """
        Write processed functions to a source file.
        
        Args:
            class_name: Name of the class
            functions: List of processed functions
            output_dir: Base output directory
            namespace: Optional namespace for subdirectory organization
            
        Returns:
            Path to the written file
        """
        # Build output path
        if namespace:
            source_dir = output_dir / "src" / namespace.replace('::', '/')
        else:
            source_dir = output_dir / "src"
        
        source_dir.mkdir(parents=True, exist_ok=True)
        source_path = source_dir / f"{class_name}.cpp"
        
        # Build source content
        content = f'#include "{class_name}.h"\n\n'
        
        for func in functions:
            content += f"// Offset: 0x{func.offset}\n"
            content += func.processed_code
            content += "\n\n"
        
        source_path.write_text(content, encoding='utf-8')
        
        return source_path
