#!/usr/bin/env python3
"""
LLM Processor
=============

Main orchestrator for LLM-based code processing.
Processes classes in dependency order, generating headers and methods
one at a time, with optional debug output.
"""

import json
import logging
import re

# Module-level logger for consistent logging
logger = logging.getLogger(__name__)
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from code_parser import (
    DatabaseHandler, DependencyAnalyzer,
    ClassHeaderGenerator, FunctionProcessor, ClassAssembler,
    LLMCache, LLMClient
)
from code_parser.class_assembler import ProcessedMethod
from code_parser.llm_client import LM_STUDIO_URL


def clean_llm_output(text: str) -> str:
    """Remove common LLM output artifacts from generated code.

    Cleans up markdown code blocks, formatting artifacts, and other common
    patterns that LLMs add to code output that would break compilation.

    Args:
        text: Raw LLM output text that may contain markdown or formatting.

    Returns:
        Cleaned text with markdown code fences and formatting removed.
        Returns empty string if input is None or empty.

    Examples:
        >>> clean_llm_output("```cpp\\nint x = 1;\\n```")
        'int x = 1;'
        >>> clean_llm_output("**bold** text")
        'bold text'
    """
    if not text:
        return ""
    
    # Remove markdown code blocks
    text = re.sub(r'^```(?:cpp|c\+\+|c|\w+)?\s*\n?', '', text, flags=re.M | re.I)
    text = re.sub(r'\n?```$', '', text, flags=re.M)
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    return text.strip()


def split_namespace(full_name: str) -> tuple[Optional[str], str]:
    """Split a fully qualified C++ name into namespace and simple name.

    Follows the mcp-sources convention where namespace is the top-level
    namespace only, not the full nested namespace path.

    Args:
        full_name: Fully qualified C++ name with '::' separators.
            Example: 'Turbine::Debug::Assert'

    Returns:
        A tuple of (namespace, simple_name) where:
        - namespace: Top-level namespace or None if no namespace.
        - simple_name: The final component of the name.

    Examples:
        >>> split_namespace('Turbine::Debug::Assert')
        ('Turbine', 'Assert')
        >>> split_namespace('PlayerModule')
        (None, 'PlayerModule')
        >>> split_namespace('Turbine::ACString')
        ('Turbine', 'ACString')
    """
    if '::' in full_name:
        parts = full_name.split('::')
        return parts[0], parts[-1]
    return None, full_name


class LLMProcessor:
    """
    Main orchestrator for LLM-based code processing.
    
    Processes classes in dependency order, generating headers and methods
    one at a time, with optional debug output.
    """
    
    def __init__(self, db_path: Path, output_dir: Path,
                 debug_dir: Optional[Path] = None,
                 dry_run: bool = False,
                 force: bool = False):
        """Initialize the LLM processor with database and output configuration.

        Sets up the processing pipeline including database connection, LLM cache,
        dependency analyzer, and class assembler. LLM client and generators are
        lazy-loaded on first use to avoid unnecessary initialization.

        Args:
            db_path: Path to the types.db SQLite database containing parsed
                type definitions and methods from the decompiled source.
            output_dir: Root directory for generated C++ files. Headers go to
                output_dir/include/{Namespace}/ and sources to
                output_dir/src/{Namespace}/.
            debug_dir: Optional directory for saving debug artifacts like
                prompts and raw LLM responses. If None, debug output is disabled.
            dry_run: If True, skip LLM calls and file writes. Useful for
                previewing what would be processed without making changes.
            force: If True, reprocess types even if they already have
                generated output files. Clears previous results before processing.

        Raises:
            sqlite3.Error: If database connection fails.
        """
        self.db = DatabaseHandler(str(db_path))
        self.output_dir = Path(output_dir)
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.dry_run = dry_run
        self.force = force
        self.processed_owners = set()
        
        # Initialize cache (separate database)
        cache_path = self.db.db_path.parent / "llm_cache.db"
        self.cache = LLMCache(cache_path)
        logger.info(f"LLM Cache: {cache_path}")
        
        # Initialize components
        self.analyzer = DependencyAnalyzer(self.db)
        self.assembler = ClassAssembler(self.output_dir)
        
        # LLM client (lazy-loaded)
        self._llm_client = None
        
        # Headers and methods processors (lazy-loaded with debug support)
        self._header_gen = None
        self._func_processor = None
    
    def get_file_owner(self, type_name: str) -> str:
        """Find the top-level struct/class that owns this type's file.

        For nested types, finds the outermost struct that contains them.
        This determines which file the type's code should be written to,
        since nested types share a file with their containing type.

        Args:
            type_name: Fully qualified type name, possibly with '::' separators
                for nested types. Example: 'Turbine::Client::PlayerModule::Inner'.

        Returns:
            The name of the top-level struct that owns this type's file.
            For non-nested types, returns the input unchanged.

        Examples:
            >>> processor.get_file_owner('Turbine::Client')
            'Turbine::Client'  # If Turbine::Client is a struct
            >>> processor.get_file_owner('Turbine::Client::Inner')
            'Turbine::Client'  # If Client contains Inner
        """
        if '::' not in type_name:
            return type_name
            
        parts = type_name.split('::')
        # Crawl from top to bottom
        for i in range(1, len(parts)):
            prefix = '::'.join(parts[:i])
            # Check if prefix is a struct
            res = self.db.get_type_by_name(prefix, 'struct')
            if res:
                return prefix
        return type_name

    def get_header_path(self, type_name: str) -> Path:
        """Get the expected header file path for a type.

        Computes the output path where the header file for this type should
        be written. Handles template instantiations specially by routing them
        to a Templates/ subdirectory.

        Args:
            type_name: Fully qualified type name. For template instantiations
                like 'PStringBase<char>', uses the base template name.

        Returns:
            Path to the header file. Structure follows:
            - Regular types: output_dir/include/{Namespace}/{SimpleName}.h
            - Templates: output_dir/include/Templates/{Namespace}/{SimpleName}.h
            - No namespace: output_dir/include/{SimpleName}.h
        """
        # If it's a template instantiation, we use its base name's owner
        effective_name = type_name
        if self.header_generator.is_template_instantiation(type_name):
            effective_name = self.header_generator.get_template_base_name(type_name)
            
        owner = self.get_file_owner(effective_name)

        if self.header_generator.is_template_instantiation(type_name):
            namespace, simple_name = split_namespace(owner)
            if namespace:
                return self.output_dir / "include" / "Templates" / namespace.replace('::', '/') / f"{simple_name}.h"
            return self.output_dir / "include" / "Templates" / f"{simple_name}.h"

        namespace, simple_name = split_namespace(owner)
        if namespace:
            return self.output_dir / "include" / namespace.replace('::', '/') / f"{simple_name}.h"
        return self.output_dir / "include" / f"{simple_name}.h"

    def get_source_path(self, type_name: str) -> Path:
        """Get the expected source (.cpp) file path for a type.

        Computes the output path where the source file containing method
        implementations should be written. Template instantiations use
        their base template name for path computation.

        Args:
            type_name: Fully qualified type name. For template instantiations,
                the base template name is used to determine the path.

        Returns:
            Path to the source file. Structure follows:
            - With namespace: output_dir/src/{Namespace}/{SimpleName}.cpp
            - No namespace: output_dir/src/{SimpleName}.cpp

        Note:
            Template instantiations typically don't have separate source files
            since their implementations are in the header.
        """
        effective_name = type_name
        if self.header_generator.is_template_instantiation(type_name):
            effective_name = self.header_generator.get_template_base_name(type_name)
            
        owner = self.get_file_owner(effective_name)
        namespace, simple_name = split_namespace(owner)
        if namespace:
            return self.output_dir / "src" / namespace.replace('::', '/') / f"{simple_name}.cpp"
        return self.output_dir / "src" / f"{simple_name}.cpp"

    @property
    def llm_client(self) -> LLMClient:
        """Lazy-loaded LLM client for code generation requests.

        Creates the LLM client on first access to avoid initialization
        overhead when not needed (e.g., in dry-run mode).

        Returns:
            Configured LLMClient instance connected to LM Studio at localhost:1234
            with response caching enabled.
        """
        if self._llm_client is None:
            self._llm_client = LLMClient(cache=self.cache, db_handler=self.db)
        return self._llm_client
    
    @property
    def header_generator(self) -> ClassHeaderGenerator:
        """Lazy-loaded header generator for C++ class declarations.

        Creates the header generator on first access. In dry-run mode,
        the generator is created without an LLM client to prevent API calls.

        Returns:
            Configured ClassHeaderGenerator instance with dependency analyzer
            attached for resolving type references.
        """
        if self._header_gen is None:
            self._header_gen = ClassHeaderGenerator(
                self.db, 
                llm_client=self.llm_client if not self.dry_run else None,
                debug_dir=self.debug_dir
            )
            self._header_gen.dependency_analyzer = self.analyzer
        return self._header_gen
    
    @property
    def function_processor(self) -> FunctionProcessor:
        """Lazy-loaded function processor for C++ method modernization.

        Creates the function processor on first access. In dry-run mode,
        the processor is created without an LLM client to prevent API calls.

        Returns:
            Configured FunctionProcessor instance for converting decompiled
            method implementations to modern C++.
        """
        if self._func_processor is None:
            self._func_processor = FunctionProcessor(
                self.db,
                llm_client=self.llm_client if not self.dry_run else None,
                debug_dir=self.debug_dir
            )
        return self._func_processor
    
    def get_processing_order(self) -> List[Dict[str, str]]:
        """Get all types to process in topologically-sorted dependency order.

        Returns types ordered so that dependencies are processed before
        dependents. Checks for a cached processing_order.json first,
        otherwise builds the order from the dependency analyzer.

        Returns:
            List of dictionaries with 'name' and 'kind' keys, sorted so
            base classes and referenced types appear before their users.
            Example: [{'name': 'BaseClass', 'kind': 'struct'},
                      {'name': 'DerivedClass', 'kind': 'struct'}]
        """
        # Check for existing processing_order.json
        order_file = self.output_dir.parent / "mcp-sources" / "processing_order.json"
        if order_file.exists():
            with open(order_file) as f:
                return json.load(f)
        
        # Build fresh from analyzer
        self.analyzer.build_dependency_graph()
        order = self.analyzer.get_processing_order()
        return [{"name": name, "kind": kind} for name, kind in order]
    
    def process_enum(self, enum_name: str, pbar: Optional[tqdm] = None) -> Optional[Path]:
        """Process an enum type by copying it to a header file.

        Enums are copied directly without LLM processing since they don't
        require modernization. In dry-run mode, creates a placeholder header.

        Args:
            enum_name: Fully qualified enum name to process.
            pbar: Optional tqdm progress bar to update on completion.

        Returns:
            Path to the generated header file, or None if processing failed.
            In dry-run mode, returns path to placeholder file.
        """
        import time
        start_time = time.time()
        
        # Get enum from database
        enums = self.db.get_type_by_name(enum_name, 'enum')
        if not enums:
            logger.warning(f"Enum not found: {enum_name}")
            return None
        
        enum_row = enums[0]
        enum_code = enum_row[5] if len(enum_row) > 5 else ""
        
        if self.dry_run:
            logger.info(f"[DRY-RUN] Would copy enum: {enum_name}")
            
            # Create placeholder header file for enum in dry run mode
            header_path = self.get_header_path(enum_name)
            header_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the same logic as get_header_path to determine namespace and simple name for enums
            effective_name = enum_name
            if self.header_generator.is_template_instantiation(enum_name):
                effective_name = self.header_generator.get_template_base_name(enum_name)
            
            owner = self.get_file_owner(effective_name)
            namespace, simple_name = split_namespace(owner)
            
            if namespace:
                enum_content = f"""#pragma once
#include <cstdint>

namespace {namespace} {{
    // Placeholder enum definition for {enum_name}
    // This is a dry run placeholder
    enum class {simple_name} {{
        // TODO: Define enum values
    }};
}} // namespace {namespace}
"""
            else:
                enum_content = f"""#pragma once
#include <cstdint>

// Placeholder enum definition for {simple_name}
// This is a dry run placeholder
enum class {simple_name} {{
    // TODO: Define enum values
}};
"""
            
            with open(header_path, 'w') as f:
                f.write(enum_content)
            
            logger.info(f"[DRY-RUN] Created placeholder enum header: {header_path}")
            return header_path
        
        # Check if already exists
        header_path = self.get_header_path(enum_name)
        if header_path.exists() and not self.force:
            logger.info(f"✓ Enum: {enum_name} (already exists, skipping)")
            return header_path

        # Write enum header
        # Use the same logic as get_header_path to determine namespace and simple name
        effective_name = enum_name
        if self.header_generator.is_template_instantiation(enum_name):
            effective_name = self.header_generator.get_template_base_name(enum_name)
        
        owner = self.get_file_owner(effective_name)
        namespace_write, simple_name_write = split_namespace(owner)
        path = self.assembler.write_enum_header(
            simple_name_write,
            enum_code,
            namespace=namespace_write
        )
        if path:
            logger.info(f"✓ Enum: {enum_name} → {path.name}")
            if pbar:
                pbar.update(1)
            
            # Record successful enum processing time
            elapsed = time.time() - start_time
            if hasattr(self, '_successful_enum_times'):
                self._successful_enum_times.append(elapsed)
            else:
                self._successful_enum_times = [elapsed]
        
        return path

    def group_templates(self, order: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Group template instantiations by their base template name.

        Multiple instantiations of the same template (e.g., PStringBase<char>
        and PStringBase<wchar_t>) share a single header file. This method
        filters the processing order to include only one representative
        instantiation per template base name.

        Args:
            order: List of type dictionaries from get_processing_order().

        Returns:
            Filtered list with duplicate template instantiations removed.
            The first instantiation encountered becomes the representative
            and is marked with 'is_representative': True.
        """
        grouped_order = []
        processed_templates = set()

        for item in order:
            name = item['name']
            if self.header_generator.is_template_instantiation(name):
                base_name = self.header_generator.get_template_base_name(name)
                if base_name not in processed_templates:
                    item['name'] = name  # Keep original as representative
                    item['is_representative'] = True
                    grouped_order.append(item)
                    processed_templates.add(base_name)
                else:
                    # Skip other instantiations for header generation
                    continue
            else:
                grouped_order.append(item)
        
        return grouped_order
    
    def process_class(self, class_name: str, pbar: Optional[tqdm] = None) -> Dict[str, Any]:
        """Process a single class by name, auto-detecting its type.

        Entry point for processing any type. Determines whether the type is
        an enum or struct and delegates to the appropriate processor. For
        nested types, redirects to the file owner.

        Args:
            class_name: Fully qualified type name to process.
            pbar: Optional tqdm progress bar to update on completion.

        Returns:
            Dictionary with processing results:
            - header_path: Path to generated header, or None if failed.
            - source_path: Path to generated source, or None if enum/failed.
            - method_count: Number of methods processed (0 for enums).
        """
        # Determine owner
        owner = self.get_file_owner(class_name)
        if owner != class_name:
            if owner in self.processed_owners:
                return {"header_path": None, "source_path": None, "method_count": 0}
            logger.info(f"Redirecting {class_name} to owner {owner}")
            return self.process_class(owner, pbar=pbar)
            
        # Check if it's an enum
        enums = self.db.get_type_by_name(class_name, 'enum')
        if enums:
            path = self.process_enum(class_name, pbar=pbar)
            self.processed_owners.add(class_name)
            return {"header_path": path, "source_path": None, "method_count": 0}
        
        # Otherwise treat as struct
        result = self.process_struct(class_name, pbar=pbar)
        self.processed_owners.add(class_name)
        return result

    def process_struct(self, class_name: str, pbar: Optional[tqdm] = None) -> Dict[str, Any]:
        """Process a struct/class by generating header and modernizing methods.

        Full processing pipeline for a struct type:
        1. Analyze class structure and dependencies using LLM.
        2. Generate modern C++ header with class declaration.
        3. Process each method to modernize the implementation.
        4. Assemble source file with all processed methods.

        Handles resumption by checking for existing output files and
        skipping already-processed methods. Template instantiations skip
        separate method processing since implementations are inline.

        Args:
            class_name: Fully qualified struct/class name to process.
            pbar: Optional tqdm progress bar to update as work completes.

        Returns:
            Dictionary with processing results:
            - header_path: Path to generated header file, or None on failure.
            - source_path: Path to generated source file, or None if no methods.
            - method_count: Number of methods successfully processed.
        """
        result = {"header_path": None, "source_path": None, "method_count": 0}
        
        if self.dry_run:
            methods = self.db.get_methods_by_parent(class_name)
            logger.info(f"[DRY-RUN] Would process: {class_name} ({len(methods)} methods)")
            
            # Create placeholder header and cpp files in dry run mode
            header_path = self.get_header_path(class_name)
            source_path = self.get_source_path(class_name)
            
            # Create header file with class name
            # For templates, make sure we create the correct directory structure
            header_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the same logic as get_header_path to determine namespace and simple name
            effective_name = class_name
            if self.header_generator.is_template_instantiation(class_name):
                effective_name = self.header_generator.get_template_base_name(class_name)
            
            owner = self.get_file_owner(effective_name)
            namespace, simple_name = split_namespace(owner)
            
            if namespace:
                header_content = f"""#pragma once
#include <cstdint>

namespace {namespace} {{
    class {simple_name} {{
    public:
        // Placeholder class definition for {class_name}
        // This is a dry run placeholder
    }};
}} // namespace {namespace}
"""
            else:
                header_content = f"""#pragma once
#include <cstdint>

class {simple_name} {{
public:
    // Placeholder class definition for {simple_name}
    // This is a dry run placeholder
}};
"""
            
            with open(header_path, 'w') as f:
                f.write(header_content)
            
            logger.info(f"[DRY-RUN] Created placeholder header: {header_path}")
            
            # Create cpp file with class and method names if there are methods
            if methods:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Use the same logic as above for consistent namespace handling
                effective_name_source = class_name
                if self.header_generator.is_template_instantiation(class_name):
                    effective_name_source = self.header_generator.get_template_base_name(class_name)
                
                owner_source = self.get_file_owner(effective_name_source)
                namespace_source, simple_name_source = split_namespace(owner_source)
                
                if namespace_source:
                    cpp_content = f"""#include "{header_path.name}"
#include <iostream>

// Placeholder method implementations for {class_name}
"""
                    for method in methods:
                        # Method structure from database: [id, name, parent_class, ...]
                        method_name = method[1] if len(method) > 1 else "unknown_method"
                        
                        # Check if simple_name contains template parameters (e.g., "PStringBase<char>")
                        # If it does, we need to define the method differently to avoid invalid C++ syntax
                        if '<' in simple_name_source and '>' in simple_name_source:
                            # For template classes, we can't implement methods this way
                            # Instead, we'll just document the method
                            cpp_content += f"""// Method: {method_name}
// Note: {simple_name_source} is a template class - method {method_name} would normally be defined inline or in a specialized way
// void {namespace_source}::{simple_name_source}::{method_name}() {{
//     // TODO: Implement {method_name}
//     // Placeholder for method {method_name}
// }}

"""
                        else:
                            # Regular class, standard method implementation
                            cpp_content += f"""// Method: {method_name}
void {namespace_source}::{simple_name_source}::{method_name}() {{
    // TODO: Implement {method_name}
    // Placeholder for method {method_name}
}}

"""
                else:
                    cpp_content = f"""#include "{header_path.name}"
#include <iostream>

// Placeholder method implementations for {simple_name_source}
"""
                    for method in methods:
                        # Method structure from database: [id, name, parent_class, ...]
                        method_name = method[1] if len(method) > 1 else "unknown_method"
                        
                        # Check if simple_name contains template parameters (e.g., "PStringBase<char>")
                        # If it does, we need to define the method differently to avoid invalid C++ syntax
                        if '<' in simple_name_source and '>' in simple_name_source:
                            # For template classes, we can't implement methods this way
                            # Instead, we'll just document the method
                            cpp_content += f"""// Method: {method_name}
// Note: {simple_name_source} is a template class - method {method_name} would normally be defined inline or in a specialized way
// void {simple_name_source}::{method_name}() {{
//     // TODO: Implement {method_name}
//     // Placeholder for method {method_name}
// }}

"""
                        else:
                            # Regular class, standard method implementation
                            cpp_content += f"""// Method: {method_name}
void {simple_name_source}::{method_name}() {{
    // TODO: Implement {method_name}
    // Placeholder for method {method_name}
}}

"""
                
                with open(source_path, 'w') as f:
                    f.write(cpp_content)
                
                logger.info(f"[DRY-RUN] Created placeholder source: {source_path}")
            
            result["header_path"] = header_path
            result["source_path"] = source_path if methods else None
            result["method_count"] = len(methods)
            
            return result
        
        logger.info(f"┌─ Processing: {class_name}")

        if self.force:
            logger.info(f"│  └─ Force enabled: Clearing previous results...")
            self.db.clear_processed_class(class_name)

        # Skip if header already exists
        header_path = self.get_header_path(class_name)
        source_path = self.get_source_path(class_name)
        is_template = self.header_generator.is_template_instantiation(class_name)
        
        # Check if output files are missing and force regeneration if so
        if not self.force:
            header_missing = not header_path.exists()
            source_missing = not is_template and not source_path.exists()
            
            if header_missing or source_missing:
                reason = "Header missing" if header_missing else "Source missing"
                if header_missing and source_missing: reason = "Header and Source missing"
                logger.info(f"│  └─ {reason}: Triggering regeneration...")
                self.db.clear_processed_class(class_name)
        
        analysis = None
        extracted_types = []
        
        if header_path.exists() and not self.force:
            logger.info(f"│  └─ Header already exists: {header_path.relative_to(self.output_dir.parent)}")
            result["header_path"] = header_path
        else:
            # Step 0: Analysis
            logger.info(f"│  └─ Analyzing class...")
            try:
                import time
                start_time = time.time()
                analysis_result = self.header_generator.analyze_class(class_name)
                if analysis_result:
                    logger.info(f"│  └─ ✓ Analysis complete")
                    # Parse the analysis result to extract referenced types
                    import json
                    try:
                        analysis_data = json.loads(analysis_result)
                        analysis = analysis_data.get("analysis", "")
                        extracted_types = analysis_data.get("referenced_types", [])
                        logger.info(f"│  └─ Found {len(extracted_types)} referenced types from analysis")
                        
                        # Record successful analysis time
                        elapsed = time.time() - start_time
                        if hasattr(self, '_successful_analysis_times'):
                            self._successful_analysis_times.append(elapsed)
                        else:
                            self._successful_analysis_times = [elapsed]
                            
                    except json.JSONDecodeError:
                        # If JSON parsing fails, use the original analysis
                        analysis = analysis_result
                        extracted_types = []
                else:
                    logger.warning(f"│  └─ ✗ Analysis failed (empty)")
            except Exception as e:
                logger.error(f"│  └─ ✗ Analysis error: {e}")

            # Step 1: Generate header
            logger.info(f"│  └─ Generating header...")
            
            try:
                import time
                start_time = time.time()
                header_code = self.header_generator.generate_header(class_name, save_to_db=True, analysis=analysis)
                if header_code:
                    header_code = clean_llm_output(header_code)
                    # Use the same logic as get_header_path to determine namespace and simple name
                    effective_name = class_name
                    if self.header_generator.is_template_instantiation(class_name):
                        effective_name = self.header_generator.get_template_base_name(class_name)
                    
                    owner = self.get_file_owner(effective_name)
                    namespace_write, simple_name_write = split_namespace(owner)
                    result["header_path"] = self.assembler.write_header_file(
                        simple_name_write,
                        header_code,
                        namespace=namespace_write,
                        path=header_path
                    )
                    logger.info(f"│  └─ ✓ Header saved")
                    if pbar:
                        pbar.update(1)
                    
                    # Record successful header generation time
                    elapsed = time.time() - start_time
                    if hasattr(self, '_successful_header_times'):
                        self._successful_header_times.append(elapsed)
                    else:
                        self._successful_header_times = [elapsed]
                        
                else:
                    logger.warning(f"│  └─ ✗ Header generation failed")
            except Exception as e:
                logger.error(f"│  └─ ✗ Header error: {e}")
        
        # Step 2: Process methods one at a time (SKIP for templates)
        if self.header_generator.is_template_instantiation(class_name):
            logger.info(f"│  └─ Template instantiation: skipping separate method processing and source assembly")
            logger.info(f"└─ Done: {class_name}")
            return result

        all_methods = []
        unprocessed_methods = []
        
        # Gather all types that belong to this file
        nested_types = self.db.get_nested_types(class_name)
        file_types = [class_name] + [nt[2] for nt in nested_types]
        
        for t in file_types:
            all_methods.extend(self.db.get_methods_by_parent(t))
            unprocessed_methods.extend(self.db.get_unprocessed_methods(parent_class=t))
            
        source_path = self.get_source_path(class_name)
        
        # Check if we should skip method processing
        if not unprocessed_methods and source_path.exists() and not self.force:
            logger.info(f"│  └─ Source already exists and all methods processed: {source_path.relative_to(self.output_dir.parent)}")
            result["source_path"] = source_path
            result["method_count"] = len(all_methods)
            logger.info(f"└─ Done: {class_name} (skipped)")
            return result

        if unprocessed_methods:
            logger.info(f"│  └─ Processing {len(unprocessed_methods)} methods for {len(file_types)} classes...")
        
        for method in unprocessed_methods:
            method_name = method[1]
            
            try:
                import time
                start_time = time.time()
                logger.info(f"│     └─ {method_name}...")
                self.function_processor.process_function(method, save_to_db=True, analysis=analysis)
                result["method_count"] += 1
                if pbar:
                    pbar.update(1)
                
                # Record successful method processing time
                elapsed = time.time() - start_time
                if hasattr(self, '_successful_method_times'):
                    self._successful_method_times.append(elapsed)
                else:
                    self._successful_method_times = [elapsed]
                
            except Exception as e:
                logger.error(f"│     └─ ✗ {method_name}: {e}")
        
        # Step 3: Assemble source file (no LLM) - Gather ALL processed methods
        # This ensures that even if we resumed, we include methods from previous runs
        processed_db_methods = []
        for t in file_types:
            processed_db_methods.extend(self.db.get_processed_methods_by_parent(t))

        if processed_db_methods:
            logger.info(f"│  └─ Assembling source with {len(processed_db_methods)} methods...")
            
            processed_methods = []
            for pm in processed_db_methods:
                processed_methods.append(ProcessedMethod(
                    name=pm['name'],
                    full_name=pm['full_name'],
                    parent_class=pm['parent_class'],
                    processed_code=clean_llm_output(pm['processed_code']),
                    dependencies=pm.get('dependencies', []),
                    offset=pm.get('offset', '0')
                ))

            # Use the same logic as get_header_path to determine namespace and simple name
            effective_name = class_name
            if self.header_generator.is_template_instantiation(class_name):
                effective_name = self.header_generator.get_template_base_name(class_name)
            
            owner = self.get_file_owner(effective_name)
            namespace_write, simple_name_write = split_namespace(owner)
            result["source_path"] = self.assembler.write_source_file(
                simple_name_write,
                processed_methods,
                namespace=namespace_write
            )
            if result["source_path"]:
                logger.info(f"│  └─ ✓ Source saved")
        
        logger.info(f"└─ Done: {class_name}")
        return result
    
        # Otherwise treat as struct
        return self.process_struct(class_name, pbar=pbar)

    def calculate_work_units(self, order: List[Dict[str, str]]) -> Dict[str, Any]:
        """Calculate the number of work units (headers + methods) needing processing.

        Scans the processing order to determine how much work remains by
        checking which output files exist and which methods are already
        processed. Used for accurate progress bar initialization.

        Args:
            order: List of type dictionaries from get_processing_order().

        Returns:
            Dictionary with work unit counts:
            - headers: Number of header files to generate.
            - methods: Number of methods to process.
            - total: Sum of headers and methods (for progress bar).
        """
        total_headers = 0
        total_methods = 0
        processed_owners = set()
        
        logger.info("Calculating total work units...")
        
        for type_info in order:
            name = type_info["name"]
            owner = self.get_file_owner(name)
            
            if owner in processed_owners:
                continue
            processed_owners.add(owner)
            
            kind = type_info["kind"] # Note: This kind might be for the nested type, but we care about the owner's kind
            # Let's get the owner's kind
            owner_rows = self.db.get_type_by_name(owner)
            owner_kind = owner_rows[0][1] if owner_rows else kind
            
            if owner_kind == "enum":
                header_path = self.get_header_path(owner)
                if not header_path.exists() or self.force:
                    total_headers += 1
            else:
                # Struct/Class
                header_path = self.get_header_path(owner)
                source_path = self.get_source_path(owner)
                is_template = self.header_generator.is_template_instantiation(owner)
                
                header_missing = not header_path.exists()
                source_missing = not is_template and not source_path.exists()
                
                needs_regeneration = self.force or header_missing or source_missing
                
                if needs_regeneration:
                    total_headers += 1
                
                if is_template:
                    continue
                
                # Gather methods for owner and all nested types
                nested_types = self.db.get_nested_types(owner)
                file_types = [owner] + [nt[2] for nt in nested_types]
                
                for t in file_types:
                    if needs_regeneration:
                        all_methods = self.db.get_methods_by_parent(t)
                        total_methods += len(all_methods)
                    else:
                        unprocessed_methods = self.db.get_unprocessed_methods(parent_class=t)
                        total_methods += len(unprocessed_methods)
                    
        return {
            "headers": total_headers,
            "methods": total_methods,
            "total": total_headers + total_methods
        }

    def process_all_internal(self, filter_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process all types in dependency order with progress tracking.

        Internal implementation of batch processing. Iterates through the
        processing order, handling each type and tracking statistics.
        Displays a progress bar with adaptive ETA based on average
        processing times.

        Args:
            filter_classes: Optional list of class names to process.
                If provided, only these types are processed; others are skipped.
                Useful for targeted reprocessing of specific classes.

        Returns:
            Dictionary with processing statistics:
            - total: Total number of types in processing order.
            - enums_processed: Count of successfully processed enums.
            - structs_processed: Count of successfully processed structs.
            - methods_processed: Total methods processed across all structs.
            - errors: List of error dictionaries with 'name' and 'error' keys.
        """
        order = self.get_processing_order()
        
        # Group templates before processing
        order = self.group_templates(order)
        
        stats = {
            "total": len(order),
            "enums_processed": 0,
            "structs_processed": 0,
            "methods_processed": 0,
            "errors": []
        }
        
        # Filter if requested
        if filter_classes:
            order = [t for t in order if t["name"] in filter_classes]
            logger.info(f"Filtered to {len(order)} types: {[t['name'] for t in order]}")
        
        # Calculate work units for better progress estimation
        work_plan = self.calculate_work_units(order)
        logger.info(f"Plan: {work_plan['headers']} headers, {work_plan['methods']} methods to process.")
        
        if work_plan["total"] == 0:
            logger.info("Nothing to process.")
            return stats

        logger.info(f"Processing {len(order)} types in dependency order...")
        
        # Track successful operation times for averaging
        successful_times = []
        
        # Initialize timing lists for different operations
        self._successful_header_times = []
        self._successful_method_times = []
        self._successful_enum_times = []
        self._successful_analysis_times = []
        
        with tqdm(total=work_plan["total"], desc="Processing", unit="task") as pbar:
            for type_info in order:
                name = type_info["name"]
                
                # Skip if owner already processed
                owner = self.get_file_owner(name)
                if owner in self.processed_owners:
                    continue
                    
                kind = type_info["kind"]
                try:
                    import time
                    start_time = time.time()
                    
                    if kind == "enum":
                        self.process_enum(name, pbar=pbar)
                        stats["enums_processed"] += 1
                    else:
                        result = self.process_struct(name, pbar=pbar)
                        if result.get("header_path"):
                            stats["structs_processed"] += 1
                        stats["methods_processed"] += result.get("method_count", 0)
                        
                    # Record successful operation time
                    elapsed = time.time() - start_time
                    successful_times.append(elapsed)
                
                except Exception as e:
                    logger.error(f"Failed to process {name}: {e}")
                    stats["errors"].append({"name": name, "error": str(e)})
                    
                # Update tqdm description with custom ETA based on average time
                # Combine all successful times (headers, methods, enums, analysis)
                all_successful_times = successful_times.copy()
                if hasattr(self, '_successful_header_times'):
                    all_successful_times.extend(self._successful_header_times)
                if hasattr(self, '_successful_method_times'):
                    all_successful_times.extend(self._successful_method_times)
                if hasattr(self, '_successful_enum_times'):
                    all_successful_times.extend(self._successful_enum_times)
                if hasattr(self, '_successful_analysis_times'):
                    all_successful_times.extend(self._successful_analysis_times)
                
                if all_successful_times:
                    avg_time = sum(all_successful_times) / len(all_successful_times)
                    remaining_tasks = work_plan["total"] - pbar.n
                    eta_seconds = avg_time * remaining_tasks
                    
                    # Convert to human-readable format
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.0f}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    
                    pbar.set_postfix({"ETA avg": eta_str, "avg_time": f"{avg_time:.1f}s"})
        
        return stats
    
    def process_all(self, filter_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process all types in dependency order with safety checks.

        Public entry point for batch processing. Wraps process_all_internal
        with safety checks to prevent accidental mass deletion of existing
        output when force mode is enabled without a filter.

        Args:
            filter_classes: Optional list of specific class names to process.
                If provided, only these types are processed. Required when
                using force mode for safety reasons.

        Returns:
            Dictionary with processing statistics from process_all_internal().

        Note:
            The force flag is ignored for bulk processing (no filter_classes)
            as a safety measure to prevent accidental reprocessing of the
            entire codebase.
        """
        # Note: We don't support force for process_all to prevent accidental mass deletion
        # unless filter_classes is provided
        
        if self.force and not filter_classes:
            logger.warning("Force flag ignored for bulk processing (safety check)")
            self.force = False
            
        stats = self.process_all_internal(filter_classes)
        return stats
    
    def show_plan(self):
        """Display a preview of the processing plan without executing.

        Logs a summary of what would be processed, including counts of
        enums and structs, and details for the first 20 types in the
        processing order. Useful for reviewing the plan before starting
        a long-running batch process.
        """
        order = self.get_processing_order()

        logger.info("=" * 60)
        logger.info(f"Processing Plan - {len(order)} types")
        logger.info("=" * 60)

        enum_count = sum(1 for t in order if t["kind"] == "enum")
        struct_count = len(order) - enum_count

        logger.info(f"Enums:   {enum_count}")
        logger.info(f"Structs: {struct_count}")
        logger.info("-" * 60)

        for i, type_info in enumerate(order[:20], 1):
            name = type_info["name"]
            kind = type_info["kind"]

            if kind == "enum":
                logger.info(f"{i:3}. [ENUM] {name}")
            else:
                methods = self.db.get_methods_by_parent(name)
                logger.info(f"{i:3}. [STRUCT] {name} ({len(methods)} methods)")

        if len(order) > 20:
            logger.info(f"... and {len(order) - 20} more types")

        logger.info("=" * 60)