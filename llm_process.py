#!/usr/bin/env python3
"""
LLM Processing Pipeline
=======================
Processes decompiled C++ classes through LLM in dependency order.

Workflow:
1. Load processing order from database/JSON
2. For each class (in dependency order):
   - Generate header via LLM
   - Process each method via LLM (one at a time)
   - Assemble final .cpp from processed methods (no LLM)
3. Output debug files showing prompts, responses, and types

Usage:
    python llm_process.py --db mcp-sources/types.db --output ./output
    python llm_process.py --db mcp-sources/types.db --class PlayerModule --debug
    python llm_process.py --db mcp-sources/types.db --dry-run
"""
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from code_parser import (
    DatabaseHandler, DependencyAnalyzer,
    ClassHeaderGenerator, FunctionProcessor, ClassAssembler,
    LLMCache, LLMProcessor, LLMClient
)
from code_parser.class_assembler import ProcessedMethod


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("llm-processor")

# ────────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Process decompiled C++ through LLM in dependency order",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all types
    python llm_process.py --db mcp-sources/types.db --output ./output
    
    # Process single class with debug output
    python llm_process.py --class AllegianceProfile --debug
    
    # Show processing plan (dry-run)
    python llm_process.py --dry-run
        """
    )
    
    parser.add_argument("--db", type=Path, default=Path("mcp-sources/types.db"),
        help="Path to types.db database")
    parser.add_argument("--output", type=Path, default=Path("./output"),
        help="Output directory for generated files")
    parser.add_argument("--debug", action="store_true",
        help="Enable debug output (prompts, responses, types)")
    parser.add_argument("--debug-dir", type=Path, default=None,
        help="Debug output directory (default: <output>/debug)")
    parser.add_argument("--class", dest="single_class", type=str, default=None,
        help="Process only this class")
    parser.add_argument("--dry-run", action="store_true",
        help="Show what would be processed without calling LLM")
    parser.add_argument("--force", action="store_true",
        help="Force re-processing (clears previous results for this class)")
    parser.add_argument("--verbose", "-v", action="store_true",
        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate database
    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        logger.error("Run 'python process.py parse' first to create it.")
        return 1
    
    # Set up debug directory
    debug_dir = None
    if args.debug:
        debug_dir = args.debug_dir or (args.output / "debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug output: {debug_dir}")
    
    # Create processor
    processor = LLMProcessor(
        db_path=args.db,
        output_dir=args.output,
        debug_dir=debug_dir,
        dry_run=args.dry_run,
        force=args.force
    )
    
    # Handle modes
    if args.dry_run and not args.single_class:
        processor.show_plan()
        # In dry run mode, also process all classes to create placeholder files
        logger.info("Processing all classes in dry run mode to create placeholder files...")
        stats = processor.process_all()
        print(f"\n{'='*60}")
        print("Dry Run Complete - Placeholder files created")
        print('='*60)
        print(f"Enums processed:   {stats['enums_processed']}")
        print(f"Structs processed: {stats['structs_processed']}")
        print(f"Methods processed: {stats['methods_processed']}")
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
        print('='*60 + "\n")
        return 0
    
    if args.single_class:
        result = processor.process_class(args.single_class)
        print(f"\nProcessed: {args.single_class}")
        print(f"  Header: {result.get('header_path', 'None')}")
        print(f"  Source: {result.get('source_path', 'None')}")
        print(f"  Methods: {result.get('method_count', 0)}")
    else:
        stats = processor.process_all()
        print(f"\n{'='*60}")
        print("Processing Complete")
        print('='*60)
        print(f"Enums processed:   {stats['enums_processed']}")
        print(f"Structs processed: {stats['structs_processed']}")
        print(f"Methods processed: {stats['methods_processed']}")
        if stats['errors']:
            print(f"Errors: {len(stats['errors'])}")
        print('='*60 + "\n")
    
    return 0


if __name__ == "__main__":
    exit(main())
