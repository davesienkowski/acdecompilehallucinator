#!/usr/bin/env python3
"""
Decompiled C++ Preprocessing Tool
=================================
Prepares decompiled C++ files for LLM processing:
- Parses header/source files and populates the database
- Analyzes dependency graph for processing order
- Provides stats on what's available

By default, runs both parse and analyze operations.
"""
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

from code_parser import (
    DatabaseHandler, DependencyAnalyzer, 
    HeaderParser, TypeWriter,
    ConstantsParser, ConstantReplacer
)
from code_parser.source_parser import SourceParser
from code_parser.offset_mapper import OffsetMapper


# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cpp-preprocessor")


# ────────────────────────────────────────────────────────────────────────────────
# Processing Modes
# ────────────────────────────────────────────────────────────────────────────────
def cmd_parse(args):
    """Parse decompiled header/source files and populate the database"""
    logger.info("Parsing decompiled source files...")
    
    header_file = Path(args.header_file)
    source_file = Path(args.source_file)
    output_dir = Path(args.output_dir)
    
    # Validate input files
    if not header_file.exists():
        logger.error(f"Header file not found: {header_file}")
        return None
    if not source_file.exists():
        logger.error(f"Source file not found: {source_file}")
        return None
    
    # Load offset mapper if available
    lines_file = Path(args.lines_file) if args.lines_file else Path("lines.txt")
    if lines_file.exists():
        logger.info(f"Loading offset mapper from: {lines_file}")
        offset_mapper = OffsetMapper(str(lines_file))
    else:
        logger.warning(f"No lines.txt found, offset mapping disabled")
        offset_mapper = None
    
    # Parse header file
    logger.info(f"Parsing header: {header_file}")
    header_parser = HeaderParser(str(header_file))
    header_parser.parse()
    
    # Parse source file
    logger.info(f"Parsing source: {source_file}")
    if offset_mapper:
        source_parser = SourceParser(str(source_file), offset_mapper)
        source_parser.parse()
    else:
        # Create a dummy offset mapper for source parsing
        class DummyMapper:
            def get_filename(self, offset): return None
        source_parser = SourceParser(str(source_file), DummyMapper())
        source_parser.parse()
    
    # Parse constants if provided
    replacer = None
    if args.constants_file:
        constants_file = Path(args.constants_file)
        if constants_file.exists():
            # Need database usage to store constants
            db_path = output_dir / 'types.db' 
            # Note: We rely on type writer to init DB, but we need it for constants now
            # So we initialize DB handler here if it doesn't exist
            # But TypeWriter will clobber output dir!
            # We should initialize TypeWriter first, which prepares the dir and DB
            
            # Wait, TypeWriter wipes the directory. We should init it first.
            pass # Logic moved below
        else:
            logger.error(f"Constants file not found: {constants_file}")

    # Write to files and database
    logger.info(f"Writing output to: {output_dir}")
    writer = TypeWriter(str(output_dir), use_database=True)
    
    # Now that TypeWriter has initialized usage, we can use the DB
    if args.constants_file and Path(args.constants_file).exists():
        logger.info(f"Parsing constants: {args.constants_file}")
        const_parser = ConstantsParser(writer.db_handler)
        const_parser.parse_file(str(args.constants_file))
        
        # Initialize replacer
        replacer = ConstantReplacer(writer.db_handler)
        logger.info("Constant replacement enabled")

    writer.write_typedefs(header_parser.typedefs)
    writer.write_structs(header_parser.structs)
    writer.write_enums(header_parser.enums, header_parser.structs)
    writer.write_funcs(source_parser.methods, header_parser.structs, replacer=replacer)
    
    # Print stats
    print("\n=== Parsing Statistics ===")
    print("Header:")
    header_parser.print_stats()
    print("\nSource:")
    source_parser.print_stats()
    
    db_path = output_dir / 'types.db'
    logger.info(f"Database created: {db_path}")
    
    return DatabaseHandler(str(db_path))


def cmd_analyze(db: DatabaseHandler, args):
    """Analyze dependencies and build processing order"""
    logger.info("Building dependency graph...")
    
    analyzer = DependencyAnalyzer(db)
    analyzer.build_dependency_graph()
    
    # Print summary
    analyzer.print_summary()
    
    # Get processing order
    order = analyzer.get_processing_order()
    
    # Save to file
    order_file = Path(args.output_dir) / "processing_order.json"
    order_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(order_file, 'w') as f:
        json.dump([{"name": name, "kind": kind} for name, kind in order], f, indent=2)
    
    logger.info(f"Processing order saved to: {order_file}")
    logger.info(f"Total types to process: {len(order)}")
    
    # Find cycles
    cycles = analyzer.find_cycles()
    if cycles:
        logger.warning(f"Found {len(cycles)} circular dependency groups:")
        for i, cycle in enumerate(cycles[:5]):  # Show first 5
            logger.warning(f"  Cycle {i+1}: {', '.join(sorted(cycle)[:3])}...")
    
    return order


def cmd_stats(db: DatabaseHandler, args):
    """Show database statistics"""
    stats = db.get_processing_stats()
    
    # Get additional counts
    structs = db.get_structs()
    enums = db.get_enums()
    methods = db.get_methods()
    parent_classes = db.get_all_parent_classes()
    
    print("\n" + "="*60)
    print("Database Statistics")
    print("="*60)
    print(f"Structs:        {len(structs)}")
    print(f"Enums:          {len(enums)}")
    print(f"Methods:        {len(methods)}")
    print(f"Parent Classes: {len(parent_classes)}")
    print("-"*60)
    print(f"Processed Types:   {stats['processed_types']}/{stats['total_types']}")
    print(f"Processed Methods: {stats['processed_methods']}/{stats['total_methods']}")
    print("="*60 + "\n")
    
    # Show some example classes
    if parent_classes and args.debug:
        print("Sample classes with methods:")
        for cls in parent_classes[:10]:
            methods = db.get_methods_by_parent(cls)
            print(f"  {cls}: {len(methods)} methods")


def cmd_list_classes(db: DatabaseHandler, args):
    """List all classes with method counts"""
    parent_classes = db.get_all_parent_classes()
    
    print(f"\nClasses with methods ({len(parent_classes)} total):\n")
    
    for cls in sorted(parent_classes):
        methods = db.get_methods_by_parent(cls)
        processed = db.get_processed_methods_by_parent(cls)
        status = f"[{len(processed)}/{len(methods)}]" if processed else ""
        print(f"  {cls}: {len(methods)} methods {status}")


# ────────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess decompiled C++ files for LLM processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run both parse and analyze (default behavior)
  python process.py
  
  # Step 1: Parse decompiled files and create database
  python process.py parse --header acclient.h --source acclient.c
  
  # Step 2: Analyze dependencies and create processing order
  python process.py analyze
  
  # Show database stats
  python process.py stats
  
  # List all classes
  python process.py list-classes
  
After preprocessing, use mcp-format.py for LLM processing.
        """
    )
    
    parser.add_argument("mode",
        nargs='?',  # Make mode optional
        choices=['parse', 'analyze', 'stats', 'list-classes', 'default'],
        default='default',  # Default to running both parse and analyze
        help="Processing mode to run (default: run both parse and analyze)"
    )
    parser.add_argument("--output", dest="output_dir", type=Path, default=Path("./mcp-sources"),
        help="Output directory (default: ./mcp-sources)")
    parser.add_argument("--header", dest="header_file", default="acclient.h",
        help="Header file to parse (for 'parse' mode)")
    parser.add_argument("--source", dest="source_file", default="acclient.c",
        help="Source file to parse (for 'parse' mode)")
    parser.add_argument("--constants", dest="constants_file", default=None,
        help="Constants file (acclient.txt) for value replacement (for 'parse' mode)")
    parser.add_argument("--lines", dest="lines_file", default="lines.txt",
        help="Lines file for offset mapping (for 'parse' mode)")
    parser.add_argument("--debug", action="store_true",
        help="Enable debug output")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle parse mode separately (doesn't need existing DB)
    if args.mode == 'parse':
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cmd_parse(args)
        return
    elif args.mode == 'default':
        # Run both parse and analyze
        args.output_dir.mkdir(parents=True, exist_ok=True)
        db = cmd_parse(args)
        if db is None:
            logger.error("Parse failed, cannot continue with analyze")
            return 1
        
        # After parsing, run analyze
        cmd_analyze(db, args)
        return
    else:
        # For other modes, need database
        db_path = args.output_dir / 'types.db'
        
        if not db_path.exists():
            logger.error(f"Database not found: {db_path}")
            logger.error("Run 'parse' mode first to create the database.")
            return 1
        
        # Initialize database handler
        db = DatabaseHandler(str(db_path))
        
        # Run requested mode
        commands = {
            'analyze': cmd_analyze,
            'stats': cmd_stats,
            'list-classes': cmd_list_classes,
        }
        
        commands[args.mode](db, args)


if __name__ == "__main__":
    main()
