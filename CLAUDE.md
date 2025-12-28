# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

A C++ code modernization pipeline that transforms raw decompiled Asheron's Call client code (`acclient.c` / `acclient.h`) into clean, modern C++17+ code.

## Architecture: Hybrid Approach

This project uses a **hybrid architecture** with backwards-compatible engine abstraction:

```
python llm_process.py --engine <lm-studio|claude-code> [options]
```

### Key Decisions

| Decision | Choice |
|----------|--------|
| Backwards compatibility | LM Studio continues to work unchanged |
| Engine selection | CLI flag `--engine` |
| Legacy improvements | Comprehensive refactoring (types, docs, errors) |
| Extensibility | Plugin architecture for future LLM backends |
| Refactoring approach | In-place updates preserving git history |

### Engine Abstraction

```
engines/
├── __init__.py       # Exports and factory
├── base.py           # Abstract LLMEngine interface
├── lm_studio.py      # Local LM Studio backend
├── claude_code.py    # Claude Code CLI/skills backend
└── registry.py       # Plugin discovery
```

Both engines share:
- `types.db` for parsed code and state tracking
- `output/` directory structure
- Same verification and quality standards

## Project Structure

```
.
├── acclient/               # Decompiled source files (gitignored)
│   ├── acclient.c          # Method implementations
│   └── acclient.h          # Structs, enums, typedefs
├── code_parser/            # Python parsing modules (18 files)
│   ├── header_parser.py    # Parses acclient.h
│   ├── source_parser.py    # Parses acclient.c
│   ├── db_handler.py       # SQLite operations
│   └── ...
├── engines/                # LLM backend abstraction (to be created)
├── skills/                 # Claude Code skills (to be created)
├── output/                 # Generated modern C++ (gitignored)
├── process.py              # Preprocessing CLI
├── llm_process.py          # LLM orchestration
├── HYBRID_MIGRATION_PLAN.md # Detailed implementation plan
└── CLAUDE.md               # This file
```

## Quick Commands

### Preprocessing (Python)
```bash
# Parse decompiled files into SQLite database
python process.py parse --header acclient/acclient.h --source acclient/acclient.c

# Show database statistics
python process.py stats

# List all classes with method counts
python process.py list-classes
```

### Processing (after engine implementation)
```bash
# Process with LM Studio (legacy, default)
python llm_process.py --engine lm-studio

# Process with Claude Code
python llm_process.py --engine claude-code

# Process specific class
python llm_process.py --engine claude-code --class Player
```

## Modernization Rules

When generating modern C++ code:

### Header Generation
- Use `#pragma once` instead of include guards
- Convert `struct __cppobj` to `class` with proper access specifiers
- Preserve inheritance and member order
- Add minimal includes, prefer forward declarations
- Do NOT inline function definitions
- Do NOT rename classes or methods

### Method Modernization
- Remove decompiler artifacts (`__thiscall`, `__cdecl`, explicit `this` pointer)
- Use modern types (`uint32_t`, `bool` instead of `int` for flags)
- Rename local variables for readability (but not parameters)
- Add comments explaining logic
- Preserve ALL original logic - no additions or removals
- Use enum names instead of magic values where known

### Verification
Every modernized method must preserve original logic:
- Compare control flow structure
- Verify arithmetic operations unchanged
- Check that all side effects are preserved
- **Allow**: variable renaming, type modernization, style changes
- **Reject**: logic changes, missing operations, added functionality

## Database Schema (types.db)

```sql
-- Parsed types from acclient.h
types(name, code, namespace, parent, is_template)

-- Parsed methods from acclient.c
methods(name, definition, parent_class, offset)

-- Processing state (shared by all engines)
processed_types(name, processed_header, engine_used, timestamp)
processed_methods(name, parent, processed_code, engine_used, confidence)

-- Named constants from acclient.txt
constants(name, value, type)
```

## Output Structure

```
output/
├── include/           # Generated .h headers
│   └── {Namespace}/
│       └── {ClassName}.h
└── src/               # Generated .cpp sources
    └── {Namespace}/
        └── {ClassName}.cpp
```

## Implementation Phases

See [HYBRID_MIGRATION_PLAN.md](HYBRID_MIGRATION_PLAN.md) for detailed implementation.

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Engine abstraction layer | Planned |
| 1 | LM Studio engine extraction | Planned |
| 2 | Legacy code refactoring | Planned |
| 3 | Claude Code engine + skills | Planned |
| 4 | Advanced features | Future |

## Code Quality Standards (for refactoring)

When refactoring the Python codebase:

### Type Hints
```python
def get_type_by_name(self, name: str) -> Optional[TypeRecord]:
    """Retrieve a type record by its fully qualified name."""
```

### Error Handling
```python
try:
    result = self.engine.generate_header(...)
except EngineError as e:
    logger.error(f"Failed to generate header: {e}")
    raise
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"Processing class: {class_name}")
logger.debug(f"Class has {len(methods)} methods")
```

## MCP Integration

### Serena MCP
For semantic code analysis of the decompiled sources:
- `find_symbol` - Locate classes, methods, types
- `get_symbols_overview` - File structure analysis
- `find_referencing_symbols` - Dependency tracking
- `create_text_file` - Write modernized output

### claude-mem
For cross-session state persistence:
- Track processed classes
- Store learned patterns
- Resume interrupted work
