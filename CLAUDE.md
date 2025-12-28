# CLAUDE.md

C++ code modernization pipeline for Asheron's Call client decompilation.

## Quick Start

```bash
# Parse decompiled files into database
python process.py parse --header acclient.h --source acclient.c

# Process with LLM (requires LM Studio at localhost:1234)
python llm_process.py

# Check progress
python process.py stats
```

## Project Structure

```
.
├── process.py              # Preprocessing CLI
├── llm_process.py          # LLM orchestration
├── code_parser/
│   ├── llm_processor.py    # Main orchestrator
│   ├── llm_client.py       # LLM API wrapper
│   ├── header_parser.py    # Parse acclient.h
│   ├── source_parser.py    # Parse acclient.c
│   ├── db_handler.py       # SQLite operations
│   ├── dependency_analyzer.py  # Topological sort
│   ├── class_header_generator.py
│   ├── function_processor.py
│   └── class_assembler.py
├── .claude/
│   ├── rules/              # Modular instructions
│   │   ├── python-style.md
│   │   ├── cpp-output.md
│   │   ├── workflow.md
│   │   └── database.md
│   └── settings.json       # Project permissions
└── output/                 # Generated C++ code
    ├── include/{Namespace}/{Class}.h
    └── src/{Namespace}/{Class}.cpp
```

## Migration Status

| Phase | Description | PR |
|-------|-------------|-----|
| 0 | Engine Abstraction | [#3](https://github.com/trevis/ACDecompileHallucinator/pull/3) |
| 1 | Engine Tracking | [#4](https://github.com/trevis/ACDecompileHallucinator/pull/4) |
| 2 | Type Hints | [#5](https://github.com/trevis/ACDecompileHallucinator/pull/5) |
| 2 | TypeResolver | [#8](https://github.com/trevis/ACDecompileHallucinator/pull/8) |
| 3 | Subprocess Engine | [#6](https://github.com/trevis/ACDecompileHallucinator/pull/6) |
| 3 | Claude Skills | [#7](https://github.com/trevis/ACDecompileHallucinator/pull/7) |

## Key Files

- **Input**: `acclient.h`, `acclient.c`, `acclient.txt` (constants)
- **Database**: `types.db` (parsed types, methods, processing state)
- **Cache**: `llm_cache.db` (LLM response cache)

## Rules Reference

Detailed instructions in `.claude/rules/`:
- [python-style.md](.claude/rules/python-style.md) - Python conventions
- [cpp-output.md](.claude/rules/cpp-output.md) - C++ generation rules
- [workflow.md](.claude/rules/workflow.md) - Processing pipeline
- [database.md](.claude/rules/database.md) - Database schema

## Commit Guidelines

- Conventional commits: `type: description`
- Types: feat, fix, refactor, docs, test, chore
- No AI attribution in commit messages
