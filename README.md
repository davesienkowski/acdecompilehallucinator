**Warning:** This project is pure vibes

# Decompiled C++ Modernization Pipeline

This toolchain transforms raw decompiled C++ code (specifically `acclient.c` and `acclient.h`) into clean, modern C++17+ code using an LLM-based pipeline. It handles dependency resolution, incremental processing, and structured code generation.

## Prerequisites

- **Python 3.10+**
- **LM Studio** (or OpenAI-compatible API) running on `http://localhost:1234/v1`
  - Recommended model: A strong coding model (e.g., DeepSeek Coder, CodeLlama)
- **Dependencies**: Install via `pip install -r requirements.txt` (if available) or install `openai tqdm tenacity`.

## Workflow Overview

The pipeline consists of three main stages:
1.  **Preprocessing**: Parsing the monolithic decompiled files into a structured SQLite database.
2.  **Analysis**: Determining the dependency order to ensure base classes and types are generated before their dependents.
3.  **Modernization**: Using an LLM to generate clean headers and implementations for each class.

---

## Step-by-Step Guide

### 1. Preparation

Ensure your decompiled source files are in the project root:
- `acclient.c`: The monolithic source file.
- `acclient.h`: The accompanying header file.

### 2. Database Creation (Preprocessing)

Parse the raw files to populate the `types.db` database. This extracts all structs, enums, methods, and typedefs.

Analyze the database to build a processing order. This prevents "incomplete type" errors in generated C++ by ensuring dependencies are processed first.

```bash
# Basic usage
python process.py parse --header acclient.h --source acclient.c
```

**Output**:
  - Creates `mcp-sources/types.db`.
  - Creates `mcp-sources/processing_order.json`.

### 3. Modernization (LLM Processing)

Run the LLM processor to generate modern C++ code. This tool iterates through the processing order, generating a header and source file for each class.

```bash
# Process all classes in dependency order
python llm_process.py --db mcp-sources/types.db --output ./output

# Dry run (see what would be processed without calling LLM)
python llm_process.py --dry-run
```

**Output**:
- `output/include/`: Generated `.h` header files.
- `output/src/`: Generated `.cpp` source files.
- `output/debug/`: (Optional) Intermediate prompts and responses for debugging.

### Debugging & Selective Processing

You can process individual classes for testing or debugging:

```bash
# Process a single class with debug logging
python llm_process.py --class Player --debug

# Force re-processing of a specific class (ignoring cache/status)
python llm_process.py --class Player --force
```

## Tools Reference

- **`process.py`**: Handles parsing and dependency analysis.
  - `parse`: Populates DB, Generates build order.
  - `stats`: Shows DB statistics.
  - `list-classes`: Lists all detected classes.
- **`llm_process.py`**: Handles the LLM generation loop.
  - Orchestrates `ClassHeaderGenerator` and `FunctionProcessor`.
  - Manages context and prompts.
