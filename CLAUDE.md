# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a C++ code modernization pipeline that transforms raw decompiled Asheron's Call client code (`acclient.c` / `acclient.h`) into clean, modern C++17+ code.

**Migration Status**: Migrating from local LLM (LM Studio + Python scripts) to a Claude Code native architecture using:
- **Claude Code** as the primary orchestrator
- **Claude Code Skills** for custom modernization workflows
- **Claude Code Agents** for parallel class processing
- **Serena MCP** for semantic code analysis and manipulation
- **claude-mem** for cross-session memory of processing state and learned patterns

## Legacy System (local-llm-code/)

The original Python-based pipeline in `local-llm-code/` remains for reference:

### Preprocessing Commands (Still Useful)
```bash
cd local-llm-code

# Parse decompiled files into SQLite database
python process.py parse --header acclient.h --source acclient.c

# Show database statistics
python process.py stats

# List all classes with method counts
python process.py list-classes
```

### Legacy Architecture
```
local-llm-code/
├── process.py              # Preprocessing CLI (parse, analyze, stats)
├── llm_process.py          # OLD: LM Studio orchestration (TO BE REPLACED)
└── code_parser/
    ├── header_parser.py    # Parses acclient.h → structs, enums, typedefs
    ├── source_parser.py    # Parses acclient.c → method bodies
    ├── db_handler.py       # SQLite operations for types.db
    ├── dependency_analyzer.py  # Topological sort with cycle detection
    └── ...
```

## New Claude Code Architecture

### Core Concept

Instead of Python calling an LLM API, Claude Code itself becomes the orchestrator:

1. **Serena MCP** replaces regex-based code parsing with semantic understanding
   - `find_symbol` to locate classes, methods, types
   - `get_symbols_overview` for file structure
   - `replace_symbol_body` for surgical code edits
   - `find_referencing_symbols` for dependency tracking

2. **Claude Code Skills** replace the hardcoded prompts in Python
   - Custom `/modernize-class` skill with few-shot examples
   - Custom `/modernize-method` skill with verification logic
   - Custom `/analyze-dependencies` skill for processing order

3. **Claude Code Agents** replace the sequential Python loop
   - Parallel processing of independent classes
   - Specialized agents for header vs source generation

4. **claude-mem** replaces the SQLite processed_* tables
   - Persistent memory of which classes are processed
   - Store learned patterns and conventions
   - Cross-session continuity

### Proposed Skill Structure

```
skills/
├── modernize-class.md      # Generate modern header for a class
├── modernize-method.md     # Modernize a single method with verification
├── analyze-deps.md         # Build dependency graph for processing order
├── process-batch.md        # Process multiple classes in parallel
└── verify-logic.md         # Verify modernized code preserves logic
```

### Processing Workflow

1. **Initialization**: Load `types.db` context via Serena, check claude-mem for prior progress
2. **Dependency Analysis**: Use Serena to trace type references, build processing order
3. **Class Processing** (parallelizable via agents):
   - Read class definition with Serena `find_symbol`
   - Generate modern header using skill prompts
   - Process each method with verification loop
   - Write output via Serena `create_text_file`
4. **State Persistence**: Update claude-mem with completed classes

### Serena MCP Usage

For semantic code analysis:
```
# Get overview of a parsed header
mcp__plugin_serena_serena__get_symbols_overview(relative_path="acclient.h")

# Find a specific class
mcp__plugin_serena_serena__find_symbol(name_path_pattern="Player", include_body=True)

# Find all references to a type
mcp__plugin_serena_serena__find_referencing_symbols(name_path="BaseClass", relative_path="acclient.h")

# Write modernized code
mcp__plugin_serena_serena__create_text_file(relative_path="output/include/Player.h", content="...")
```

### claude-mem Usage

For persistent state across sessions:
```
# Search for prior work on a class
/claude-mem:mem-search query="Player class modernization"

# Store completion state
# (via automatic observation capture)
```

## Input Files

Required in project root:
- `acclient.h` - Decompiled header (structs, enums, forward declarations)
- `acclient.c` - Decompiled source (method implementations)
- `acclient.txt` (optional) - Constants/debug type info from PDB
- `lines.txt` (optional) - Offset-to-filename mapping

## Database Schema (types.db)

The SQLite database from preprocessing contains:
- `types`: Raw parsed structs/enums with code, namespace, parent
- `methods`: Raw parsed methods with definition, parent class, offset
- `processed_types`: Tracks which types have been modernized
- `processed_methods`: Tracks which methods have been modernized
- `constants`: Named constants from acclient.txt

## Modernization Rules

When generating modern C++ code:

### Header Generation
- Use `#pragma once` instead of include guards
- Convert `struct __cppobj` to `class` with proper access specifiers
- Preserve inheritance and member order
- Add minimal includes, prefer forward declarations
- Do not inline function definitions
- Do not rename classes or methods

### Method Modernization
- Remove decompiler artifacts (`__thiscall`, `__cdecl`, explicit `this` pointer)
- Use modern types (`uint32_t`, `bool` instead of `int` for flags)
- Rename local variables for readability (but not parameters)
- Add comments explaining logic
- Preserve ALL original logic - no additions or removals
- Use enum names instead of magic values where known

### Verification
Every modernized method should be verified to preserve original logic:
- Compare control flow structure
- Verify arithmetic operations unchanged
- Check that all side effects are preserved
- Allow: variable renaming, type modernization, style changes
- Reject: logic changes, missing operations, added functionality

## Output Structure

```
output/
├── include/           # Generated .h headers
│   ├── Turbine/       # Namespaced subdirectories
│   │   └── Player.h
│   └── ...
└── src/               # Generated .cpp sources
    ├── Turbine/
    │   └── Player.cpp
    └── ...
```

---

# Implementation Plan: Claude Code Migration

## Executive Summary

This plan migrates the C++ modernization pipeline from **Python + LM Studio** to **Claude Code native architecture**. The migration preserves the preprocessing layer (Python parsing to SQLite) while replacing the LLM integration layer with Claude Code Skills, Agents, Serena MCP, and claude-mem.

**Key Insight**: Claude Code IS the LLM—we don't call an API, we ARE the API. Skills become prompt templates, Agents enable parallelization, and claude-mem provides cross-session state.

---

## Architecture Overview

### Component Mapping

| Legacy Component | Claude Code Replacement | Status |
|------------------|------------------------|--------|
| `llm_process.py` | Claude Code session orchestration | Phase A |
| `LLMClient.generate()` | Direct Claude prompts (native) | Phase A |
| `ClassHeaderGenerator` | `/modernize-class` skill | Phase A |
| `FunctionProcessor` | `/modernize-method` skill | Phase A |
| `verify_logic()` 5-retry | Tiered verification in skills | Phase A |
| `FEW_SHOT_*` prompts | Skill markdown files | Phase A |
| `llm_cache.db` | claude-mem observations | Phase A |
| `processed_*` tables | claude-mem state tracking | Phase A |
| Sequential processing | Parallel agents | Phase B |
| Regex dependency analysis | Serena semantic analysis | Phase C |

### Components to PRESERVE (Python)

These work well and should remain:
- `process.py` - CLI for preprocessing
- `header_parser.py` - Regex-based struct/enum extraction
- `source_parser.py` - Method body extraction
- `db_handler.py` - SQLite CRUD for types.db
- `dependency_analyzer.py` - Topological sort
- `struct.py`, `method.py`, `enum.py` - Data classes
- `constants.py` - Ignore lists and filtering

---

## Phase 0: Hybrid Foundation (Backwards Compatibility)

**Goal**: Refactor legacy code to support multiple LLM engines while maintaining full backwards compatibility

### 0.1 Engine Abstraction Layer

Create an abstract interface that both LM Studio and Claude Code can implement:

```
local-llm-code/
├── engines/
│   ├── __init__.py           # Engine exports and factory
│   ├── base.py               # Abstract LLMEngine interface
│   ├── lm_studio.py          # Current code, refactored
│   ├── claude_code.py        # Calls Claude Code CLI/skills
│   └── hybrid.py             # Smart routing between engines
```

**File: `engines/base.py`**
```python
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any

class LLMEngine(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate_header(self, class_name: str, class_info: Dict[str, Any],
                        context: str, is_template: bool = False) -> str:
        """Generate modern C++ header for a class."""
        pass

    @abstractmethod
    def generate_method(self, method_name: str, method_definition: str,
                        parent_class: str, context: str) -> str:
        """Modernize a single method."""
        pass

    @abstractmethod
    def verify_logic(self, original: str, modernized: str) -> Tuple[bool, str, str]:
        """
        Verify equivalence between original and modernized code.
        Returns: (is_equivalent, confidence, reason)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier for logging."""
        pass
```

### 0.2 LM Studio Engine (Refactored Legacy)

**File: `engines/lm_studio.py`**
```python
from .base import LLMEngine
from openai import OpenAI

class LMStudioEngine(LLMEngine):
    """Engine using local LM Studio API."""

    def __init__(self, base_url: str = "http://localhost:1234/v1",
                 temperature: float = 0.2):
        self.client = OpenAI(base_url=base_url, api_key="lm-studio")
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "lm-studio"

    def generate_header(self, class_name, class_info, context, is_template=False):
        prompt = self._build_header_prompt(class_name, class_info, context, is_template)
        return self._call_llm(prompt)

    # ... existing logic from class_header_generator.py
```

### 0.3 Claude Code Engine

**File: `engines/claude_code.py`**
```python
import subprocess
import json
from .base import LLMEngine

class ClaudeCodeEngine(LLMEngine):
    """Engine using Claude Code CLI with skills."""

    def __init__(self, project_root: str = "."):
        self.project_root = project_root

    @property
    def name(self) -> str:
        return "claude-code"

    def generate_header(self, class_name, class_info, context, is_template=False):
        # Write context to temp file for skill to read
        self._write_context(class_name, class_info, context)

        # Invoke Claude Code with skill
        result = subprocess.run(
            ["claude", "-p", f"/modernize-class {class_name}", "--print"],
            capture_output=True,
            text=True,
            cwd=self.project_root,
            timeout=300
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude Code failed: {result.stderr}")

        return self._extract_code_block(result.stdout)

    def generate_method(self, method_name, method_definition, parent_class, context):
        result = subprocess.run(
            ["claude", "-p",
             f"/modernize-method {parent_class} {method_name}",
             "--print"],
            capture_output=True,
            text=True,
            timeout=300
        )
        return self._extract_code_block(result.stdout)

    def verify_logic(self, original, modernized):
        # Use Claude's semantic understanding for verification
        result = subprocess.run(
            ["claude", "-p", "/verify-logic", "--print"],
            input=f"ORIGINAL:\n{original}\n\nMODERNIZED:\n{modernized}",
            capture_output=True,
            text=True
        )

        response = json.loads(self._extract_json(result.stdout))
        return (
            response.get("equivalent", False),
            response.get("confidence", "low"),
            response.get("reason", "")
        )
```

### 0.4 Hybrid Engine (Smart Routing)

**File: `engines/hybrid.py`**
```python
from .base import LLMEngine
from .lm_studio import LMStudioEngine
from .claude_code import ClaudeCodeEngine

class HybridEngine(LLMEngine):
    """Routes to optimal engine based on task complexity."""

    def __init__(self, complexity_threshold: int = 10,
                 prefer_claude_for_verification: bool = True):
        self.lm_studio = LMStudioEngine()
        self.claude_code = ClaudeCodeEngine()
        self.threshold = complexity_threshold
        self.prefer_claude_verification = prefer_claude_for_verification

    @property
    def name(self) -> str:
        return "hybrid"

    def _is_complex(self, class_info: dict) -> bool:
        """Determine if class warrants Claude Code processing."""
        method_count = len(class_info.get('methods', []))
        has_templates = class_info.get('is_template', False)
        has_deep_inheritance = class_info.get('inheritance_depth', 0) > 2

        return (method_count > self.threshold or
                has_templates or
                has_deep_inheritance)

    def generate_header(self, class_name, class_info, context, is_template=False):
        if self._is_complex(class_info) or is_template:
            return self.claude_code.generate_header(...)
        return self.lm_studio.generate_header(...)

    def verify_logic(self, original, modernized):
        if self.prefer_claude_verification:
            return self.claude_code.verify_logic(original, modernized)
        return self.lm_studio.verify_logic(original, modernized)
```

### 0.5 Updated CLI Interface

**File: `llm_process.py` (modified)**
```python
from engines import get_engine

def main():
    parser = argparse.ArgumentParser()
    # ... existing arguments ...

    # New engine selection
    parser.add_argument('--engine',
                        choices=['lm-studio', 'claude-code', 'hybrid'],
                        default='lm-studio',
                        help='LLM engine to use (default: lm-studio)')
    parser.add_argument('--complexity-threshold',
                        type=int, default=10,
                        help='Method count threshold for hybrid routing')

    args = parser.parse_args()

    # Get configured engine
    engine = get_engine(
        args.engine,
        complexity_threshold=args.complexity_threshold
    )

    processor = LLMProcessor(db_path=args.db, engine=engine)
    processor.process_all()

# Engine factory
def get_engine(name: str, **kwargs) -> LLMEngine:
    if name == 'lm-studio':
        return LMStudioEngine()
    elif name == 'claude-code':
        return ClaudeCodeEngine()
    elif name == 'hybrid':
        return HybridEngine(**kwargs)
    raise ValueError(f"Unknown engine: {name}")
```

### 0.6 Shared State Protocol

Both engines use SQLite for state tracking:

```python
# In LLMProcessor (shared by both engines)
def process_class(self, class_name: str):
    # Check if already processed (works for both engines)
    if self.db.is_type_processed(class_name):
        logger.info(f"Skipping {class_name} (already processed)")
        return

    # Process with configured engine
    header = self.engine.generate_header(class_name, ...)

    # Store result (same table for both engines)
    self.db.store_processed_type(
        name=class_name,
        processed_header=header,
        engine_used=self.engine.name  # Track which engine
    )
```

**Schema addition** (optional but useful):
```sql
ALTER TABLE processed_types ADD COLUMN engine_used TEXT;
ALTER TABLE processed_methods ADD COLUMN engine_used TEXT;
```

### 0.7 Phase 0 Deliverables

| Deliverable | Description | File |
|-------------|-------------|------|
| Engine interface | Abstract LLMEngine class | `engines/base.py` |
| LM Studio engine | Refactored legacy code | `engines/lm_studio.py` |
| Claude Code engine | CLI wrapper | `engines/claude_code.py` |
| Hybrid engine | Smart routing | `engines/hybrid.py` |
| Updated CLI | `--engine` flag | `llm_process.py` |
| State compatibility | SQLite shared state | `db_handler.py` |

### 0.8 Phase 0 Success Criteria

- [ ] `python llm_process.py` works exactly as before (no breaking changes)
- [ ] `python llm_process.py --engine lm-studio` produces identical output
- [ ] `python llm_process.py --engine claude-code` works with Claude Code
- [ ] Both engines read/write same SQLite tables
- [ ] Can switch engines mid-processing without state loss
- [ ] Hybrid engine routes based on complexity threshold

### 0.9 Migration Path

```
Step 1: Implement Phase 0 (engine abstraction)
        ↓
Step 2: Test LM Studio engine (should be identical to current)
        ↓
Step 3: Implement Phase A skills
        ↓
Step 4: Test Claude Code engine with skills
        ↓
Step 5: A/B compare outputs between engines
        ↓
Step 6: Enable hybrid mode for production
```

---

## Phase A: Foundation

**Goal**: Create working skills that replicate legacy LLM functionality

### A.1 Project Structure Setup

Create the skills directory structure:

```
skills/
├── modernize-class.md          # Header generation skill
├── modernize-method.md         # Method modernization skill
├── verify-logic.md             # Equivalence verification skill
├── query-types-db.md           # Database query helper skill
└── process-class.md            # Orchestration skill (header + methods)
```

### A.2 Skill: `/modernize-class`

**File**: `skills/modernize-class.md`

**Purpose**: Generate modern C++ header for a class from types.db

**Inputs**:
- `class_name` - Name of class to modernize (e.g., "Player", "Turbine::GameState")

**Workflow**:
1. Query types.db for struct definition via `db_handler.get_type_by_name()`
2. Query methods table for all class methods
3. Query dependency analyzer for referenced types
4. Build context with type definitions (max 10 referenced types)
5. Apply FEW_SHOT_CLASS or FEW_SHOT_TEMPLATE based on `is_template_instantiation()`
6. Generate modern header
7. Write to `output/include/{namespace}/{ClassName}.h`
8. Record completion in claude-mem

**Few-Shot Template** (embedded in skill):
```markdown
## Few-Shot Example: Class Modernization

### Input (Decompiled)
struct __cppobj Player : GameObject {
    int _health;
    int _mana;
    void Update(float dt);
    void TakeDamage(int amount);
};

### Output (Modern C++17)
#pragma once

#include "GameObject.h"

namespace Turbine {

class Player : public GameObject {
public:
    void Update(float dt);
    void TakeDamage(int amount);

private:
    int m_health{0};
    int m_mana{0};
};

} // namespace Turbine
```

**Rules** (embedded in skill):
- Output ONLY code, no explanations
- Use `#pragma once`
- Do NOT rename classes or methods
- Do NOT inline function definitions
- Use file paths from context for includes
- Never forward declare referenced types

### A.3 Skill: `/modernize-method`

**File**: `skills/modernize-method.md`

**Purpose**: Modernize a single method with tiered verification

**Inputs**:
- `class_name` - Parent class name
- `method_name` - Method to modernize
- `--no-verify` - Skip verification (optional)

**Workflow**:
1. Query methods table for method definition
2. Load parent class header from `output/include/` (if exists)
3. Query referenced types for context
4. Apply FEW_SHOT_FUNCTION template
5. Generate modernized method
6. **Tiered Verification**:
   - Level 1: Syntax check (valid C++)
   - Level 2: Structure check (control flow preserved)
   - Level 3: Semantic equivalence (LLM verification)
7. If verification fails: adaptive retry (max 5)
8. Write to method collection for assembly
9. Record in claude-mem

**Verification Tiers**:

```markdown
## Tier 1: Syntax Validation
- Check for balanced braces
- Verify semicolons present
- Validate function signature format

## Tier 2: Structure Check
- Count if/else/switch/for/while blocks
- Verify same number of return statements
- Check loop structures match

## Tier 3: Semantic Equivalence
- Use LLM to compare logic
- Allow: variable renaming, type updates, style changes
- Reject: logic changes, missing operations, added functionality
```

**Adaptive Retry Strategy**:
```markdown
Attempt 1: Standard generation
Attempt 2: If syntax error → specific fix prompt
Attempt 3: If logic change → minimal diff approach
Attempt 4: Conservative transformation (less cleanup)
Attempt 5: Fallback with explicit constraints
```

**Confidence Scoring**:
```markdown
HIGH (auto-accept):
- All 3 tiers pass on first attempt
- No complex control flow changes

MEDIUM (queue for review):
- Passes after 1-2 retries
- Complex nested logic present

LOW (manual intervention):
- 3+ retries required
- Verification uncertainty
```

### A.4 Skill: `/verify-logic`

**File**: `skills/verify-logic.md`

**Purpose**: Standalone verification for comparing original vs modernized code

**Inputs**:
- `original` - Original decompiled code
- `modernized` - Modernized code to verify

**Output**: JSON with `{equivalent: bool, confidence: string, reason: string}`

**Verification Prompt**:
```markdown
You are a senior C++ code reviewer. Compare these two functions and determine if they are logically equivalent.

ALLOWED changes (still equivalent):
- Variable renaming for readability
- Type updates (int → bool for flags, int → uint32_t)
- Early returns instead of nested ifs
- Added null checks for safety
- Style/formatting changes
- Comments added

NOT ALLOWED (NOT equivalent):
- Different arithmetic operations
- Missing or added function calls
- Changed loop bounds
- Different conditional logic
- Removed error handling

Respond with JSON: {"equivalent": true/false, "confidence": "high/medium/low", "reason": "..."}
```

### A.5 Skill: `/query-types-db`

**File**: `skills/query-types-db.md`

**Purpose**: Helper skill to query the SQLite database

**Inputs**:
- `--class <name>` - Get class definition
- `--methods <class>` - Get all methods for a class
- `--stats` - Show database statistics
- `--unprocessed` - List unprocessed types

**Implementation**: Wraps Python `db_handler.py` calls via subprocess or direct SQLite queries.

### A.6 Skill: `/process-class`

**File**: `skills/process-class.md`

**Purpose**: Orchestrate full class processing (header + all methods)

**Inputs**:
- `class_name` - Class to process
- `--force` - Reprocess even if completed
- `--header-only` - Only generate header
- `--interactive` - Pause for feedback after each step

**Workflow**:
```markdown
1. Check claude-mem for prior completion
   - If completed and not --force: skip

2. Generate header
   - Invoke /modernize-class {class_name}
   - If --interactive: show result, ask for feedback

3. Process each method
   - Query methods for this class
   - For each method:
     - Invoke /modernize-method {class_name} {method_name}
     - If --interactive: show result, ask for feedback

4. Assemble source file
   - Combine all processed methods
   - Write to output/src/{namespace}/{ClassName}.cpp

5. Record completion
   - Store in claude-mem: class name, method count, timestamp
   - Update processing statistics
```

### A.7 Interactive Refinement Mode

Key differentiator from legacy system—conversational iteration:

```markdown
## Interactive Processing Example

User: /process-class Player --interactive

Claude: Generated header for Player class:
[shows header code]
Does this look correct? Any changes needed?

User: Rename _health to m_health to match project convention

Claude: Updated header with m_health naming:
[shows updated header]
Proceeding to methods. Processing Player::Update...

User: Looks good, continue

Claude: Player::Update modernized:
[shows method]
Verification: HIGH confidence, equivalent

User: Add a comment explaining the delta time calculation

Claude: Added comment:
[shows updated method with comment]

User: Perfect, continue with remaining methods

Claude: Processing Player::TakeDamage...
[continues]
```

### A.8 claude-mem State Schema

**Observation Types**:

```markdown
## Processed Class Record
- Type: class_processed
- Class: {fully_qualified_name}
- Methods: {count}
- Header: {output_path}
- Source: {output_path}
- Timestamp: {ISO datetime}
- Confidence: {high/medium/low}

## Processing Session
- Type: session_start
- Total classes: {count}
- Completed: {count}
- Current position: {class_name}

## Learned Pattern
- Type: pattern_learned
- Pattern: {description}
- Example: {code snippet}
- Applies to: {class pattern}

## Error Record
- Type: verification_failure
- Class: {name}
- Method: {name}
- Attempts: {count}
- Final status: {success/failed}
- Reason: {description}
```

### A.9 Phase A Deliverables

| Deliverable | Description | File |
|-------------|-------------|------|
| modernize-class skill | Header generation with few-shot | `skills/modernize-class.md` |
| modernize-method skill | Method modernization with verification | `skills/modernize-method.md` |
| verify-logic skill | Equivalence checking | `skills/verify-logic.md` |
| query-types-db skill | Database query helper | `skills/query-types-db.md` |
| process-class skill | Full class orchestration | `skills/process-class.md` |
| State schema | claude-mem observation types | Documented in skills |
| Interactive mode | Conversational refinement | Built into process-class |

### A.10 Phase A Success Criteria

- [ ] Can process a single class end-to-end via `/process-class Player`
- [ ] Tiered verification catches syntax errors without LLM
- [ ] Adaptive retry improves pass rate over fixed retry
- [ ] Interactive mode allows mid-process refinement
- [ ] claude-mem tracks completion state across sessions
- [ ] Output matches legacy system quality

---

## Phase B: Parallelization

**Goal**: Process independent classes in parallel using Claude Code agents

### B.1 Dependency Level Analysis

Group classes by dependency depth for safe parallelization:

```
Level 0: Classes with no dependencies on other project classes
         [BaseObject, Enum types, POD structs]
         → Can process all in parallel

Level 1: Classes depending only on Level 0
         [GameObject : BaseObject, PlayerState : Enum]
         → Wait for Level 0 → Process all in parallel

Level 2: Classes depending on Level 0 or 1
         [Player : GameObject, NPC : GameObject]
         → Wait for Level 1 → Process all in parallel

... continue until all classes processed
```

### B.2 Skill: `/analyze-deps`

**File**: `skills/analyze-deps.md`

**Purpose**: Build dependency levels for parallel processing

**Inputs**:
- `--output <file>` - Write levels to JSON file
- `--visualize` - Generate Mermaid diagram

**Output**:
```json
{
  "levels": {
    "0": ["BaseObject", "GameEnums", "Vector3"],
    "1": ["GameObject", "GameState"],
    "2": ["Player", "NPC", "Item"],
    "3": ["PlayerInventory", "NPCBehavior"]
  },
  "total_classes": 150,
  "max_depth": 3,
  "cycles_detected": []
}
```

**Implementation**:
1. Use existing `dependency_analyzer.py` for base graph
2. Enhance with Serena `find_referencing_symbols` for accuracy
3. Compute levels via BFS from root nodes
4. Detect and report cycles

### B.3 Skill: `/process-batch`

**File**: `skills/process-batch.md`

**Purpose**: Process multiple classes in parallel using agents

**Inputs**:
- `--level <n>` - Process specific dependency level
- `--all` - Process all levels in order
- `--max-parallel <n>` - Max concurrent agents (default: 4)
- `--continue` - Resume from last position

**Workflow**:
```markdown
1. Load dependency levels
   - Invoke /analyze-deps if not cached

2. Check claude-mem for progress
   - Find completed classes
   - Determine resume point

3. For each level (0 to max):
   a. Get classes at this level
   b. Filter out completed classes
   c. Partition into batches of --max-parallel
   d. For each batch:
      - Launch parallel agents via Task tool
      - Each agent invokes /process-class for one class
      - Wait for all agents to complete
      - Aggregate results
      - Record progress in claude-mem
   e. Sync point: ensure all level N complete before level N+1

4. Report final statistics
   - Total processed
   - Success/failure counts
   - Verification confidence distribution
```

### B.4 Agent Configuration

**Agent Type**: `feature-dev:code-architect`

**Agent Prompt Template**:
```markdown
Process the class {class_name} from the AC decompiler modernization project.

Context:
- Database: types.db at local-llm-code/mcp-sources/
- Output directory: output/
- Class definition and methods are in the database

Instructions:
1. Read the class definition from types.db
2. Generate modern C++ header
3. Process each method with verification
4. Write output files
5. Report success/failure with confidence

Use the /process-class skill or implement equivalent logic.
```

### B.5 Work Unit Sizing

Optimize agent workload based on class complexity:

| Class Size | Methods | Strategy |
|------------|---------|----------|
| Small | < 5 | Single agent, single unit |
| Medium | 5-20 | Single agent, header first, then methods |
| Large | > 20 | Header agent + method batch agents |

**Large Class Decomposition**:
```markdown
Class: ComplexGameObject (45 methods)

Agent 1: Generate header
Agent 2: Methods 1-15
Agent 3: Methods 16-30
Agent 4: Methods 31-45

Coordinator: Assemble final source file
```

### B.6 Progress Tracking Dashboard

**Skill**: `/process-status`

**Output**:
```
AC-Decompile Modernization Progress
===================================
Total Classes:     847
Completed:         234 (27.6%)
In Progress:       4
Failed:            2
Remaining:         607

Current Level:     2 of 5
Level Progress:    [████████░░░░░░░░] 45/89

Active Agents:
  - Agent 1: Processing Player (method 5/12)
  - Agent 2: Processing NPC (header complete, methods 0/8)
  - Agent 3: Processing Item (verification retry 2)
  - Agent 4: Processing Weapon (complete, writing files)

Verification Stats:
  - First-attempt pass: 78%
  - After retry pass:   94%
  - Manual review queue: 12 methods

Estimated remaining: ~2 hours at current rate
```

### B.7 Phase B Deliverables

| Deliverable | Description | File |
|-------------|-------------|------|
| analyze-deps skill | Dependency level computation | `skills/analyze-deps.md` |
| process-batch skill | Parallel orchestration | `skills/process-batch.md` |
| process-status skill | Progress dashboard | `skills/process-status.md` |
| Agent templates | Configured agent prompts | `skills/agents/` |
| Work sizing logic | Complexity-based partitioning | In process-batch |

### B.8 Phase B Success Criteria

- [ ] Dependency levels computed correctly (no cycles in parallel batches)
- [ ] 4 agents process classes concurrently
- [ ] Speedup of 3x+ over sequential processing
- [ ] Progress persists across sessions via claude-mem
- [ ] Large classes decomposed into parallel method groups
- [ ] Status dashboard shows real-time progress

---

## Phase C: Intelligence Layer

**Goal**: Learn from processing to improve future results

### C.1 Pattern Library

Store successful transformation patterns in claude-mem:

**Pattern Schema**:
```json
{
  "type": "transformation_pattern",
  "name": "SmartArray to vector",
  "input_pattern": "SmartArray<${T}>",
  "output_pattern": "std::vector<${T}>",
  "context": "Member variable declarations",
  "confidence": "high",
  "examples": [
    {"input": "SmartArray<int> m_items", "output": "std::vector<int> m_items"}
  ],
  "learned_from": ["Player", "Inventory", "Container"]
}
```

**Pattern Types**:
1. **Type Mappings**: `SmartArray<T>` → `std::vector<T>`
2. **Idiom Replacements**: RefCount pattern → `shared_ptr`
3. **Naming Conventions**: `_member` → `m_member`
4. **Decompiler Artifacts**: `__thiscall` removal patterns
5. **Error Handling**: Exception style preferences

### C.2 Skill: `/learn-pattern`

**File**: `skills/learn-pattern.md`

**Purpose**: Extract and store a pattern from successful transformation

**Inputs**:
- `class_name` - Class where pattern was observed
- `--auto` - Automatically extract patterns from recent work

**Workflow**:
1. Compare original and modernized code
2. Identify recurring transformation
3. Generalize to pattern template
4. Store in claude-mem with examples

### C.3 Skill: `/suggest-patterns`

**File**: `skills/suggest-patterns.md`

**Purpose**: Query pattern library for applicable transformations

**Inputs**:
- `class_name` - Class about to be processed
- `--code <snippet>` - Specific code to match

**Output**:
```markdown
## Applicable Patterns for Player

1. **SmartArray to vector** (confidence: high)
   - Found: SmartArray<Item*> m_inventory
   - Suggest: std::vector<Item*> m_inventory

2. **RefCount to shared_ptr** (confidence: medium)
   - Found: class Player : public RefCount
   - Suggest: Use std::enable_shared_from_this

3. **Naming convention** (confidence: high)
   - Found: _health, _mana, _level
   - Suggest: m_health, m_mana, m_level
```

### C.4 Class Similarity Clustering

Group classes by structural similarity for better few-shot context:

**Similarity Metrics**:
- Inheritance depth
- Method count
- Member variable count
- Namespace
- Base class type

**Skill**: `/find-similar`

**Output**:
```markdown
## Classes Similar to Player

1. **NPC** (92% similar)
   - Same base: GameObject
   - Similar method count: 15 vs 12
   - Already processed: YES
   - Use as few-shot example

2. **Monster** (87% similar)
   - Same base: GameObject
   - Similar member types
   - Already processed: YES

3. **Vehicle** (71% similar)
   - Same base: GameObject
   - Fewer methods
   - Not yet processed
```

### C.5 Error Memory (Anti-Patterns)

Track verification failures to avoid repeating mistakes:

**Anti-Pattern Schema**:
```json
{
  "type": "anti_pattern",
  "description": "Do not remove null checks in destructors",
  "trigger": "Destructor with this->member access",
  "bad_transformation": "Removed if(this->ptr) check",
  "correct_approach": "Preserve null check for safety",
  "learned_from": ["GameResource::~GameResource", "FileHandle::~FileHandle"],
  "occurrences": 3
}
```

**Skill**: `/check-antipatterns`

Runs before generation to warn about known issues:
```markdown
## Anti-Pattern Warnings for Player::~Player

⚠️ This destructor has member pointer access
   Known issue: Do not remove null checks
   Affected: 3 previous classes

⚠️ RefCount-derived class destructor
   Known issue: Preserve Release() call order
   Affected: 5 previous classes
```

### C.6 Serena Semantic Enhancement

Replace regex-based dependency detection with semantic analysis:

**Current** (regex):
```python
# Finds "SomeType*" but misses templates, typedefs, etc.
pattern = r'\b([A-Z][A-Za-z0-9_]*)\s*[\*&]'
```

**Enhanced** (Serena):
```markdown
1. Use find_symbol to get class body
2. Use find_referencing_symbols on each type reference
3. Trace through typedefs to actual types
4. Handle template instantiations properly
5. Detect virtual inheritance chains
```

**Benefits**:
- Accurate type resolution
- Template parameter tracking
- Virtual dispatch understanding
- Include path generation

### C.7 Phase C Deliverables

| Deliverable | Description | File |
|-------------|-------------|------|
| learn-pattern skill | Pattern extraction | `skills/learn-pattern.md` |
| suggest-patterns skill | Pattern application | `skills/suggest-patterns.md` |
| find-similar skill | Class similarity | `skills/find-similar.md` |
| check-antipatterns skill | Error prevention | `skills/check-antipatterns.md` |
| Pattern library schema | claude-mem format | Documented |
| Serena integration | Semantic analysis | In analyze-deps |

### C.8 Phase C Success Criteria

- [ ] Pattern library populated after processing 50+ classes
- [ ] Patterns reused in 50%+ of subsequent classes
- [ ] Anti-patterns prevent repeat failures
- [ ] Similar class detection improves few-shot quality
- [ ] Serena semantic analysis more accurate than regex
- [ ] First-attempt verification pass rate >85%

---

## Phase D: Extended Capabilities

**Goal**: Add capabilities beyond the original pipeline

### D.1 Documentation Generation

**Skill**: `/generate-docs`

**Inputs**:
- `class_name` - Generate docs for specific class
- `--all` - Generate for all processed classes
- `--format <doxygen|markdown>` - Output format

**Output**:
- Doxygen comments in header files
- Class hierarchy diagrams (Mermaid)
- API reference documentation

**Example Output**:
```cpp
/**
 * @class Player
 * @brief Represents a player character in the game world.
 *
 * The Player class handles all player-specific functionality including
 * movement, combat, inventory management, and state persistence.
 *
 * @inherits GameObject
 * @see NPC, Monster
 */
class Player : public GameObject {
public:
    /**
     * @brief Updates the player state for the current frame.
     * @param dt Delta time since last update in seconds.
     */
    void Update(float dt);

    /**
     * @brief Applies damage to the player.
     * @param amount The amount of damage to apply.
     * @return true if player is still alive, false if dead.
     */
    bool TakeDamage(int amount);
};
```

### D.2 Test Stub Generation

**Skill**: `/generate-tests`

**Inputs**:
- `class_name` - Generate tests for specific class
- `--framework <gtest|catch2>` - Test framework

**Output**:
```cpp
// test/PlayerTest.cpp
#include <gtest/gtest.h>
#include "Turbine/Player.h"

class PlayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        player = std::make_unique<Player>();
    }
    std::unique_ptr<Player> player;
};

TEST_F(PlayerTest, Update_WithPositiveDelta_UpdatesState) {
    // TODO: Implement based on original behavior
    // Original logic: Updates position based on velocity
    player->Update(0.016f);
    // EXPECT_...
}

TEST_F(PlayerTest, TakeDamage_WithPositiveAmount_ReducesHealth) {
    // TODO: Implement based on original behavior
    // Original logic: Subtracts amount from _health
    bool alive = player->TakeDamage(10);
    // EXPECT_TRUE(alive);
}
```

### D.3 Build System Generation

**Skill**: `/generate-cmake`

**Output**:
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(ACClient VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generated from dependency analysis
add_library(ACClient
    src/Turbine/BaseObject.cpp
    src/Turbine/GameObject.cpp
    src/Turbine/Player.cpp
    src/Turbine/NPC.cpp
    # ... auto-generated list
)

target_include_directories(ACClient PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Dependencies from processing order
target_link_libraries(ACClient PUBLIC
    # External dependencies detected
)
```

### D.4 Compilation Validation

**Skill**: `/validate-compilation`

**Workflow**:
1. Generate CMakeLists.txt
2. Run cmake configure
3. Run cmake build
4. Capture compiler errors
5. Parse errors into actionable feedback
6. Feed back into refinement loop

**Example Feedback**:
```markdown
## Compilation Errors for Player.cpp

1. **Error: 'Vector3' was not declared**
   - Line: 45
   - Fix: Add #include "Math/Vector3.h"
   - Auto-fix available: YES

2. **Error: Cannot convert 'int' to 'bool'**
   - Line: 78
   - Context: return _health;
   - Fix: return _health > 0;
   - Auto-fix available: YES (requires verification)

3. **Warning: Unused variable 'temp'**
   - Line: 102
   - Fix: Remove or use variable
   - Auto-fix available: NO (review required)
```

### D.5 Cross-Reference Database

**Skill**: `/generate-compile-commands`

Generate `compile_commands.json` for IDE integration:
```json
[
  {
    "directory": "/path/to/project",
    "command": "clang++ -std=c++17 -I/path/to/include -c src/Player.cpp",
    "file": "src/Player.cpp"
  }
]
```

**Benefits**:
- Clangd integration for VS Code
- Go-to-definition
- Find all references
- Semantic highlighting

### D.6 Phase D Deliverables

| Deliverable | Description | File |
|-------------|-------------|------|
| generate-docs skill | Documentation generation | `skills/generate-docs.md` |
| generate-tests skill | Test stub creation | `skills/generate-tests.md` |
| generate-cmake skill | Build system generation | `skills/generate-cmake.md` |
| validate-compilation skill | Compile and feedback | `skills/validate-compilation.md` |
| generate-compile-commands skill | IDE integration | `skills/generate-compile-commands.md` |

### D.7 Phase D Success Criteria

- [ ] Doxygen comments generated for all public APIs
- [ ] Test stubs created for all classes
- [ ] CMakeLists.txt successfully configures
- [ ] 95%+ of output compiles without errors
- [ ] compile_commands.json enables IDE features

---

## Implementation Schedule

### Phase A: Foundation
**Scope**: Core skills and interactive processing

**Tasks**:
1. Create skills/ directory structure
2. Implement /modernize-class skill
3. Implement /modernize-method skill with tiered verification
4. Implement /verify-logic skill
5. Implement /query-types-db skill
6. Implement /process-class skill with interactive mode
7. Define claude-mem state schema
8. Test with 10 sample classes
9. Validate output matches legacy quality

### Phase B: Parallelization
**Scope**: Multi-agent parallel processing

**Tasks**:
1. Implement /analyze-deps skill
2. Design agent pool configuration
3. Implement /process-batch skill
4. Add work unit sizing logic
5. Implement /process-status dashboard
6. Test with full class set
7. Measure speedup vs sequential
8. Handle edge cases (cycles, failures)

### Phase C: Intelligence
**Scope**: Learning and pattern recognition

**Tasks**:
1. Design pattern library schema
2. Implement /learn-pattern skill
3. Implement /suggest-patterns skill
4. Implement /find-similar skill
5. Implement /check-antipatterns skill
6. Enhance analyze-deps with Serena
7. Measure pattern reuse rate
8. Validate improved first-attempt pass rate

### Phase D: Extended Capabilities
**Scope**: Documentation, tests, build system

**Tasks**:
1. Implement /generate-docs skill
2. Implement /generate-tests skill
3. Implement /generate-cmake skill
4. Implement /validate-compilation skill
5. Implement /generate-compile-commands skill
6. End-to-end validation
7. Documentation and handoff

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Verification pass rate (1st attempt) | >80% | Logged per method |
| Verification pass rate (after retry) | >95% | Logged per method |
| Average retries per method | <1.5 | Aggregated stats |
| Parallel speedup factor | >3x | Wall clock comparison |
| Pattern reuse rate | >50% (after 100 classes) | Pattern match count |
| Compilation success rate | >95% | cmake build result |
| Cross-session resume | 100% | claude-mem persistence |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Large files overwhelm context | Chunk processing, progressive loading |
| Circular dependencies | Cycle detection, manual intervention queue |
| Verification false positives | Confidence scoring, human review queue |
| Agent coordination failures | Checkpoint/resume, orphan detection |
| Pattern overfitting | Pattern confidence thresholds, decay |
| Compilation errors | Iterative feedback loop, conservative fallback |

---

## Appendix: File Reference

### Skills Directory Structure
```
skills/
├── README.md                    # Skills overview
├── modernize-class.md           # Phase A
├── modernize-method.md          # Phase A
├── verify-logic.md              # Phase A
├── query-types-db.md            # Phase A
├── process-class.md             # Phase A
├── analyze-deps.md              # Phase B
├── process-batch.md             # Phase B
├── process-status.md            # Phase B
├── learn-pattern.md             # Phase C
├── suggest-patterns.md          # Phase C
├── find-similar.md              # Phase C
├── check-antipatterns.md        # Phase C
├── generate-docs.md             # Phase D
├── generate-tests.md            # Phase D
├── generate-cmake.md            # Phase D
├── validate-compilation.md      # Phase D
└── generate-compile-commands.md # Phase D
```

### Serena Memory Files
```
.serena/memories/
├── project_overview.md          # Project context
├── suggested_commands.md        # CLI reference
├── code_style_conventions.md    # Style guide
├── task_completion_checklist.md # Quality gates
├── migration_architecture.md    # Component mapping
└── enhancement_roadmap.md       # This plan summary
```

### Output Structure
```
output/
├── include/                     # Generated headers
│   └── {Namespace}/
│       └── {ClassName}.h
├── src/                         # Generated sources
│   └── {Namespace}/
│       └── {ClassName}.cpp
├── test/                        # Generated test stubs (Phase D)
│   └── {Namespace}/
│       └── {ClassName}Test.cpp
├── docs/                        # Generated documentation (Phase D)
│   ├── api/
│   └── diagrams/
├── CMakeLists.txt               # Generated build system (Phase D)
└── compile_commands.json        # IDE integration (Phase D)
```
