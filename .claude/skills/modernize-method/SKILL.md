---
name: modernize-method
description: Modernize decompiled C++ functions to clean, idiomatic C++17+. Use when converting raw decompiled methods, removing decompiler artifacts like __thiscall, or improving function readability while preserving exact logic.
allowed-tools: Read, Grep, Glob
---

# Modernize Method

Transform decompiled C++ functions into clean, modern C++ while preserving exact logic.

## When to Use

- Converting raw decompiled functions with artifacts like `__thiscall`, `__cdecl`
- Modernizing explicit `this` pointer usage to implicit member access
- Improving local variable names for readability
- Adding explanatory comments to complex logic

## Instructions

1. Read the original decompiled function
2. Identify the parent class and look up its processed header
3. Find all referenced types and get their definitions
4. Generate modernized code following strict rules
5. Verify logic preservation (use /verify-logic if needed)

## Strict Rules

- Output ONLY the function code, no explanations or extra declarations
- Do NOT rename the function or parameters
- Renaming local variables for readability IS encouraged
- Remove decompiler artifacts:
  - `__thiscall`, `__cdecl`, `__stdcall`, `__fastcall`
  - Explicit `this` pointer (e.g., `this->_health` -> `_health`)
  - `__userpurge`, `__usercall`
- Use modern types:
  - `uint32_t` instead of `unsigned int`
  - `bool` instead of `int` for boolean values
  - `nullptr` instead of `NULL`
- Add comments to explain logic/purpose
- Do NOT add or remove logic
- Keep function signature compatible with class header
- Use enum names instead of magic values where known
- Do NOT invent constants that don't exist

## Example

### Input
```cpp
void __thiscall Player::Update(Player *this, float dt)
{
    int v1 = this->_health;
    if (v1 <= 0) {
        this->_flags = this->_flags | 0x1;
    }
}
```

### Output
```cpp
/*
    Updates the player's state.
*/
void Player::Update(float dt) {
    if (_health <= 0) {
        // Add dead flag
        _flags |= 0x1;
    }
}
```

## Verification

After modernizing, the function should be verified to ensure logic equivalence:
- Control flow must be identical
- All arithmetic operations preserved
- All side effects maintained
- Only allowed changes: variable names, types, style

## Output Path Convention

Source files are written to: `output/src/{Namespace}/{ClassName}.cpp`

Each function includes an offset comment:
```cpp
// Offset: 0x00401234
void Player::Update(float dt) {
    // ...
}
```
