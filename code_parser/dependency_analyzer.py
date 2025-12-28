"""
Dependency Analyzer
==================
Extracts type references from code, builds dependency graphs, and provides
topological sorting for proper processing order.
"""
import logging
import re
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TypeNode:
    """Represents a type in the dependency graph"""
    name: str
    kind: str  # 'struct', 'enum', 'class'
    dependencies: Set[str]
    code: str = ""
    
    def __hash__(self):
        return hash(self.name)


class DependencyAnalyzer:
    """
    Analyzes type dependencies in C++ code and provides processing order.
    
    Uses regex-based pattern matching to extract type references from:
    - Base class declarations
    - Member variable types
    - Function parameters and return types
    - Template parameters
    """
    
    def __init__(self, db_handler):
        self.db = db_handler
        self.type_nodes: Dict[str, TypeNode] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.known_types: Set[str] = set()
        
        # Regex patterns for type extraction
        self.patterns = {
            # Base class: "class Foo : public Bar" or "struct Foo : Bar"
            'base_class': re.compile(
                r'(?:class|struct)\s+\w+\s*:\s*(?:public\s+|private\s+|protected\s+)?(\w+)',
                re.MULTILINE
            ),
            # Pointer/reference types: "SomeType*" or "SomeType&" or "SomeType *"
            'pointer_ref': re.compile(
                r'\b([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*[*&]',
                re.MULTILINE
            ),
            # Template parameters: "SmartArray<ItemType>"
            'template': re.compile(
                r'\b(\w+)\s*<\s*([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)',
                re.MULTILINE
            ),
            # Member declarations: "SomeType memberName;"
            'member_decl': re.compile(
                r'^\s*([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s+\w+\s*;',
                re.MULTILINE
            ),
            # Function parameters: "(SomeType param, OtherType* param2)"
            'func_param': re.compile(
                r'\(\s*(?:[^()]*,\s*)?([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*[*&]?\s+\w+',
                re.MULTILINE
            ),
            # Return types: "SomeType* ClassName::Method"
            'return_type': re.compile(
                r'^([A-Z][A-Za-z0-9_]*(?:::[A-Z][A-Za-z0-9_]*)*)\s*[*&]?\s+\w+::',
                re.MULTILINE
            ),
        }
        
        # Common types to ignore (primitives, std library, etc.)
        self.ignore_types = {
            'int', 'unsigned', 'char', 'short', 'long', 'float', 'double',
            'void', 'bool', 'size_t', 'uint8_t', 'uint16_t', 'uint32_t',
            'uint64_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'BYTE', 'WORD', 'DWORD', 'QWORD', 'BOOL', 'HANDLE', 'HRESULT',
            'TRUE', 'FALSE', 'NULL', 'nullptr', 'string', 'vector', 'map',
            'set', 'list', 'array', 'pair', 'tuple', 'optional', 'variant',
            'unique_ptr', 'shared_ptr', 'weak_ptr', 'String', 'Vector',
        }
    
    def load_known_types(self):
        """Load all known type names from the database"""
        # Load structs
        structs = self.db.get_structs()
        for s in structs:
            name = s[2]  # name column
            self.known_types.add(name)
            # Also add simple name if it's namespaced
            if '::' in name:
                self.known_types.add(name.split('::')[-1])
        
        # Load enums
        enums = self.db.get_enums()
        for e in enums:
            name = e[2]  # name column
            self.known_types.add(name)
            if '::' in name:
                self.known_types.add(name.split('::')[-1])
    
    def extract_type_references(self, code: str) -> Set[str]:
        """
        Extract all type names referenced in the code.
        Returns a set of type names after filtering out primitives and unknowns.
        """
        references = set()
        
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(code)
            for match in matches:
                # Handle tuple results from some patterns
                if isinstance(match, tuple):
                    for m in match:
                        if m and self._is_valid_type(m):
                            references.add(m)
                elif match and self._is_valid_type(match):
                    references.add(match)
        
        return references
    
    def _is_valid_type(self, type_name: str) -> bool:
        """Check if a type name is valid and should be tracked"""
        if not type_name:
            return False
        
        # Ignore primitives and common types
        if type_name in self.ignore_types:
            return False
        
        # Ignore all-lowercase names (likely variables, not types)
        if type_name.islower():
            return False
        
        # Must start with uppercase or contain ::
        if not (type_name[0].isupper() or '::' in type_name):
            return False
        
        return True
    
    def build_dependency_graph(self):
        """Build the dependency graph from all types in the database"""
        self.load_known_types()
        
        # Process structs
        structs = self.db.get_structs()
        for s in structs:
            name = s[2]  # name column
            code = s[5]  # code column
            is_ignored = s[7]  # is_ignored column
            
            if is_ignored:
                continue
            
            refs = self.extract_type_references(code)
            # Filter to only known types
            refs = refs & self.known_types
            # Remove self-references
            refs.discard(name)
            simple_name = name.split('::')[-1] if '::' in name else name
            refs.discard(simple_name)
            
            self.type_nodes[name] = TypeNode(
                name=name,
                kind='struct',
                dependencies=refs,
                code=code
            )
            self.dependencies[name] = refs
        
        # Process enums
        enums = self.db.get_enums()
        for e in enums:
            name = e[2]  # name column
            code = e[5]  # code column
            is_ignored = e[7] if len(e) > 7 else False
            
            if is_ignored:
                continue
            
            # Enums typically don't have dependencies, but check anyway
            refs = self.extract_type_references(code)
            refs = refs & self.known_types
            refs.discard(name)
            
            self.type_nodes[name] = TypeNode(
                name=name,
                kind='enum',
                dependencies=refs,
                code=code
            )
            self.dependencies[name] = refs
    
    def find_cycles(self) -> List[Set[str]]:
        """Find all cycles in the dependency graph using Tarjan's algorithm"""
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        sccs = []  # Strongly connected components
        
        def strongconnect(node):
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for successor in self.dependencies.get(node, set()):
                if successor not in index:
                    strongconnect(successor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[successor])
                elif on_stack.get(successor, False):
                    lowlinks[node] = min(lowlinks[node], index[successor])
            
            if lowlinks[node] == index[node]:
                scc = set()
                while True:
                    successor = stack.pop()
                    on_stack[successor] = False
                    scc.add(successor)
                    if successor == node:
                        break
                # Only report cycles (SCCs with more than one node)
                if len(scc) > 1:
                    sccs.append(scc)
        
        for node in self.dependencies:
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def topological_sort(self) -> List[str]:
        """
        Return types in topological order (dependencies first).
        Handles cycles by grouping them together.
        """
        if not self.dependencies:
            self.build_dependency_graph()
        
        # Find cycles
        cycles = self.find_cycles()
        cycle_members = set()
        for cycle in cycles:
            cycle_members.update(cycle)
        
        # Build in-degree map
        in_degree = defaultdict(int)
        for node in self.dependencies:
            if node not in in_degree:
                in_degree[node] = 0
            for dep in self.dependencies[node]:
                # Don't count edges within cycles
                if dep in cycle_members and node in cycle_members:
                    continue
                in_degree[dep]  # Ensure dep exists in map
        
        # Types with no dependencies first
        for node, deps in self.dependencies.items():
            for dep in deps:
                if dep not in cycle_members or node not in cycle_members:
                    in_degree[node] += 1 if dep in self.dependencies else 0
        
        # Kahn's algorithm
        result = []
        queue = []
        
        # Start with nodes that have no dependencies
        for node in self.dependencies:
            if not self.dependencies[node] - cycle_members:
                queue.append(node)
        
        # Also add cycle groups (processed together)
        processed_cycles = set()
        
        visited = set()
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            # If part of a cycle, add entire cycle
            in_cycle = False
            for cycle in cycles:
                if node in cycle and id(cycle) not in processed_cycles:
                    result.extend(sorted(cycle))
                    visited.update(cycle)
                    processed_cycles.add(id(cycle))
                    in_cycle = True
                    break
            
            if not in_cycle:
                result.append(node)
            
            # Add dependents that now have all deps satisfied
            for potential_next in self.dependencies:
                if potential_next not in visited:
                    deps_remaining = self.dependencies[potential_next] - visited
                    if not deps_remaining:
                        queue.append(potential_next)
        
        # Add any remaining nodes not in the graph
        for node in self.dependencies:
            if node not in visited:
                result.append(node)
        
        return result
    
    def get_processing_order(self) -> List[Tuple[str, str]]:
        """
        Return (type_name, type_kind) tuples in processing order.
        Dependencies come before dependents.
        """
        if not self.type_nodes:
            self.build_dependency_graph()
        
        order = self.topological_sort()
        
        result = []
        for name in order:
            if name in self.type_nodes:
                node = self.type_nodes[name]
                result.append((name, node.kind))
        
        return result
    
    def get_type_dependencies(self, type_name: str) -> Set[str]:
        """Get the direct dependencies of a specific type"""
        if not self.dependencies:
            self.build_dependency_graph()
        
        return self.dependencies.get(type_name, set())
    
    def get_dependency_stats(self) -> Dict[str, any]:
        """Get statistics about the dependency graph"""
        if not self.dependencies:
            self.build_dependency_graph()
        
        cycles = self.find_cycles()
        
        total_deps = sum(len(deps) for deps in self.dependencies.values())
        max_deps = max((len(deps) for deps in self.dependencies.values()), default=0)
        
        types_with_no_deps = sum(1 for deps in self.dependencies.values() if not deps)
        
        return {
            'total_types': len(self.type_nodes),
            'total_dependencies': total_deps,
            'average_dependencies': total_deps / len(self.dependencies) if self.dependencies else 0,
            'max_dependencies': max_deps,
            'types_with_no_deps': types_with_no_deps,
            'circular_dependency_groups': len(cycles),
            'types_in_cycles': sum(len(c) for c in cycles),
        }
    
    def print_summary(self):
        """Print a summary of the dependency analysis"""
        stats = self.get_dependency_stats()
        logger.info("=" * 60)
        logger.info("Dependency Analysis Summary")
        logger.info("=" * 60)
        logger.info(f"Total types: {stats['total_types']}")
        logger.info(f"Total dependency edges: {stats['total_dependencies']}")
        logger.info(f"Average dependencies per type: {stats['average_dependencies']:.2f}")
        logger.info(f"Max dependencies: {stats['max_dependencies']}")
        logger.info(f"Types with no dependencies: {stats['types_with_no_deps']}")
        logger.info(f"Circular dependency groups: {stats['circular_dependency_groups']}")
        logger.info(f"Types involved in cycles: {stats['types_in_cycles']}")
        logger.info("=" * 60)
