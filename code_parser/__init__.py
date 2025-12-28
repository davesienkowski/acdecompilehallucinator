"""
Code Parser module - A tool for parsing C++ header files and extracting struct definitions.
"""

from .header_parser import HeaderParser
from .type_writer import TypeWriter
from .struct import Struct
from .enum import Enum
from .method import Method
from .db_handler import DatabaseHandler
from .dependency_analyzer import DependencyAnalyzer
from .class_header_generator import ClassHeaderGenerator
from .function_processor import FunctionProcessor
from .class_assembler import ClassAssembler
from .constants_parser import ConstantsParser
from .constant_replacer import ConstantReplacer
from .llm_cache import LLMCache
from .type_resolver import TypeResolver
# NOTE: LLMClient and LLMProcessor have been removed.
# Use engines.get_engine("lm-studio") or engines.get_engine("claude-code") instead.
# The LLMProcessor class is now in llm_process.py (CLI entry point).
from .exceptions import (
    ACDecompileError,
    ParsingError, HeaderParsingError, SourceParsingError,
    MethodSignatureError, EncodingError, InvalidStructureError,
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    DatabaseConstraintError, DatabaseMigrationError,
    LLMError, LLMConnectionError, LLMResponseError,
    LLMProcessingError, LLMVerificationError, LLMCacheError,
    FileIOError, FileNotFoundError, FileWriteError,
    FileEncodingError, PathError,
    ConfigurationError, MissingConfigurationError,
    InvalidConfigurationError, DependencyMissingError,
    ProcessingError, DependencyResolutionError,
    TypeResolutionError, CodeGenerationError, ValidationError
)

__all__ = [
    'HeaderParser', 'TypeWriter', 'Struct', 'Enum', 'Method',
    'DatabaseHandler', 'DependencyAnalyzer', 'ClassHeaderGenerator',
    'FunctionProcessor', 'ClassAssembler', 'ConstantsParser',
    'ConstantReplacer', 'LLMCache', 'TypeResolver',
    # Exceptions
    'ACDecompileError',
    'ParsingError', 'HeaderParsingError', 'SourceParsingError',
    'MethodSignatureError', 'EncodingError', 'InvalidStructureError',
    'DatabaseError', 'DatabaseConnectionError', 'DatabaseQueryError',
    'DatabaseConstraintError', 'DatabaseMigrationError',
    'LLMError', 'LLMConnectionError', 'LLMResponseError',
    'LLMProcessingError', 'LLMVerificationError', 'LLMCacheError',
    'FileIOError', 'FileNotFoundError', 'FileWriteError',
    'FileEncodingError', 'PathError',
    'ConfigurationError', 'MissingConfigurationError',
    'InvalidConfigurationError', 'DependencyMissingError',
    'ProcessingError', 'DependencyResolutionError',
    'TypeResolutionError', 'CodeGenerationError', 'ValidationError'
]