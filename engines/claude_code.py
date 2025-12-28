"""
Claude Code engine implementation.

This engine uses the Claude Code CLI to generate responses,
enabling Claude's advanced reasoning for C++ code modernization.

Note: This is a stub implementation for Phase 3. It requires:
- Claude Code CLI installed and configured
- Skills defined in skills/ directory
"""

import logging
import subprocess
import shutil
from typing import Optional, Any

from .base import (
    LLMEngine,
    EngineConfig,
    EngineError,
    EngineConnectionError,
    EngineTimeoutError,
    EngineResponseError,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Default Claude Code configuration
DEFAULT_TIMEOUT = 300  # 5 minutes for complex code generation


class ClaudeCodeEngine(LLMEngine):
    """Engine using Claude Code CLI.

    This engine invokes the Claude Code CLI to generate responses,
    leveraging Claude's advanced reasoning capabilities for code
    modernization tasks.

    Requirements:
        - Claude Code CLI installed (npm install -g @anthropic/claude-code)
        - Valid API key configured

    Configuration (via config.extra):
        project_root: Working directory for Claude Code (default: .)
        cli_timeout: Subprocess timeout in seconds (default: 300)
        print_mode: Use --print flag for immediate output (default: True)

    Example:
        from engines import get_engine, EngineConfig

        config = EngineConfig(
            timeout=300,
            extra={"project_root": "/path/to/project"}
        )
        engine = get_engine("claude-code", config=config)
        response = engine.generate("Modernize this code...")

    Note:
        Unlike LM Studio, Claude Code uses a conversation-based interface.
        Each generate() call starts a new conversation with the prompt.
    """

    def __init__(self, config: Optional[EngineConfig] = None, cache: Optional[Any] = None):
        """Initialize the Claude Code engine.

        Args:
            config: Engine configuration.
            cache: Optional cache instance for response caching.

        Raises:
            EngineError: If Claude Code CLI is not found.
        """
        super().__init__(config, cache)

        # Extract Claude Code specific config
        self.project_root = self.config.extra.get("project_root", ".")
        self.cli_timeout = self.config.extra.get("cli_timeout", DEFAULT_TIMEOUT)
        self.print_mode = self.config.extra.get("print_mode", True)

        # Check if Claude Code CLI is available
        self.cli_path = shutil.which("claude")
        if not self.cli_path:
            logger.warning(
                "Claude Code CLI not found. Install with: npm install -g @anthropic/claude-code"
            )
            # Don't raise error - allow engine to be registered but not used
            self._initialized = False
        else:
            self._initialized = True
            logger.info(f"Claude Code CLI found at: {self.cli_path}")

    @property
    def name(self) -> str:
        """Engine identifier."""
        return "claude-code"

    def generate(self, prompt: str) -> str:
        """Generate a response using Claude Code CLI.

        Calls the Claude Code CLI with the given prompt and returns
        the generated response.

        Args:
            prompt: The prompt to send to Claude.

        Returns:
            The generated response text.

        Raises:
            EngineError: If Claude Code CLI is not available.
            EngineTimeoutError: If the request times out.
            EngineConnectionError: If the CLI fails to execute.
            EngineResponseError: If the response cannot be parsed.
        """
        if not self._initialized:
            raise EngineError(
                "Claude Code CLI not available. Install with: npm install -g @anthropic/claude-code",
                engine_name=self.name
            )

        # Check cache first
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                logger.debug("Using cached response")
                return cached

        try:
            # Build command
            cmd = [self.cli_path, "-p", prompt]
            if self.print_mode:
                cmd.append("--print")

            logger.debug(f"Executing: {' '.join(cmd[:3])}...")

            # Execute Claude Code CLI
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=self.cli_timeout,
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise EngineResponseError(
                    f"Claude Code CLI returned error: {error_msg}",
                    engine_name=self.name
                )

            response = result.stdout.strip()
            if not response:
                raise EngineResponseError(
                    "Claude Code returned empty response",
                    engine_name=self.name
                )

            # Extract code block if present
            response = self._extract_code_block(response)

            # Cache the response
            if self.cache and response:
                self.cache.store(prompt, response)

            logger.debug(f"Generated {len(response)} characters")
            return response

        except subprocess.TimeoutExpired:
            raise EngineTimeoutError(
                f"Claude Code timed out after {self.cli_timeout}s",
                engine_name=self.name
            )
        except FileNotFoundError:
            raise EngineConnectionError(
                "Claude Code CLI not found",
                engine_name=self.name
            )
        except subprocess.SubprocessError as e:
            raise EngineError(
                f"Failed to execute Claude Code: {e}",
                engine_name=self.name,
                cause=e
            )

    def _extract_code_block(self, response: str) -> str:
        """Extract code from markdown code blocks if present.

        Claude Code often wraps responses in markdown code blocks.
        This method extracts the content from within the blocks.

        Args:
            response: The raw response from Claude Code.

        Returns:
            The extracted code or original response if no blocks found.
        """
        import re

        # Try to find C++ code blocks first
        cpp_pattern = r'```(?:cpp|c\+\+|c)\n(.*?)```'
        matches = re.findall(cpp_pattern, response, re.DOTALL)
        if matches:
            return '\n\n'.join(matches)

        # Try generic code blocks
        generic_pattern = r'```\n?(.*?)```'
        matches = re.findall(generic_pattern, response, re.DOTALL)
        if matches:
            return '\n\n'.join(matches)

        # No code blocks found, return as-is
        return response

    def is_available(self) -> bool:
        """Check if Claude Code CLI is available and configured.

        Returns:
            True if Claude Code can be used, False otherwise.
        """
        if not self._initialized:
            return False

        try:
            # Try to get version
            result = subprocess.run(
                [self.cli_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def __repr__(self) -> str:
        """String representation."""
        status = "available" if self.is_available() else "not available"
        return f"ClaudeCodeEngine({status})"
