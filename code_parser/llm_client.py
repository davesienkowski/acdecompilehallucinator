#!/usr/bin/env python3
"""
LLM Client
==========

Simple LLM client using OpenAI-compatible API (e.g., LMStudio)
"""

import logging
import time
from typing import Optional

from code_parser.llm_cache import LLMCache

logger = logging.getLogger("llm-processor")

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
MAX_LLM_TOKENS = 131072


class LLMClient:
    """Simple LLM client using OpenAI-compatible API (e.g., LMStudio)."""

    def __init__(self, base_url: str = LM_STUDIO_URL, temperature: float = 0.2, cache: Optional[LLMCache] = None, db_handler=None):
        """Initialize the OpenAI-compatible client.

        Args:
            base_url: Base URL for the LLM API endpoint. Defaults to LM Studio
                local server at localhost:1234.
            temperature: Sampling temperature for generation. Lower values
                produce more deterministic outputs. Defaults to 0.2.
            cache: Optional LLMCache instance for caching responses to avoid
                redundant API calls.
            db_handler: Optional DatabaseHandler for storing token usage metrics.

        Raises:
            ImportError: If the openai package is not installed.
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key="lm-studio")
            self.temperature = temperature
            self.cache = cache
            self.db_handler = db_handler
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = MAX_LLM_TOKENS) -> str:
        """Generate a response from the LLM, checking cache first.

        Sends the prompt to the LLM API and returns the generated response.
        If a cache is configured, checks for a cached response before making
        the API call. Token usage is logged and optionally stored in the database.

        Args:
            prompt: The input prompt to send to the LLM.
            max_tokens: Maximum number of tokens for the response. Defaults to
                MAX_LLM_TOKENS (131072).

        Returns:
            The generated text response from the LLM, or an empty string if
            no response was generated.
        """
        # Check cache
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                logger.info("Cache hit! Using stored response.")
                # For cached responses, we don't track timing since no actual LLM call was made
                return cached

        # Record start time
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a C++ modernization expert. Output ONLY clean code, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            timeout=3000
        )
        
        # Calculate request time
        request_time = time.time() - start_time
        
        result = response.choices[0].message.content or ""

        # Store in cache
        if self.cache and result:
            self.cache.set(prompt, result)
        
        # Log token usage
        if hasattr(response, 'usage'):
            u = response.usage
            logger.debug(f"Tokens: {u.prompt_tokens}↑ + {u.completion_tokens}↓ = {u.total_tokens}")
            logger.debug(f"Request time: {request_time:.2f} seconds")
            
            # Store token usage in database if available
            if self.db_handler:
                try:
                    self.db_handler.store_token_usage(
                        prompt=prompt,
                        prompt_tokens=u.prompt_tokens,
                        completion_tokens=u.completion_tokens,
                        total_tokens=u.total_tokens,
                        model=response.model if hasattr(response, 'model') else "local-model",
                        request_time=request_time
                    )
                except Exception as e:
                    logger.error(f"Failed to store token usage in database: {e}")
        
        return result
    
    def __call__(self, prompt: str) -> str:
        """Allow using the client instance as a callable.

        Provides a convenient shorthand for calling generate() directly
        on the client instance.

        Args:
            prompt: The input prompt to send to the LLM.

        Returns:
            The generated text response from the LLM.
        """
        return self.generate(prompt)
