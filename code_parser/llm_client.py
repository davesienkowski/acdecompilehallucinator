#!/usr/bin/env python3
"""
LLM Client
==========

Simple LLM client using OpenAI-compatible API (e.g., LMStudio)
"""

import logging
from typing import Optional

from code_parser.llm_cache import LLMCache

logger = logging.getLogger("llm-processor")

# Configuration
LM_STUDIO_URL = "http://localhost:1234/v1"
MAX_LLM_TOKENS = 131072


class LLMClient:
    """Simple LLM client using OpenAI-compatible API (e.g., LMStudio)"""
    
    def __init__(self, base_url: str = LM_STUDIO_URL, temperature: float = 0.2, cache: Optional[LLMCache] = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key="lm-studio")
            self.temperature = temperature
            self.cache = cache
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def generate(self, prompt: str, max_tokens: int = MAX_LLM_TOKENS) -> str:
        """Generate response from LLM, checking cache first"""
        # Check cache
        if self.cache:
            cached = self.cache.get(prompt)
            if cached:
                logger.info("Cache hit! Using stored response.")
                return cached

        response = self.client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": "You are a C++ modernization expert. Output ONLY clean code, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
            timeout=300
        )
        
        result = response.choices[0].message.content or ""

        # Store in cache
        if self.cache and result:
            self.cache.set(prompt, result)
        
        # Log token usage
        if hasattr(response, 'usage'):
            u = response.usage
            logger.debug(f"Tokens: {u.prompt_tokens}↑ + {u.completion_tokens}↓ = {u.total_tokens}")
        
        return result
    
    def __call__(self, prompt: str) -> str:
        """Allow using client as callable"""
        return self.generate(prompt)
