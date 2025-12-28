#!/usr/bin/env python3
"""
Script to query token usage statistics from the database
"""
import sqlite3
import argparse
from pathlib import Path

def calculate_cost(total_prompt_tokens, total_completion_tokens, input_cost_per_mtok, output_cost_per_mtok):
    """Calculate cost based on token counts and pricing per MTok"""
    input_cost = (total_prompt_tokens or 0) * (input_cost_per_mtok / 1_000_000)
    output_cost = (total_completion_tokens or 0) * (output_cost_per_mtok / 1_000_000)
    return input_cost + output_cost

def query_token_usage(db_path: Path):
    """Query token usage statistics from the database"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Query to get total prompt, total response, avg prompt, avg response, and total prompt request count
    query = """
    SELECT
        SUM(prompt_tokens) as total_prompt_tokens,
        SUM(completion_tokens) as total_completion_tokens,
        AVG(prompt_tokens) as avg_prompt_tokens,
        AVG(completion_tokens) as avg_completion_tokens,
        COUNT(*) as total_requests,
        SUM(request_time) as total_request_time,
        AVG(request_time) as avg_request_time
    FROM llm_token_usage
    """
    
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result:
        print(f"Total Prompt Tokens:       {result['total_prompt_tokens'] or 0}")
        print(f"Total Completion Tokens:   {result['total_completion_tokens'] or 0}")
        print(f"Average Prompt Tokens:     {result['avg_prompt_tokens'] or 0:.2f}")
        print(f"Average Completion Tokens: {result['avg_completion_tokens'] or 0:.2f}")
        print(f"Total Requests:            {result['total_requests'] or 0}")
        
        # Display timing statistics if available
        total_request_time = result['total_request_time']
        avg_request_time = result['avg_request_time']
        if total_request_time is not None:
            print(f"Total Request Time:        {total_request_time:.2f}s")
            print(f"Average Request Time:      {avg_request_time:.2f}s")
        else:
            print(f"Total Request Time:        N/A")
            print(f"Average Request Time:      N/A")
        
        print("")
        
        # Calculate and display costs for Claude models
        print("\nEstimated Costs for Claude Models:")
        
        # Claude Opus 4.5 pricing: $5/Mtok input, $25/Mtok output
        opus_cost = calculate_cost(
            result['total_prompt_tokens'],
            result['total_completion_tokens'],
            5.0, 25.0
        )
        print(f"Claude Opus 4.5:     ${opus_cost:.4f}")
        
        # Claude Sonnet 4.5 pricing: $3/Mtok input, $15/Mtok output
        sonnet_cost = calculate_cost(
            result['total_prompt_tokens'],
            result['total_completion_tokens'],
            3.0, 15.0
        )
        print(f"Claude Sonnet 4.5:   ${sonnet_cost:.4f}")
        
        # Claude Haiku 4.5 pricing: $1/Mtok input, $5/Mtok output
        haiku_cost = calculate_cost(
            result['total_prompt_tokens'],
            result['total_completion_tokens'],
            1.0, 5.0
        )
        print(f"Claude Haiku 4.5:    ${haiku_cost:.4f}")
        
    else:
        print(f"No token usage data found in database: {db_path}")
    
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Query token usage statistics from database")
    parser.add_argument("--db", type=Path, default=Path("mcp-sources/types.db"), help="Path to the database file")
    
    args = parser.parse_args()
    
    if not args.db.exists():
        print(f"Error: Database file does not exist: {args.db}")
        return 1
    
    query_token_usage(args.db)
    return 0

if __name__ == "__main__":
    exit(main())