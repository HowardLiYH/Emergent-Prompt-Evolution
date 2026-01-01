#!/usr/bin/env python3
"""Debug test script to diagnose API connection issues."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from genesis.llm_client import LLMClient
from genesis.config import LLMConfig

# Test configuration
CONFIG = LLMConfig(
    provider='openai',
    api_key='sk-6o83BXFATUyr0Y8CJw5ufBFzuNT3CfQy4AABn8AJlLg5GI6b',
    api_base='http://123.129.219.111:3000/v1',
    model='gpt-4o-mini',
    max_retries=3,
    timeout=60,
)

async def test_connection():
    print("=" * 50)
    print("DEBUG TEST: API Connection")
    print("=" * 50)
    print(f"Endpoint: {CONFIG.api_base}")
    print(f"Timeout: {CONFIG.timeout}s")
    print(f"Max retries: {CONFIG.max_retries}")
    print()

    client = LLMClient(CONFIG)

    # Run 5 sequential requests with 2 second gaps
    for i in range(5):
        print(f"Request {i+1}/5...", end=" ", flush=True)
        try:
            response = await client.generate("Say 'test' in one word.", max_tokens=10)
            print(f"OK: {response.strip()}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")

        await asyncio.sleep(2)  # 2 second gap between requests

    print()
    print(client.get_usage_report())
    print()
    print("Debug logs written to: /Users/yuhaoli/code/MAS_For_Finance/MAS_Final_With_Agents/.cursor/debug.log")

    await client.close()

if __name__ == "__main__":
    asyncio.run(test_connection())
