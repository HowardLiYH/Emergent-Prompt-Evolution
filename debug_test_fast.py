#!/usr/bin/env python3
"""Debug test - fast requests (no delay) to reproduce 502 errors."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from genesis.llm_client import LLMClient
from genesis.config import LLMConfig

CONFIG = LLMConfig(
    provider='openai',
    api_key='sk-6o83BXFATUyr0Y8CJw5ufBFzuNT3CfQy4AABn8AJlLg5GI6b',
    api_base='http://123.129.219.111:3000/v1',
    model='gpt-4o-mini',
    max_retries=3,
    timeout=60,
)

async def test_fast_requests():
    print("=" * 50)
    print("DEBUG TEST: Fast Requests (No Delay)")
    print("=" * 50)
    print("Testing 5 rapid sequential requests...")
    print()

    client = LLMClient(CONFIG)

    # Run 5 requests with NO delay
    for i in range(5):
        print(f"Request {i+1}/5...", end=" ", flush=True)
        try:
            response = await client.generate("Say 'test'", max_tokens=10)
            print(f"OK: {response.strip()}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
        # NO DELAY - immediate next request

    print()
    print(client.get_usage_report())
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_fast_requests())
