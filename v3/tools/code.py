"""
L1 Code Execution Tool - Uses Gemini's native code execution.
"""
import os
from typing import Optional, Dict, Any
import asyncio


class CodeExecutionTool:
    """
    L1 Tool: Python code execution using Gemini's code execution API.

    This provides REAL code execution, not simulated.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize code execution tool.

        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self._client = None
        self._model = None

    def _get_client(self):
        """Lazily initialize the Gemini client with code execution."""
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            # Configure model with code execution enabled
            self._model = genai.GenerativeModel(
                model_name='gemini-2.5-flash',
                tools='code_execution'
            )
            self._client = genai
        return self._model

    async def execute(
        self,
        code: Optional[str] = None,
        prompt: Optional[str] = None,
        timeout: int = 30
    ) -> str:
        """
        Execute Python code or solve a coding problem.

        If code is provided, execute it directly.
        If prompt is provided, generate and execute code to solve it.

        Args:
            code: Python code to execute
            prompt: Problem description (will generate code)
            timeout: Execution timeout in seconds

        Returns:
            Execution result or generated answer
        """
        model = self._get_client()

        if code:
            # Direct code execution
            full_prompt = f"""Execute this Python code and return the result:

```python
{code}
```

Return the output of the code execution."""
        elif prompt:
            # Generate and execute code to solve the problem
            full_prompt = f"""Solve this problem by writing and executing Python code:

{prompt}

Write Python code to solve this problem, execute it, and return the final answer."""
        else:
            return "Error: No code or prompt provided"

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(full_prompt)
            )

            # Extract the result
            if response.text:
                return response.text

            # Check for code execution results in parts
            for part in response.parts:
                if hasattr(part, 'executable_code'):
                    code = part.executable_code.code
                    return f"Executed code:\n```python\n{code}\n```"
                if hasattr(part, 'code_execution_result'):
                    output = part.code_execution_result.output
                    return f"Output: {output}"

            return str(response)

        except Exception as e:
            return f"Code execution error: {e}"

    async def execute_with_context(
        self,
        problem: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code with additional context.

        Args:
            problem: The problem to solve
            context: Additional context or constraints

        Returns:
            Dict with 'answer', 'code', 'output'
        """
        full_prompt = problem
        if context:
            full_prompt = f"Context: {context}\n\nProblem: {problem}"

        result = await self.execute(prompt=full_prompt)

        return {
            'answer': self._extract_answer(result),
            'code': self._extract_code(result),
            'output': result
        }

    def _extract_answer(self, result: str) -> str:
        """Extract final answer from result."""
        result_lower = result.lower()

        # Look for answer patterns
        for prefix in ['the answer is', 'result:', 'output:', 'final answer:']:
            if prefix in result_lower:
                idx = result_lower.index(prefix) + len(prefix)
                answer = result[idx:].strip()
                # Take first line or sentence
                for sep in ['\n', '.', ',']:
                    if sep in answer:
                        answer = answer.split(sep)[0].strip()
                        break
                return answer

        return result[:200]

    def _extract_code(self, result: str) -> Optional[str]:
        """Extract code block from result."""
        if '```python' in result:
            start = result.index('```python') + 9
            end = result.index('```', start) if '```' in result[start:] else len(result)
            return result[start:end].strip()
        return None
