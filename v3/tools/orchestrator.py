"""
L5 Orchestrator Tool - Multi-step workflow orchestration using LangGraph.
"""
import os
from typing import Optional, List, Dict, Any
import asyncio


class OrchestratorTool:
    """
    L5 Tool: Complex multi-step workflow orchestration.

    Uses LangGraph for state machine-based orchestration.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize orchestrator tool.

        Args:
            api_key: Gemini API key for sub-tasks
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self._model = None
        self._graph = None

    def _get_model(self):
        """Lazily initialize the Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel('gemini-2.5-flash')
        return self._model

    async def execute(
        self,
        task: str,
        max_steps: int = 5
    ) -> str:
        """
        Execute a complex multi-step task.

        Args:
            task: Complex task description
            max_steps: Maximum number of steps

        Returns:
            Final result
        """
        model = self._get_model()

        # Step 1: Decompose the task
        decompose_prompt = f"""You are a task orchestrator. Break down this complex task into a series of steps:

Task: {task}

Provide {max_steps} or fewer steps, each on a new line starting with a number.
Each step should be specific and actionable."""

        try:
            loop = asyncio.get_event_loop()
            decomposition = await loop.run_in_executor(
                None,
                lambda: model.generate_content(decompose_prompt)
            )

            steps = self._parse_steps(decomposition.text)

            if not steps:
                # Fall back to direct execution
                return await self._direct_execute(task)

            # Step 2: Execute each step
            results = []
            context = f"Original task: {task}\n\n"

            for i, step in enumerate(steps[:max_steps], 1):
                step_prompt = f"""{context}
Previous results:
{chr(10).join(results[-2:]) if results else 'None'}

Current step ({i}/{len(steps)}): {step}

Execute this step and provide the result."""

                step_result = await loop.run_in_executor(
                    None,
                    lambda: model.generate_content(step_prompt)
                )

                results.append(f"Step {i} ({step}): {step_result.text[:300]}")

            # Step 3: Synthesize final result
            synthesis_prompt = f"""You completed a multi-step task.

Original task: {task}

Steps completed:
{chr(10).join(results)}

Synthesize the final answer based on all the steps above.
Provide a clear, concise final answer."""

            final = await loop.run_in_executor(
                None,
                lambda: model.generate_content(synthesis_prompt)
            )

            return final.text

        except Exception as e:
            return f"Orchestration error: {e}"

    async def _direct_execute(self, task: str) -> str:
        """Execute task directly without decomposition."""
        model = self._get_model()

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(task)
            )
            return response.text
        except Exception as e:
            return f"Error: {e}"

    def _parse_steps(self, text: str) -> List[str]:
        """Parse numbered steps from text."""
        import re

        steps = []
        lines = text.strip().split('\n')

        for line in lines:
            # Match numbered lines like "1. Step" or "1) Step"
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)$', line)
            if match:
                step = match.group(1).strip()
                if step:
                    steps.append(step)

        return steps

    async def execute_with_tools(
        self,
        task: str,
        available_tools: Dict[str, Any],
        max_steps: int = 5
    ) -> Dict[str, Any]:
        """
        Execute a task with access to other tools.

        Args:
            task: Task to execute
            available_tools: Dict of tool_name -> tool_instance
            max_steps: Maximum steps

        Returns:
            Dict with 'result', 'steps_taken', 'tools_used'
        """
        model = self._get_model()

        tool_descriptions = "\n".join([
            f"- {name}: Available for use"
            for name in available_tools.keys()
        ])

        plan_prompt = f"""You are an orchestrator with access to these tools:
{tool_descriptions}

Task: {task}

Create a plan with up to {max_steps} steps.
For each step, specify if a tool is needed and which one.
Format: "Step N: [TOOL_NAME or NONE] - Description"
"""

        try:
            loop = asyncio.get_event_loop()
            plan = await loop.run_in_executor(
                None,
                lambda: model.generate_content(plan_prompt)
            )

            # Parse and execute plan
            steps_taken = []
            tools_used = []
            context = ""

            for line in plan.text.split('\n'):
                if 'Step' in line and ':' in line:
                    # Parse step
                    parts = line.split(':', 1)
                    if len(parts) < 2:
                        continue

                    step_desc = parts[1].strip()

                    # Check for tool usage
                    tool_used = None
                    for tool_name in available_tools.keys():
                        if tool_name.upper() in step_desc.upper():
                            tool_used = tool_name
                            tools_used.append(tool_name)
                            break

                    if tool_used and tool_used in available_tools:
                        # Execute with tool
                        tool = available_tools[tool_used]
                        result = await tool.execute(step_desc)
                    else:
                        # Execute with LLM
                        result = await self._direct_execute(
                            f"{context}\n\nExecute: {step_desc}"
                        )

                    steps_taken.append({
                        'step': step_desc,
                        'tool': tool_used,
                        'result': str(result)[:200]
                    })

                    context += f"\n{step_desc}: {result[:100]}"

            # Final synthesis
            result = await self._direct_execute(
                f"Task: {task}\n\nCompleted steps:\n{context}\n\nProvide the final answer."
            )

            return {
                'result': result,
                'steps_taken': steps_taken,
                'tools_used': list(set(tools_used))
            }

        except Exception as e:
            return {
                'result': f"Error: {e}",
                'steps_taken': [],
                'tools_used': []
            }
