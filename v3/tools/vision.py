"""
L2 Vision Tool - Uses Gemini's native vision capabilities.
"""
import os
import base64
from typing import Optional, Dict, Any, Union
import asyncio
from pathlib import Path


class VisionTool:
    """
    L2 Tool: Image understanding using Gemini Vision API.

    This provides REAL vision capabilities, not simulated.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize vision tool.

        Args:
            api_key: Gemini API key
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self._client = None
        self._model = None

    def _get_client(self):
        """Lazily initialize the Gemini client."""
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            self._model = genai.GenerativeModel('gemini-2.5-flash')
            self._client = genai
        return self._model

    def _load_image(self, image_source: Union[str, bytes]) -> Any:
        """Load image from path, URL, or bytes."""
        import google.generativeai as genai

        if isinstance(image_source, bytes):
            # Raw bytes
            return {'mime_type': 'image/jpeg', 'data': base64.b64encode(image_source).decode()}

        if isinstance(image_source, str):
            if image_source.startswith('data:'):
                # Data URL
                parts = image_source.split(',', 1)
                mime_type = parts[0].split(':')[1].split(';')[0]
                data = parts[1] if len(parts) > 1 else ''
                return {'mime_type': mime_type, 'data': data}

            elif image_source.startswith(('http://', 'https://')):
                # URL - download first
                import httpx
                response = httpx.get(image_source)
                mime_type = response.headers.get('content-type', 'image/jpeg')
                return {
                    'mime_type': mime_type,
                    'data': base64.b64encode(response.content).decode()
                }

            elif os.path.exists(image_source):
                # File path
                path = Path(image_source)
                suffix = path.suffix.lower()
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(suffix, 'image/jpeg')

                with open(image_source, 'rb') as f:
                    data = base64.b64encode(f.read()).decode()
                return {'mime_type': mime_type, 'data': data}

            else:
                # Assume base64 encoded
                return {'mime_type': 'image/jpeg', 'data': image_source}

        raise ValueError(f"Cannot load image from: {type(image_source)}")

    async def execute(
        self,
        question: str,
        image: Optional[Union[str, bytes]] = None
    ) -> str:
        """
        Answer a question about an image.

        Args:
            question: Question about the image
            image: Image source (path, URL, base64, or bytes)

        Returns:
            Answer to the question
        """
        model = self._get_client()

        if image is None:
            # No image - fall back to text-only
            return await self._text_only(question)

        try:
            # Load image
            image_data = self._load_image(image)

            # Prepare content
            content = [
                image_data,
                question
            ]

            # Run in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(content)
            )

            return response.text if response.text else str(response)

        except Exception as e:
            return f"Vision error: {e}"

    async def _text_only(self, question: str) -> str:
        """Handle text-only question (no image provided)."""
        model = self._get_client()

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: model.generate_content(question)
            )
            return response.text if response.text else str(response)
        except Exception as e:
            return f"Error: {e}"

    async def analyze_chart(
        self,
        image: Union[str, bytes],
        chart_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Analyze a chart or graph.

        Args:
            image: Chart image
            chart_type: Type hint (bar, line, pie, etc.)

        Returns:
            Dict with 'description', 'data_points', 'trends'
        """
        prompt = f"""Analyze this chart/graph in detail.
Chart type hint: {chart_type}

Please provide:
1. A description of what the chart shows
2. Key data points or values you can read from the chart
3. Any trends or patterns you observe
4. The main takeaway or conclusion

Be specific and extract actual numbers when visible."""

        result = await self.execute(prompt, image)

        return {
            'analysis': result,
            'chart_type': chart_type,
        }

    async def extract_text(self, image: Union[str, bytes]) -> str:
        """
        Extract text from an image (OCR).

        Args:
            image: Image containing text

        Returns:
            Extracted text
        """
        prompt = """Extract all text visible in this image.
Return the text exactly as it appears, preserving layout where possible.
If no text is visible, respond with 'No text found'."""

        return await self.execute(prompt, image)
