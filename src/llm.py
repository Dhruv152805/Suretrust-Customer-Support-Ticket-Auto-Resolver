"""
LLM integration module for Ollama.
Generates customized responses based on retrieved ticket solutions.
"""
import ollama
from src.config import OLLAMA_MODEL, OLLAMA_HOST, logger

class OllamaGenerator:
    """
    Handles communication with Ollama for generating customized responses.
    """
    def __init__(self, model_name: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.client = ollama.Client(host=host)
        logger.info(f"Initialized OllamaGenerator with model: {model_name} at {host}")

    def generate_response(self, query: str, context: list[dict]) -> str:
        """
        Generate a personalized response based on the query and retrieved context.
        """
        if not context:
            return "I'm sorry, I couldn't find any relevant solutions in our database to help with that."

        # Format the context for the prompt
        context_str = ""
        for i, item in enumerate(context, 1):
            context_str += f"\nSolution {i}:\nIssue: {item['similar_issue']}\nResponse: {item['suggested_solution']}\n"

        prompt = f"""
You are a helpful customer support assistant. A customer has sent the following query:
"{query}"

Based on similar previously resolved tickets provided below, generate a personalized, polite, and concise response to the customer. 
Do not mention "Solution 1" or "Solution 2" directly, but synthesize the best advice into a single helpful reply.

IMPORTANT: Output ONLY the customer-facing response. Do NOT include any meta-commentary like "Here's a response:" or explanations about what the response aims to do. Start directly with the customer response.

Similar Resolved Tickets:
{context_str}

Your Response:
"""
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt)
            cleaned = self._clean_meta_commentary(response['response'].strip())
            return cleaned
        except Exception as e:
            logger.warning(f"Ollama generation failed (service might be down): {str(e)}")
            return self._heuristic_fallback(query, context)

    def _clean_meta_commentary(self, text: str) -> str:
        """
        Remove common meta-commentary patterns that LLMs sometimes add.
        """
        import re
        
        # Split by newlines and process
        lines = text.split('\n')
        
        # Skip leading meta-commentary lines
        start_idx = 0
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            # Skip lines that are meta-commentary
            if any([
                line_lower.startswith("here's"),
                line_lower.startswith("here is"),
                line_lower.startswith("this response"),
                line_lower.startswith("this reply"),
                line_lower.startswith("response:"),
                line_lower.startswith("reply:"),
                line_lower.startswith("* "),
                line_lower.startswith("- "),
                line_lower == "",
            ]):
                start_idx = i + 1
            else:
                break
        
        # Take only the actual response
        cleaned_lines = lines[start_idx:]
        result = '\n'.join(cleaned_lines)
        
        # Also remove trailing meta-commentary (bullet lists explaining what the response does)
        if '\n*' in result or '\nThis response' in result or '\nThis aims' in result:
            # Find where meta-explanation starts
            meta_patterns = [
                r'\n\s*This (?:response|reply) (?:aims to|will|is designed to):',
                r'\n\s*\*\s+(?:Acknowledge|Emphasize|Provide|Encourage)',
            ]
            for pattern in meta_patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    result = result[:match.start()]
                    break
        
        return result.strip()

    def _heuristic_fallback(self, query: str, context: list[dict]) -> str:
        """
        Rich fallback: synthesize unique action steps from all top matches
        into a polished, query-aware customer response.
        """
        import re

        if not context:
            return "I've looked through our records but couldn't find a specific solution. Please contact a support agent for further assistance."

        top = context[0]
        category = top.get('category', 'your request').replace('_', ' ').title()

        # Collect unique, non-trivial solution sentences across all matches
        seen = set()
        action_points = []
        for s in context:
            sol = s.get('suggested_solution', '').strip()
            sentences = re.split(r'(?<=[.!?])\s+', sol)
            for sent in sentences:
                sent = sent.strip()
                key = sent.lower()[:60]
                if len(sent) > 20 and key not in seen:
                    seen.add(key)
                    action_points.append(sent)
                if len(action_points) >= 4:
                    break
            if len(action_points) >= 4:
                break

        steps = "\n".join(f"{i+1}. {pt}" for i, pt in enumerate(action_points))

        return (
            f"Thank you for reaching out. I understand you're having an issue with: \"{query[:120]}\"\n\n"
            f"Based on similar {category} cases we've resolved, here are the recommended steps:\n\n"
            f"{steps}\n\n"
            f"Please try the above steps in order. If the issue persists, reply here and a support "
            f"specialist will review your case directly."
        )

if __name__ == "__main__":
    # Quick test
    gen = OllamaGenerator()
    test_query = "How do I cancel my subscription?"
    test_context = [
        {"similar_issue": "I want to stop my monthly plan", "suggested_solution": "Go to settings -> subscription and click cancel."}
    ]
    print(gen.generate_response(test_query, test_context))
