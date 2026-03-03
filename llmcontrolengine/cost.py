"""
cost.py — Model pricing registry and cost calculation logic.

Responsibilities:
- Store per-model pricing (input/output per 1K tokens)
- Provide a single calculate_cost() function
- Keep pricing data centralized and easily updatable
"""

# Pricing in USD per 1,000 tokens.
# Source: public model provider pricing pages (approximate at package release).
# Update this dict when provider prices change.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4":             {"input": 0.03,    "output": 0.06},
    "gpt-4-turbo":       {"input": 0.01,    "output": 0.03},
    "gpt-4o":            {"input": 0.005,   "output": 0.015},
    "gpt-3.5-turbo":     {"input": 0.0005,  "output": 0.0015},
    "claude-3-opus":     {"input": 0.015,   "output": 0.075},
    "claude-3-sonnet":   {"input": 0.003,   "output": 0.015},
    "claude-3-haiku":    {"input": 0.00025, "output": 0.00125},
    "gemini-pro":        {"input": 0.00025, "output": 0.0005},
    "default":           {"input": 0.01,    "output": 0.03},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the estimated cost for a model execution.

    Args:
        model:         Model name string (case-insensitive).
        input_tokens:  Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.

    Returns:
        Estimated cost in USD, rounded to 6 decimal places.
        Falls back to 'default' pricing if model is not found in registry.
    """
    pricing = MODEL_PRICING.get(model.lower(), MODEL_PRICING["default"])
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 6)
