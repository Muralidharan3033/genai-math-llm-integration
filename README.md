# Integration of Mathematical Calculations with a Chat Completion System

# AIM
To design and implement a Python function for calculating the volume of a cylinder and integrate it with a chat completion system utilizing the function-calling feature of a Large Language Model (LLM).

# PROBLEM STATEMENT

In many engineering and scientific applications, mathematical computations must be performed dynamically based on natural language inputs. Traditional chat systems cannot execute real-time calculations. This experiment addresses that gap by integrating a Python-based cylinder volume calculator with Claude (Anthropic LLM) using its function-calling (tool-use) feature, enabling the model to invoke the local Python function and return accurate computed results within a conversational interface.

# DESIGN STEPS

**STEP 1 — Define the Mathematical Function**
Implement `calculate_cylinder_volume(radius, height)` in Python using the formula $V = \pi \times r^2 \times h$ from the `math` module. Add input validation to reject non-positive values and return a structured dictionary containing the radius, height, computed volume (rounded to 4 decimal places), and the formula used.

**STEP 2 — Register the Function as an LLM Tool**
Describe the Python function to the LLM using a JSON tool schema. This schema specifies the tool's name, a natural-language description, and the expected input parameters with their types and constraints. Claude uses this schema to decide when and how to call the function automatically.

**STEP 3 — Implement the Agentic Chat Loop**
Build a `chat_with_function_calling()` function that:
- Sends the user's natural-language message to Claude along with the tool schema.
- Detects when Claude returns `stop_reason = "tool_use"` (i.e., it wants to call the function).
- Executes the local Python function with Claude's extracted arguments.
- Returns the result back to Claude in a `tool_result` message.
- Receives Claude's final natural-language answer with the computed volume embedded.

# PROGRAM

```python
import json
import math
import os
import anthropic


# STEP 1: Mathematical Function
def calculate_cylinder_volume(radius: float, height: float) -> dict:
    """Calculate volume of a cylinder: V = π × r² × h"""
    if radius <= 0 or height <= 0:
        raise ValueError("Both radius and height must be positive numbers.")
    volume = math.pi * (radius**2) * height
    return {
        "radius": radius,
        "height": height,
        "volume": round(volume, 4),
        "formula": "V = π × r² × h",
    }


# STEP 2: Tool Schema
tools = [
    {
        "name": "calculate_cylinder_volume",
        "description": "Calculates the volume of a cylinder using V = π × r² × h.",
        "input_schema": {
            "type": "object",
            "properties": {
                "radius": {
                    "type": "number",
                    "description": "Radius of the circular base (must be > 0)",
                },
                "height": {
                    "type": "number",
                    "description": "Height of the cylinder (must be > 0)",
                },
            },
            "required": ["radius", "height"],
        },
    }
]

# Tool dispatcher
def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    if tool_name == "calculate_cylinder_volume":
        result = calculate_cylinder_volume(
            tool_input["radius"], tool_input["height"]
        )
        return json.dumps(result)
    return json.dumps({"error": "Unknown tool"})


# STEP 3: Agentic Chat Loop
def chat_with_function_calling(user_message: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    print(f"User         : {user_message}")

    # First API call
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    # Agentic loop — repeat until no more tool calls
    while response.stop_reason == "tool_use":
        tool_use_blocks = [
            b for b in response.content if b.type == "tool_use"
        ]
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for b in tool_use_blocks:
            print(f"Tool Called  : {b.name}")
            print(f"Arguments    : {b.input}")
            output = dispatch_tool(b.name, b.input)
            print(f"Tool Result  : {output}")
            tool_results.append(
                {"type": "tool_result", "tool_use_id": b.id, "content": output}
            )

        messages.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

    final = "".join(
        block.text for block in response.content if hasattr(block, "text")
    )
    print(f"Assistant    : {final}")
    return final


# Run
if __name__ == "__main__":
    API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    chat_with_function_calling(
        "What is the volume of a cylinder with radius 5 cm and height 10 cm?",
        API_KEY,
    )
```

# OUTPUT

<img width="1437" height="699" alt="image" src="https://github.com/user-attachments/assets/096f3b9e-879a-41b5-a5a2-dfc2c3250f09" />

```plaintext
User         : What is the volume of a cylinder with radius 5 cm and height 10 cm?
Tool Called  : calculate_cylinder_volume
Arguments    : {'radius': 5, 'height': 10}
Tool Result  : {"radius": 5, "height": 10, "volume": 785.3982, "formula": "V = \u03c0 \u00d7 r\u00b2 \u00d7 h"}
Assistant    : The volume of a cylinder with radius 5 cm and height 10 cm is 
               approximately 785.3982 cm³.

               This is calculated using the formula V = π × r² × h:
               V = π × (5)² × 10 = π × 25 × 10 ≈ 785.3982 cm³
```

# RESULT
The experiment was successfully completed. A Python function `calculate_cylinder_volume(radius, height)` was implemented using the formula $V = \pi \times r^2 \times h$ and registered as a callable tool for the Claude LLM. The chat completion system correctly identified the user's intent, invoked the Python function via LLM function-calling, retrieved the computed volume ($785.3982\text{ cm}^3$ for $r = 5$, $h = 10$), and returned a precise natural-language response. This demonstrates successful integration of mathematical function-calling with large language models.
