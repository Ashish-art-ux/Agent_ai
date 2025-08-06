import json
from google.generativeai.types import Tool, FunctionDeclaration
import google.generativeai as genai
from dotenv import load_dotenv
import os
from google.colab import userdata

load_dotenv()
api_key = userdata.get('GOOGLE_API_KEY')
genai.configure(api_key=api_key)

def add(a, b):
    return a + b

tools = [
    Tool(
        function_declarations=[
            FunctionDeclaration(
                name="add",
                description="Add two numbers",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "a": {"type": "NUMBER"},
                        "b": {"type": "NUMBER"},
                    },
                    "required": ["a", "b"],
                }
            )
        ]
    )
]

messages = [
    {"role": "system", "content": "You are a helpful assistant that can add two numbers."}
]
user_input = input("Tell me which numbers you want to add (e.g., 'Add 5 and 7'): ")
messages.append({"role": "user", "content": user_input})

model = genai.GenerativeModel("gemini-2.5-flash")

while True:
    response = model.generate_content(
        contents=messages,
        tools=tools
    )

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call.function.name
        argument = json.loads(tool_call.function.arguments)

        print(f"Model wants to call: {tool_name} with arguments {argument}")

        if tool_name == "add":
            result = add(**argument)
        else:
            result = "Unknown function"

        print(f"Function output: {result}")

        # Append assistant's function call message
        messages.append({"role": "assistant", "content": response.text, "tool_calls": response.tool_calls})
        # Append result from tool
        messages.append({
            'role': "tool",
            "parts": [{"text": json.dumps(result)}],
            "name": tool_name,
        })

    else:
        print("Final answer from model:")
        print(response.text)
        break
