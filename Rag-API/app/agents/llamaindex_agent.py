from dotenv import load_dotenv
import dspy
load_dotenv()

from llama_index.llms.openai import OpenAI


import dspy
from dspy import InputField, OutputField, Signature, Tool
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
# 1. Create the OpenAI LLM wrapper
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)


# 2. Define your functions/tools
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b

def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

# Wrap tools if you like, but DSPy will wrap for you:
tools = [add, multiply]

# 3. Build a Signature that includes your system instructions
sig = Signature(
    {"question": InputField()}, 
    instructions=(
        "SYSTEM: You are a precise math assistant. "
        "Always show your reasoning before giving the numeric result."
    )
).append("answer", OutputField(), type_=float)

# 4. Create ReAct with this signature
react = dspy.ReAct(signature=sig, tools=tools)

# 5. Call the agent
pred = react(question="what is 2+3")
print(pred.answer)







