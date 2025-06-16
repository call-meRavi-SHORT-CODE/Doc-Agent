from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import FunctionAgent

def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b

llm = OpenAI(model="gpt-4o-mini")


def llamaindex_agent(llm,system_prompt,tools):


    llam_agent = FunctionAgent(
        tools=[multiply, add],
        llm=llm,
        system_prompt="You are an agent that can perform basic mathematical operations using tools.",
    )




