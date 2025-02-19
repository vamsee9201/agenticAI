#%%
import json

# Open and read the JSON file
with open('openai_key.json', 'r') as file:
    data = json.load(file)  # Load JSON data as a Python dictionary

# Print the entire JSON data
print(data["api_key"])
#%%
import os

os.environ['OPENAI_API_KEY'] = data["api_key"]
#%%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
# %%
def operation_1(a: int, b: int) -> int:
    """
    This is a multiple function
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """
    hello
    """
    return a + b

def operation_2(a: int, b: int) -> float:
    """
    This is a division function
    """
    return a / b
# %%
# search tools
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

search.invoke("How old is Brad Pitt?")
# %%
tools = [add, multiply, divide, search]

llm_with_tools = llm.bind_tools(tools)
# %%
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search and performing arithmetic on a set of inputs.")
# %%

# Node
def reasoner(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}
# %%

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition # this is the checker for the if you got a tool back
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

# Graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools)) # for the tools

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
    # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()

# Display the graph
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
# %%
messages = [HumanMessage(content="What is 2 times Brad Pitt's age?")]
messages = react_graph.invoke({"messages": messages})

# %%
for m in messages['messages']:
    m.pretty_print()
# %%
import yfinance as yf

def get_stock_price(ticker: str) -> float:
    """Gets a stock price from Yahoo Finance.

    Args:
        ticker: ticker str
    """
    # """This is a tool for getting the price of a stock when passed a ticker symbol"""
    stock = yf.Ticker(ticker)
    return stock.info['previousClose']
# %%
get_stock_price("AAPL")
# %%
def reasoner(state):
    query = state["query"]
    messages = state["messages"]
    # System message
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with using search, the yahoo finance tool and performing arithmetic on a set of inputs.")
    message = HumanMessage(content=query)
    messages.append(message)
    result = [llm_with_tools.invoke([sys_msg] + messages)]
    return {"messages":result}
# %%
tools = [add, multiply, divide, search, get_stock_price]

llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools)
# %%
tools[4]
# %%
from typing import Annotated, TypedDict
import operator
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State of the graph."""
    query: str
    finance: str
    final_answer: str
    # intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    messages: Annotated[list[AnyMessage], operator.add]

# %%
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition # this is the checker for the
from langgraph.prebuilt import ToolNode


# Graph
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", ToolNode(tools)) # for the tools

# Add Edges
workflow.add_edge(START, "reasoner")

workflow.add_conditional_edges(
    "reasoner",
    # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
    # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
    tools_condition,
)
workflow.add_edge("tools", "reasoner")
react_graph = workflow.compile()

# Show
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
# %%
response = react_graph.invoke({"query": "What is 2 times Brad Pitt's age?", "messages": []})
# %%
response['messages'][-1].pretty_print()
# %%
response = react_graph.invoke({"query": "What is the stock price of Apple?", "messages": []})
# %%
for m in response['messages']:
    m.pretty_print()
# %%
response = react_graph.invoke({"query": "What is the stock price of the company that Jensen Huang is CEO of?", "messages": []})
# %%
for m in response['messages']:
    m.pretty_print()
# %%
response = react_graph.invoke({"query": "What will be the price of nvidia stock if it doubles?", "messages": []})
# %%
for m in response['messages']:
    m.pretty_print()
# %%
display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
# %%

#reasoner : intent mapping tool, sees the question and executes the intent.
#vertex ai tool calls.