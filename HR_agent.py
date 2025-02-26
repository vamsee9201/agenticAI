#%%

import json
import os 
from langchain_openai import ChatOpenAI

#%%
# Open and read the JSON file
with open('openai_key.json', 'r') as file:
    data = json.load(file)  # Load JSON data as a Python dictionary
#set open_ai key as an environment variable
os.environ['OPENAI_API_KEY'] = data["api_key"]
#%%
llm = ChatOpenAI(model="gpt-4o")
#%%

#tool
def get_employee_data():
    """
    This tool is used to get the employee data. 
    """
    return {"messages":"john doe's age is 23"}

#%%

#llm with tools
tools = [get_employee_data]
llm_with_tools = llm.bind_tools(tools)




#%%
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

sys_msg = SystemMessage(content="You are a helpful assistant tasked with getting employee data out of a database.")
def reasoner(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

#%%
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition # this is the checker for the if you got a tool back
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

builder = StateGraph(MessagesState)


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
#%%

messages = [HumanMessage(content="what is john doe's age")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()
#%%









#%%
#defining tools

#ESS (employee self service)

def update_profile():

    return ""

def request_leave():

    return ""


#%%
#MSS (manager self service)

def employee_count():

    return ""

