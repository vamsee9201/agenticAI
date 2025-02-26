#%%
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal

memory = MemorySaver()
#%%
with open('openai_key.json', 'r') as file:
    data = json.load(file)
os.environ['OPENAI_API_KEY'] = data["api_key"]

model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
#%%
def call_model(state:MessagesState):
    print("full messages",state["messages"])
    response = model.invoke(state["messages"])
    return {"messages":response}
#%%

workflow = StateGraph(MessagesState)

workflow.add_node("call_model",call_model)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model",END)
app = workflow.compile(checkpointer=memory)
#%%
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
# %%
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "3"}}
input_message = HumanMessage(content="hi! I'm bob")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
#%%

input_message = HumanMessage(content="what's my name?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%
