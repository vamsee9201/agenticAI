#%%
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
import json
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

memory = MemorySaver()
#%%
with open('openai_key.json', 'r') as file:
    data = json.load(file)
os.environ['OPENAI_API_KEY'] = data["api_key"]

model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
#%%
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_question:str

#%%


def update_user_question(state):
    #print(str(state["messages"]))
    conversation_history = "\n".join(
        f"{'Human:' if isinstance(message, HumanMessage) else 'AI:'} {message.content}" 
        for message in state["messages"]
    )
    print("conversation history :",conversation_history + "end of conversation history")
    prompt = f"""
    Rewrite the latest human question to be fully self-contained while preserving its meaning. 
    If it refers to context from the conversation, replace pronouns or vague terms with the correct reference. 
    If it is already clear, keep it unchanged. Do not add explanations or modify intent.      
    Conversation history: 
    {conversation_history}
    """
    response = model.invoke(prompt)
    return {"user_question":response.content}

#tools = [update_user_question]

#model_tools = model.bind_functions(tools)
def call_model(state):
    print("full messages",state["messages"])
    try:
        print("user question",state["user_question"])
    except:
        pass
    response = model.invoke(state["messages"])
    return {"messages":response}
    


#%%
from langgraph.prebuilt import tools_condition
workflow = StateGraph(AgentState)

workflow.add_node("call_model",call_model)
workflow.add_node("update_user_question",update_user_question)

workflow.add_edge(START, "call_model")
workflow.add_edge("call_model","update_user_question")
workflow.add_edge("update_user_question",END)
app = workflow.compile(checkpointer=memory)
#%%
from IPython.display import Image, display
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
# %%
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "17"}}
input_message = HumanMessage(content="who is ellyse perry")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
#%%

input_message = HumanMessage(content="how old is she?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
app.get_state(config=config)

# %%

#testing commits
