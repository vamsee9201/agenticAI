#%%
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
import json
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
#%%
with open('openai_key.json', 'r') as file:
    data = json.load(file)
os.environ['OPENAI_API_KEY'] = data["api_key"]

#%%
# This is vector store for the rag agentic system. 
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()
#%%
memory = MemorySaver()
#%%


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

#tools , tools with model 
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.messages import AIMessage

"""
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)
"""

def retrieve_node(state):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(state["user_question"], k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return {"messages":[serialized]}

def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    print("This is generate node")
    print("full messages :",messages)
    question = state["user_question"]
    print("question:",question)
    last_message = messages[-1]
    docs = last_message.content
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    print("response:",response)
    return {"messages": [AIMessage(content=response)]}





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

#workflow.add_node("call_model",call_model)
workflow.add_node("update_user_question",update_user_question)
workflow.add_node("retrieve_node",retrieve_node)
workflow.add_node("generate",generate)

#  workflow.add_edge(START, "call_model")
workflow.add_edge(START,"update_user_question")
workflow.add_edge("update_user_question","retrieve_node")
workflow.add_edge("retrieve_node","generate")
workflow.add_edge("generate",END)
app = workflow.compile(checkpointer=memory)
#%%
from IPython.display import Image, display
display(Image(app.get_graph(xray=True).draw_mermaid_png()))
# %%
from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "1"}}
input_message = HumanMessage(content="who is lilian weng")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
#%%

input_message = HumanMessage(content="what did she say about agent memory?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
app.get_state(config=config)

# %%

input_message = HumanMessage(content="what did she say about adversarial attacks?")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
app.get_state(config=config)

#testing commits
#Now i have to add a tool to the model that can retrieve from a vector database. 

# %%
input_message = HumanMessage(content="Thank you")
for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()
app.get_state(config=config)

# %%
