#%%
import json
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#%%
with open('openai_key.json', 'r') as file:
    data = json.load(file)
os.environ['OPENAI_API_KEY'] = data["api_key"]
#%%
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
#vectorstore.persist()

#%%
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

tools = [retriever_tool]
#%%
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
#%%
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition
#%%
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
def rewrite_question(state):
    """
    To maintain a conversation , If there is a conversation history this tool takes the latest question and conversation history and rewrites the question.
    """
    messages = state["messages"]
    print(messages)
    conversations = []
    for message in messages:
        print("first message",message)
        if isinstance(message, HumanMessage) or isinstance(message,AIMessage):
            conversations.append(message)
    print("conversations",conversations)
    conversation_string = ""
    for conversation in conversations:
        prefix = ""
        if isinstance(conversation, HumanMessage):
            prefix = "Human"
        else :
            prefix = "AI"
        temp_convo = f"""{prefix} : {conversation.content}\n"""
        conversation_string+=temp_convo
    prompt = f"""
    Using this conversation history and the latest human question. use the context to rewrite the human question to capture meaning.
    conversation history : {conversation_string}
    """
    print("prompt",prompt)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    response = llm.invoke(prompt)
    #print(conversation_string)
    return {"messages":response.content}
#%%
sample_state = {"messages":[HumanMessage(content='who is lilian weng?', additional_kwargs={}, response_metadata={}, id='0c4e624d-4422-4b7d-b925-ad76d65f1471'), AIMessage(content='Lilian Weng is a researcher and writer known for her work in the field of artificial intelligence, particularly in areas related to large language models (LLMs), prompt engineering, and adversarial attacks on LLMs. She maintains a blog where she shares insights, research findings, and discussions on these topics, contributing to the understanding and development of AI technologies. Her work is influential in the AI community, especially among those interested in the practical applications and implications of LLMs.', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_709714d124'}, id='run-910551a8-fd3d-423c-a8cb-71b0c829b1c5-0'), HumanMessage(content='what did she say about agent memory?', additional_kwargs={}, response_metadata={}, id='dfcc8400-9a7e-4101-be82-06b8dd6f3256')]}
rewrite_question(sample_state)
#%%

def agent(state):
    messages = state["messages"]
    print("This is agent node")
    print("full messages",messages)
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o-mini")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    return {"messages": [response]}


def retriever_tool():


    return ""

def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    print("This is generate node")
    print("full messages :",messages)
    question = messages[0].content
    print("question:",question)
    last_message = messages[-1]
    docs = last_message.content
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, streaming=True)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

#%%
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)

#%%
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
#%%
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent)  # agent
workflow.add_node("rewrite",rewrite_question) # testing rewrite question
retrieve = ToolNode([retriever_tool])
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)



workflow.add_edge(START, "agent")
#workflow.add_edge("agent", "retrieve")
workflow.add_conditional_edges("agent",)
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_edge("retrieve","generate")
workflow.add_edge("generate",END)
# %%
from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()
graph = workflow.compile()



#%%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
# %%
import pprint

#What does Lilian Weng say about the types of agent memory??
inputs = {
    "messages": [
        ("user", "What does Lilian Weng say about the types of agent memory??"),
    ]
}
for output in graph.stream(inputs):
    for key, value in output.items():
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint("---")
        pprint.pprint(value, indent=2, width=80, depth=None)
    pprint.pprint("\n---\n")
# %%

#next step is to add another node (retrieve node) and see what is happening.

from langchain_core.messages import HumanMessage

config = {"configurable": {"thread_id": "4"}}
input_message = HumanMessage(content="What does Lilian Weng say about the types of agent memory?")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

#%%
input_message = HumanMessage(content="what did she say about agent memory?")
for event in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    event["messages"][-1].pretty_print()

# %%
from langchain_core.messages import HumanMessage
if isinstance(input_message, HumanMessage):
    print("input_message is an instance of HumanMessage")
else:
    print("input_message is NOT an instance of HumanMessage")
#%%
print(input_message is HumanMessage)
# %%
