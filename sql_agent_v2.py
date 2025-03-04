#%%
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")
# %%
from typing_extensions import TypedDict
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str
#%%
import getpass
import os
import json
#%%
with open('openai_key.json', 'r') as file:
    data = json.load(file)
os.environ['OPENAI_API_KEY'] = data["api_key"]
#%%
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
#%%
from langchain import hub

query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

assert len(query_prompt_template.messages) == 1
query_prompt_template.messages[0].pretty_print()
#%%
from typing_extensions import Annotated


class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]
#%%
print(db.get_table_info())
print(db.dialect)

#%%
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}
#%%
write_query({"question": "How many Employees are there?"})
# %%
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}
# %%
execute_query({"query": "SELECT COUNT(EmployeeId) AS EmployeeCount FROM Employee;"})
# %%
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}
# %%
from langgraph.graph import START,END, StateGraph

#graph_builder = StateGraph(State).add_sequence(
#    [write_query, execute_query, generate_answer]
#)
#graph_builder.add_edge(START, "write_query")
#graph = graph_builder.compile()

graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_answer")
graph_builder.add_edge("generate_answer", END)
graph = graph_builder.compile()
# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
# %%
for step in graph.stream(
    {"question": "List the top 5 customers who spent the most money , also list their names"}, stream_mode="updates"
):
    print(step)
# %%
