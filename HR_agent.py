#%%

import json
import os 
from langchain_openai import ChatOpenAI
import mysql.connector  # Added import for MySQL connector

#%%
# Open and read the JSON file
with open('openai_key.json', 'r') as file:
    data = json.load(file)  # Load JSON data as a Python dictionary
#set open_ai key as an environment variable
os.environ['OPENAI_API_KEY'] = data["api_key"]

# Open and read the JSON file for SQL password
with open('sql_password.json', 'r') as file:  # Added code to read SQL password
    sql_data = json.load(file)  # Load SQL JSON data as a Python dictionary

#%%
llm = ChatOpenAI(model="gpt-4o")

cnx = mysql.connector.connect(user='root', password=sql_data["sql_password"],  # Added SQL connection
                              host='127.0.0.1',
                              database='HRDB')
cursor = cnx.cursor()
#%%

#tool
def get_employee_email(name:str):
    """
    This tool is used to get the employee data. 
    """
    normalized_name = name.lower()
    query = f"SELECT employee_email FROM employees WHERE LOWER(employee_name) = '{normalized_name}'"
    cursor.execute(query)
    result = cursor.fetchone()
    if result:
        return {"messages": result[0]}
    else:   
        return {"messages": "Employee not found"}

#%%
def generate_sql_query(query:str):
    """
    This tool is used to generate a sql query for employee data.
    employee data is stored in the employees_data table.
    the table has the following columns: employee_id, employee_name, employee_email, employee_phone, employee_address, employee_city, employee_state, employee_zip, employee_country, employee_department, employee_job_title, employee_salary, employee_hire_date, employee_termination_date
    """
    print("This is get sql query tool")
    response = llm.invoke("Generate a sql query to update employee data based on the following question: " + query + "\n" + "You are an AI that generates SQL queries from natural language questions. Your task is to take a given question and generate only the SQL query, with no explanations, comments, or additional text. The output should be a valid SQL statement formatted properly. Do not include any preamble or extra words—just the SQL query itself.")
    print(response.content)
    return {"messages": response.content}

generate_sql_query("change johnn doe's email to johndoe2@gmail.com")
#%%
#%%

#llm with tools
#tools = [get_employee_email]
tools = [generate_sql_query]
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

messages = [HumanMessage(content="change jogn does email to johndoe2@gmail.com")]
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

#%%


#agent needs to know where the data is. 
#we should have scripts to do everything : get description, function to add the description to the tool. 

