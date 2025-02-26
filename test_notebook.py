#%%
import mysql.connector

import json

# Open and read the JSON file
with open('sql_password.json', 'r') as file:
    data = json.load(file)  # Load JSON data as a Python dictionary

# Print the entire JSON data
#print(data["sql_password"])
#%%
cnx = mysql.connector.connect(user='root', password='Maxverstappen@33',
                              host='127.0.0.1',
                              database='HRDB')
cursor = cnx.cursor()
#%%
query = ("SELECT * FROM employees")

cursor.execute(query)

# %%
print(cursor)
results = cursor.fetchall()
# %%
print(results)

# %%
