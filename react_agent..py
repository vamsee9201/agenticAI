#%%
import json

# Open and read the JSON file
with open('openai_key.json', 'r') as file:
    data = json.load(file)  # Load JSON data as a Python dictionary

# Print the entire JSON data
print(data["api_key"])
#%%
