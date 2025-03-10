# Persisting the networkX data into ArangoDB

from arango import ArangoClient

# Connect to ArangoDB

client = ArangoClient()
db = client.db("_system", username="root", password="2003")

#Building an AI agent

from arango import ArangoClient
from langchain_community.graphs import ArangoGraph
from langchain.chains import ArangoGraphQAChain
from langchain_groq import ChatGroq
import re
from langchain_core.tools import tool
import networkx as nx
import json
import os
import matplotlib.pyplot as plt  # For visualization
import textwrap
import plotly
from groq import Groq


# Set GROQ API key
os.environ["GROQ_API_KEY"] = "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"

# Instantiate the ArangoDB-LangChain Graph
arango_graph = ArangoGraph(db)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),  # This is the default and can be omitted
)


# Tool 1: Translate text to AQL and back to text

@tool
def text_to_aql_to_text(query: str):
    """This tool is available to invoke the
    ArangoGraphQAChain object, which enables you to
    translate a Natural Language Query into AQL, execute
    the query, and translate the result back into Natural Language.
    """
    llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=arango_graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    result = chain.invoke(query)
    return str(result["result"])

# Tool 2: Translate text to AQL, build a NetworkX graph, run an algorithm, and visualize
@tool
def text_to_aql_to_nxalgorithm(query: str):
    """This tool takes a natural language query, generates a sample graph (nodes and edges)
    based on the query context, builds a NetworkX graph, runs an algorithm, and returns
    the raw result (FINAL_RESULT).
    """
    
    graph_prompt = f"""This tool is available to invoke the
    ArangoGraphQAChain object, which enables you to
    translate a Natural Language Query into AQL, execute
    the query : {query}, and translate the result back into Natural Language.
    """
    llm = ChatGroq(temperature=0, model_name="qwen-2.5-32b")
    chain = ArangoGraphQAChain.from_llm(
        llm=llm,
        graph=arango_graph,
        verbose=True,
        allow_dangerous_requests=True
    )
    result = chain.invoke(query)
    result = str(result["result"])


    chat_completion = client.chat.completions.create(
                model="qwen-2.5-coder-32b",
                messages=[
                    {  
                        "role": "system",
                        "content": (
    
                            "You are a Data Visualization Assistant. Your task is to generate only Python code for networkx visualizations. The below Rules to Follow "
                            f"based on user query :{query} from our data {result}. The output must be pure Python code, without any text, imports, explanations."

                            "### Rules to Follow"
                            f"1. Generate a working visualization code for {result} data using the exact format and structure from the provided examples."
                            "2. After generating the code, verify it thoroughly to ensure there are no errors and that it functions as expected."

                            "### Examples "

                            "**Example 1:** "
                            "**query:** 'list all the products of users' "
                            "**result:** 'Summary:'"
                                        'The list of products purchased by users includes the following items:'
                                        '- Wireless Mouse (product ID: products/prod1) was bought twice by one user and once by another, each at a price of $29.99.'
                                        '- Another product (product ID: products/prod2) was purchased once by one user and three times by another, each at a price of $12.50.'

                                        'This summary provides a breakdown of the products that have been purchased by users, detailing the product ID, quantity, and price for each purchase.'

                            "**Expected Model Output:** "

                            "```python "
                            """
                            import networkx as nx
                            import matplotlib.pyplot as plt

                            # Create a directed graph
                            G_clean = nx.DiGraph()

                            # Product details
                            product_details = {
                                "products/prod1": ("Wireless Mouse", 29.99),
                                "products/prod2": ("Ceramic Mug", 12.50)
                            }

                            # Updated product nodes with names and prices
                            product_nodes = {pid: f'{name}\n(${price})' for pid, (name, price) in product_details.items()}

                            # Define users and purchases
                            users = ["User1", "User2"]
                            purchases = [("User1", "products/prod1", 2),
                                        ("User2", "products/prod1", 1),
                                        ("User1", "products/prod2", 1),
                                        ("User2", "products/prod2", 3)]

                            # Add product nodes
                            for pid, label in product_nodes.items():
                                G_clean.add_node(label, color="lightgreen")

                            # Add user nodes and purchase edges
                            for user in users:
                                G_clean.add_node(user, color="lightblue")  # Users as blue nodes

                            for user, pid, qty in purchases:
                                G_clean.add_edge(user, product_nodes[pid], weight=qty)

                            # Update node colors
                            node_colors = [G_clean.nodes[n]["color"] for n in G_clean.nodes]

                            # Draw the graph
                            plt.figure(figsize=(7, 5))
                            pos = nx.spring_layout(G_clean, seed=42)  # Fixed seed for consistent layout

                            # Draw nodes and edges
                            nx.draw(G_clean, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=2500, font_size=9)
                            edge_labels = {(u, v): G_clean[u][v]['weight'] for u, v in G_clean.edges}
                            nx.draw_networkx_edge_labels(G_clean, pos, edge_labels=edge_labels, font_size=9)

                            plt.title("Network Graph: User Product Purchases")
                            plt.show()
                            """
                            "``` "
                        ),
                    }
                    
                ]
            )
    

    
    text_to_nx = chat_completion.choices[0].message.content
    pattern = r"python\s*(.*?)\s*```"
    match = re.search(pattern, text_to_nx, re.DOTALL)

    if match:
        text_to_nx_cleaned = match.group(1).strip()
        print(text_to_nx_cleaned)  # Output only the extracted code
    else:
        print("No match found")

    return text_to_nx_cleaned


# Define tools list
tools = [text_to_aql_to_text, text_to_aql_to_nxalgorithm]

# Query handler function
def query_graph(query):
    """Handles user queries and selects the appropriate tool."""
    if "visualize" in query.lower() or "show" in query.lower():
        tool = text_to_aql_to_nxalgorithm

    else:
        tool = text_to_aql_to_text

    result = tool.invoke(query)
    return result

# Define tools list
tools = [text_to_aql_to_text, text_to_aql_to_nxalgorithm]

# Query handler function
def query_graph(query):
    """Handles user queries and selects the appropriate tool."""
    if "visualize" in query.lower() or "show" in query.lower():
        tool = text_to_aql_to_nxalgorithm

    else:
        tool = text_to_aql_to_text

    result = tool.invoke(query)
    return result

# Example usage
if __name__ == "__main__":
    # Get user input
    user_query = input("Enter your query about your ArangoDB database: ")
    
    # Execute query
    print("\nExecuting query...")
    result = query_graph(user_query)
    print("\nResult:")
    print(result)
    try:
       exec(textwrap.dedent(result)) 
    except:
        print(result)
