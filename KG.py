import logging
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import movielens
from wikidata import (
    search_wikidata,
    find_wikidata_id,
    query_entity_links,
    read_linked_entities,
    query_entity_description,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

print(f"System version: {sys.version}")

# Parameters
MOVIELENS_DATA_SIZE = "100k"
MOVIELENS_SAMPLE = True
MOVIELENS_SAMPLE_SIZE = 50

# Load MovieLens data
logger.info("Loading MovieLens dataset...")
df = movielens.load_pandas_df(
    MOVIELENS_DATA_SIZE,
    header=("UserId", "ItemId", "Rating", "Timestamp"),
    title_col="Title",
    genres_col="Genres",
    year_col="Year",
)
logger.info(f"MovieLens dataset loaded with shape: {df.shape}")

# Sample movies
movies = df[["Title", "ItemId"]].drop_duplicates().reset_index()
if MOVIELENS_SAMPLE:
    movies = movies.head(MOVIELENS_SAMPLE_SIZE)
logger.info(f"Selected {len(movies)} movies for analysis.")

# Search for entities in Wikidata
names = [t + " film" for t in movies["Title"]]
logger.info("Searching Wikidata for movie entities...")
results = search_wikidata(names, extras=movies[["Title", "ItemId"]].to_dict())
logger.info("Wikidata search completed.")

# Save results to CSV
csv_file = "wikidata_results.csv"
results.to_csv(csv_file, index=False)
logger.info(f"Results saved to {csv_file}")

# Print the head of the results
print("Result head:")
print(results.head())

# Analyze and visualize the graph
logger.info("Creating graph from Wikidata results...")
G = nx.from_pandas_edgelist(results, "original_entity", "linked_entities")

# Prepare node labels
target_names = (
    results[["linked_entities", "name_linked_entities"]]
    .drop_duplicates()
    .rename(columns={"linked_entities": "labels", "name_linked_entities": "name"})
)
source_names = (
    results[["original_entity", "name"]]
    .drop_duplicates()
    .rename(columns={"original_entity": "labels"})
)
names = pd.concat([target_names, source_names])
names = names.set_index("labels").to_dict()["name"]

# Visualize part of the graph
logger.info("Visualizing a subgraph...")
subgraph_nodes = list(G.nodes)[:50]  # Visualize only the first 50 nodes
subgraph = G.subgraph(subgraph_nodes)

# Compute layout for the subgraph
pos = nx.spring_layout(subgraph)

# Filter the names dictionary to include only nodes in the subgraph
filtered_names = {node: names[node] for node in subgraph_nodes if node in names}

# Plot the subgraph
plt.figure(figsize=(12, 12))
nx.draw(subgraph, pos, node_size=60, font_size=9, width=0.2)
nx.draw_networkx_labels(subgraph, pos, filtered_names, font_size=9)
plt.title("Subgraph of Wikidata connections")
plt.show()

# Print statistics
logger.info(f"Number of unique movies: {len(results['Title'].unique())}")
logger.info(f"Graph contains {len(G.nodes)} nodes and {len(G.edges)} edges.")
