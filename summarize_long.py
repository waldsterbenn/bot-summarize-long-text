from warnings import simplefilter
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from transformers import AutoTokenizer
import os
import numpy as np
from transformers import AutoTokenizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

batch_token_limit = 10000  # SHould be under LLM's context window size
max_tokens = 10000  # Maximum tokens for a single chunk
overlap = 10  # Tokens to overlap between chunks to ensure continuity
num_clusters = 10

is_podcast = False

show_scatter_plot = True
show_scatter_plot = False

embedding_func = SentenceTransformer("all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2")

# Function to chunk the text
directory = "textfiledata"
# List all PDF files in the directory
files = [os.path.join(directory, f)
         for f in os.listdir(directory) if f.endswith('.txt')]


def chunk_text(text, max_tokens, overlap):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk))
    return chunks


with open(files[0], 'r', encoding='utf-8') as file:
    lines = file.readlines()
    text = "".join(lines)  # Combine all lines into one string

clusters = chunk_text(text, max_tokens, overlap)

embeddings = embedding_func.encode(clusters)
ids = [f"{i}" for i in range(len(clusters))]
# collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)

kmeans = KMeans(n_clusters=num_clusters, random_state=33).fit(embeddings)

if show_scatter_plot:
    from sklearn.manifold import TSNE

    simplefilter(action='ignore', category=FutureWarning)

    # Step 1: Dimensionality reduction
    # Using PCA for simplicity and faster computation
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Alternatively, for a potentially better visualization use t-SNE (uncomment below)
    # tsne = TSNE(n_components=2, random_state=33)
    # embeddings_2d = tsne.fit_transform(embeddings)

    # Step 2: Plot embeddings and color code by cluster
    plt.figure(figsize=(12, 10))

    # Define a colormap that will be used for both points and centroids
    colors = plt.cm.tab20(np.linspace(0, 1, num_clusters))

    for i in range(num_clusters):
        # Cluster points
        points = embeddings_2d[kmeans.labels_ == i]
        plt.scatter(points[:, 0], points[:, 1], s=50,
                    color=colors[i], alpha=0.6, label=f"Cluster {i+1}")

    # Plot centroids
    # Make sure this matches your dimensionality reduction choice
    centroids_2d = pca.transform(kmeans.cluster_centers_)
    for i, centroid in enumerate(centroids_2d):
        plt.scatter(centroid[0], centroid[1], marker='x', s=200,
                    color=colors[i])  # Match centroid color with cluster
        plt.text(centroid[0], centroid[1], f"  Centroid {
            i+1}", color='black', fontsize=12, va='center', ha='left')

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Embeddings Cluster Visualization with Clearly Marked Centroids")
    plt.legend()
    plt.show()

# Map reduce and find closest bundled embeddings/clusters
closest_indices = []

for i in range(num_clusters):
    distances = np.linalg.norm(
        embeddings - kmeans.cluster_centers_[i], axis=1)
    closest_index = np.argmin(distances)
    closest_indices.append(closest_index)

selected_indecies = sorted(closest_indices)

selected_docs = [clusters[chunk] for chunk in selected_indecies]

# always add intro
if (0 not in selected_indecies):
    selected_docs.insert(0, clusters[0])


client = OpenAI(base_url="http://localhost:1235/v1", api_key="not-needed")

partial_summaries = []
tokens_in_selected_docs = len(tokenizer.tokenize("".join(selected_docs)))
print(f"Number of tokens for selection: {tokens_in_selected_docs}")
if tokens_in_selected_docs < batch_token_limit:
    partial_summaries = selected_docs
    print("No need for partial summaries, as text will fit in context window")
else:
    for transcription in selected_docs:
        if is_podcast:
            map_prompt = f"""<s>[DIST]
            You will be given passages of a transcribed text. That section will be enclosed in triple backticks (```).
            Your goal is to give a summary of this section so that the reader will have a full understanding of what was discussed.
            Your response should be a verbose summary of at least 5 paragraphs, and fully encompass what was said in the passage.
            Don't add explanatory sentences like "in this segment" or "the speaker says".
            Dont be lazy!
            ```
            {transcription}
            ```
            FULL SUMMARY:
            [/DIST]
            """
        else:
            map_prompt = f"""<s>[DIST]
            You will be given passages of a transcribed text. That section will be enclosed in triple backticks (```).
            Your goal is to give a summary of this section so that the reader will have a full understanding of what the text is about.
            Your response should be a verbose summary of at least 5 paragraphs, and fully encompass the meaning of the text.
            Dont be lazy!
            ```
            {transcription}
            ```
            FULL SUMMARY:
            [/DIST]
            """

        completion = client.completions.create(
            model="local-model",  # this field is currently unused
            prompt=map_prompt
        )
        partial_summaries.append(completion.choices[0].text)

merged_partials = "\n".join(partial_summaries)
print(f"Number of tokens for partials: {
    len(tokenizer.tokenize(merged_partials))}")

with open(f"merged-summary-chunks.txt", "w", encoding='utf-8') as file:
    file.write(merged_partials)

if is_podcast:
    final_prompt = f"""<s>[INST]
    You will be given a series summaries from a transcribed podcast.
    That summaries section will be enclosed in triple backticks (```).
    You're goal is to give a summary of what happened in the podcast.
    The reader should be able to grasp what happened, and what was talked about.
    Add names, titles, places where possible.
    Filter out things from sponsors or advertisement of products.
    The "Summary" part should be elaborate and verbose.
    The "Key Points" and "Word cloud" parts should be concise.
    Dont be lazy!
    Format the output as a Markdown (.md) file, following the template.
    Template section is enclosed in tripe underscore (___):
    ___
    # Title: <title of show>

    ## Participants
    - **<Person Name>**, Host
    - **<Person Name>**, <Title>

    ## Summary
    <long summary>
    > quoute
    <more long summary>

    ## Key Points
    1. <key point 1>.
    2. <topic 1>.

    ## Word cloud
    - <defining word>
    - <another defining word>
    ___

    ```
    {merged_partials}
    ```
    VERBOSE SUMMARY:
    [/INST]
    """
else:
    final_prompt = f"""
    <s>[INST]
    You will be given a series summaries from a transcribed text.
    That summaries section will be enclosed in triple backticks (```).
    You're goal is to give a summary of what happened in the text.
    The reader should be able to grasp what happened, and what it is about.
    The "Summary" part should be long, elaborate and verbose.
    Dont be lazy!
    Format the output as a Markdown (.md) file, following the template.
    Template section is enclosed in tripe underscore (___):
    ___
    # Title: <title>

    ## Names
    - **<Person Name>**
    - **<Person Name>**

    ## Verbose Summary
    <long summary>
    > quoute
    <more long summary>
    ___

    ```
    {merged_partials}
    ```
    VERBOSE SUMMARY:
    [/INST]
    """
print(f"Tokens for final prompt: {
    len(tokenizer.tokenize(final_prompt))}")

completion = client.completions.create(
    model="local-model",  # this field is currently unused
    prompt=final_prompt
)
summary = completion.choices[0].text

text_file = open(f"result-summary.md",
                 "w", encoding='utf-8')
text_file.writelines(summary)
text_file.close()
