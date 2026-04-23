# GraphRAG and Context Retrieval via Link Prediction on the Wikipedia Vote Network

**Course:** 42913 Social and Information Network Analysis  
**Topic:** Topic 2 — GraphRAG and Context Retrieval  

---

## 1. Introduction

Large Language Models (LLMs) such as ChatGPT and Gemini have demonstrated impressive capabilities across a wide range of tasks, but they face a fundamental limitation: they can only reason over the context provided to them at inference time. When a question requires knowledge that is private, recent, or highly specialised, LLMs frequently "hallucinate" — generating confident-sounding but incorrect answers. Retrieval-Augmented Generation (RAG) addresses this by retrieving relevant documents from an external knowledge base and injecting them into the model's context before it generates a response.

Standard RAG relies on dense vector similarity or keyword matching to find relevant documents. While effective for direct lookups, this approach fails when the relevant relationship is indirect. For example, if a user asks "How is the research of Professor A indirectly influenced by the theories of Scientist B?", a keyword search will fail unless A and B appear in the same document. The key insight behind GraphRAG is that structuring the knowledge base as a graph makes it possible to trace multi-hop relationships — finding the path from A to B through intermediate collaborators — and thereby retrieve context that flat retrieval would miss entirely.

This project builds a simplified retrieval module for a GraphRAG system by framing context retrieval as a **link prediction** task. On the Wikipedia Vote Network, if an algorithm predicts a strong link between User A and User B, it treats B as highly relevant context for A. We implement and compare four network analysis algorithms — Common Neighbors, Jaccard Coefficient, Adamic/Adar Index, and Personalized PageRank — to calculate relevance scores, predict missing connections, and evaluate each approach using Precision@K.

---

## 2. Dataset

We use the **Wikipedia Vote Network** from the Stanford Network Analysis Project (SNAP). The dataset captures the administrative election process of Wikipedia, where editors vote for or against other users being promoted to administrator status.

- **Nodes:** Wikipedia users (editors). Each unique user ID is one node.
- **Edges:** A directed edge from node *i* to node *j* means user *i* voted for user *j*, expressing trust or relevance.

### Raw dataset statistics

| Property | Value |
|---|---|
| Number of nodes | 7,115 |
| Number of directed edges (raw) | 103,689 |
| Graph density | ≈ 0.00205 |
| Average in-degree | ≈ 14.57 |
| Average out-degree | ≈ 14.57 |

The graph is **directed** because a vote flows from the voter to the candidate — the relationship is asymmetric. It is **sparse**: with ~7,000 nodes and ~100,000 edges, only about 0.2% of all possible directed edges exist. Despite this sparsity, the graph has a large weakly connected component, making link prediction a non-trivial and realistic task. In-degree captures the trust or authority of a user (how many others voted for them), while out-degree captures how active a user is in endorsing others.

The degree distribution is **heavy-tailed**: a small number of highly popular users receive a very large number of votes, while most users receive few. This power-law-like distribution is characteristic of real social networks and motivates graph-based approaches over flat keyword retrieval, since the most-connected nodes act as bridging hubs for indirect relationships.

---

## 3. Data Processing

### 3.1 Raw data format

The SNAP file `wiki-Vote.txt` contains comment lines prefixed with `#` followed by whitespace-separated pairs of integer user IDs, one edge per line. The dataset was downloaded directly from SNAP using `scripts/download_data.py`.

### 3.2 Cleaning steps

The raw file was parsed by `scripts/preprocess_data.py` using a custom reader that skips comment lines and parses each pair of IDs as integers. Two cleaning operations were applied:

1. **Self-loop removal:** edges where source equals target are dropped. These carry no information for link prediction since they represent a user voting for themselves (which does not occur meaningfully in this network).
2. **Duplicate removal:** repeated `(source, target)` pairs are collapsed to a single edge. A directed graph carries at most one edge per ordered pair.

After cleaning, the processed edge list contains **103,689 directed edges** stored in `data/processed/graph_edges.csv`.

### 3.3 Train/test split strategy

We adopted a **Random Split** strategy (80/20):

- All cleaned edges are shuffled using a fixed random seed (42) for reproducibility.
- The top 20% of shuffled edges are reserved as the **test set** (positive edges), and the remaining 80% form the **training set**.
- A connectivity constraint is applied during the split: an edge is eligible for the test set only if both its source and target nodes retain at least one edge in the training set. This prevents isolated nodes in the training graph, which would otherwise make scoring impossible for those nodes.

| Split | Edges |
|---|---|
| Training set | 82,951 |
| Test set | 20,738 |

### 3.4 Negative sampling

Standard link prediction evaluation requires not only true positive edges (the test edges) but also **negative edges** — pairs that do not exist — to simulate the realistic scenario where an algorithm must distinguish relevant from irrelevant candidates.

For each source node that appears in the test set, we sample **10 negative edges per positive test edge** from the space of non-existent pairs. Sampling is stratified by source node to ensure every node has enough candidates for a fair Top-K evaluation. All sampled negative edges are guaranteed not to exist in either the training or test set.

| Candidate type | Edges |
|---|---|
| Positive (test edges) | 20,738 |
| Negative (sampled non-edges) | 207,380 |
| **Total candidates** | **228,118** |
| Source nodes evaluated | 2,807 |

---

## 4. Methods

All four methods are applied to the **candidate set** — the union of positive test edges and sampled negative edges — rather than all ~50 million possible directed pairs. This follows the standard efficiency recommendation in the assignment specification.

The three local similarity methods (Common Neighbors, Jaccard, Adamic/Adar) operate on an **undirected projection** of the training graph. This choice reflects the bidirectional nature of social trust: if A voted for B, both A and B share a social context. Personalized PageRank operates on the **directed training graph** to respect the direction of trust flows.

### 4.1 Common Neighbors

For a pair (u, v), the score is the number of nodes that are neighbours of both u and v in the undirected training graph:

$$\text{CN}(u, v) = |N(u) \cap N(v)|$$

**Assumption:** Two users are more likely to be connected if they share many mutual connections. This is analogous to finding indirect relationships in a GraphRAG context — if A and B are both connected to a hub C, C provides the contextual bridge between them.

### 4.2 Jaccard Coefficient

Jaccard normalises Common Neighbors by the size of the union of the two neighbourhoods:

$$\text{Jaccard}(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$

**Assumption:** A high overlap proportion indicates structural similarity, independent of how large the individual neighbourhoods are. This penalises highly-connected hub nodes that have large neighbourhoods by design.

### 4.3 Adamic/Adar Index

Adamic/Adar sums the inverse log-degree of each shared neighbour:

$$\text{AA}(u, v) = \sum_{w \in N(u) \cap N(v)} \frac{1}{\log |N(w)|}$$

**Assumption:** Shared neighbours that are themselves low-degree (niche) nodes carry more meaningful information than high-degree hub nodes, which are connected to nearly everyone. This is a refined version of Common Neighbors that discounts generic bridging nodes.

### 4.4 Personalized PageRank

Personalized PageRank (PPR) computes a stationary distribution over the directed training graph starting from a single source node. For a source node u, we set the teleportation vector so that all probability mass restarts at u, and run PageRank with damping factor α = 0.85:

$$\mathbf{r}_u = \alpha \mathbf{A}^T \mathbf{r}_u + (1 - \alpha) \mathbf{e}_u$$

The score for a candidate target v is the probability mass assigned to v in the resulting distribution $\mathbf{r}_u$.

**Assumption:** Nodes that are reachable from u via short, high-weight directed paths will accumulate more probability mass and are therefore more relevant to u. Unlike the local similarity methods, PPR considers **global graph structure** and multi-hop paths, making it particularly suitable for the indirect relationship discovery that motivates GraphRAG.

---

## 5. Evaluation Setup

### 5.1 Precision@K

For each source node, the algorithm ranks all candidate (source, target) pairs by score in descending order and recommends the **Top-K = 10** most likely links. Precision@K measures the fraction of the top-K predictions that are true positive test edges:

$$\text{Precision@K}(u) = \frac{|\text{Top-K predictions for } u \cap \text{true test edges for } u|}{K}$$

The reported metric is the **mean Precision@10** across all evaluated source nodes.

### 5.2 Why Precision@K fits the GraphRAG use case

In a GraphRAG context retrieval scenario, the algorithm selects the K most relevant nodes as context for a query. What matters is how many of those K selected nodes are genuinely relevant (i.e., would appear in the ground-truth context). Precision@K measures exactly this: the quality of the top-K retrieved context items. A higher Precision@10 means the GraphRAG engine retrieves better-quality context.

### 5.3 Parameter settings

| Parameter | Value |
|---|---|
| Test fraction | 20% |
| Negatives per positive | 10 |
| Top-K (K) | 10 |
| PageRank damping factor (α) | 0.85 |
| Random seed | 42 |

---

## 6. Results

### 6.1 Precision@10 comparison

| Method | Mean Precision@10 | Std Dev | Total Hits | Evaluated Nodes |
|---|---:|---:|---:|---:|
| **Personalized PageRank** | **0.3369** | 0.2389 | 9,456 | 2,807 |
| Common Neighbors | 0.3029 | 0.2328 | 8,502 | 2,807 |
| Adamic/Adar | 0.3027 | 0.2320 | 8,496 | 2,807 |
| Jaccard | 0.2941 | 0.2320 | 8,254 | 2,807 |

**Personalized PageRank** achieves the highest mean Precision@10 of **0.3369**, meaning that on average, about 3.4 out of 10 predicted links are genuine test edges. This substantially outperforms random guessing: with 20,738 positive edges out of 228,118 total candidates, a random ranker would achieve a baseline Precision@10 of approximately 0.091. All four methods beat this baseline by a wide margin.

### 6.2 Relative performance

- **PPR vs. Common Neighbors:** PPR improves by +3.4 percentage points (relative improvement of ~11%). The gap reflects PPR's ability to follow multi-hop directed paths rather than only looking at immediate shared neighbours.
- **Common Neighbors vs. Adamic/Adar:** The two methods are nearly identical (0.3029 vs. 0.3027). This suggests that in this graph, the inverse-log weighting of Adamic/Adar provides minimal additional signal over the raw neighbour count — likely because the degree distribution is heavy-tailed enough that most shared neighbours are already low-to-medium degree nodes.
- **Jaccard:** The lowest scorer (0.2941). Normalising by the union penalises high-degree nodes, but in a social trust network like this one, high-degree nodes are genuine connectors and their shared neighbours are meaningful signals. Removing this degree signal hurts performance.

### 6.3 Stability across source nodes

All methods show substantial variance (standard deviation ≈ 0.23), indicating that performance varies widely across source nodes. Some nodes benefit strongly from graph-based retrieval (precision close to 1.0) while others score near zero. This heterogeneity is expected: nodes with sparse neighbourhoods or that are weakly connected in the training graph give the local similarity methods little signal to work with.

PPR has a slightly higher standard deviation (0.2389 vs. 0.2320 for the local methods) because it leverages global paths — nodes well-connected to the rest of the network benefit more from PPR than poorly-connected nodes.

---

## 7. Discussion

### 7.1 Why PPR outperforms local similarity methods

The Wikipedia Vote Network has a hub-and-spoke structure with a small number of highly-trusted administrators who received votes from a large fraction of the community. Personalised PageRank naturally exploits this structure: if user A voted for hub H, and hub H also voted for many other administrators, then those administrators receive indirect probability mass from A's personalised walk. This multi-hop inference is exactly the kind of indirect reasoning that motivates GraphRAG — connecting A to B through intermediate nodes that would never be found by keyword search.

The three local similarity methods (CN, Jaccard, AA) are limited to one-hop shared neighbourhoods. They cannot find relevant connections between nodes that are two or more hops apart in the training graph. This is a fundamental limitation for a GraphRAG retrieval use case where the most valuable context often lies at a distance.

### 7.2 Connection to GraphRAG context retrieval

The central claim of GraphRAG is that structuring knowledge as a graph enables richer context retrieval than flat keyword search. This experiment provides a concrete, measurable demonstration of that claim:

- A **pure keyword baseline** would retrieve context only when source and target share explicit co-occurrence signals. On a graph where most nodes are 2–3 hops apart, this fails frequently.
- **Graph-based methods** like PPR retrieve indirect connections by following the structure of trust and collaboration, achieving Precision@10 ≈ 0.34 — approximately 3.7× better than random selection.

In an LLM pipeline, this means that a GraphRAG retrieval module using PPR would, on average, provide the model with 3–4 genuinely relevant context nodes for every 10-node context window, compared to only 1 node for a random retriever. This is the concrete efficiency gain that motivates investing in graph-structured retrieval over simpler keyword-based approaches.

### 7.3 Limitations

1. **Simplified evaluation environment:** The test set consists of held-out edges from the same graph, not truly unseen queries. In a production GraphRAG system, queries come from users in natural language and the relevant context may span multiple subgraphs.

2. **Undirected projection for local methods:** Common Neighbors, Jaccard, and Adamic/Adar discard edge direction. Using the directed graph (e.g., computing scores separately for in-neighbours and out-neighbours) might recover some signal.

3. **Static graph:** Wikipedia's trust network evolves over time. A time-based split (where training contains earlier edges and testing contains later edges) would better simulate a production deployment scenario, though it introduces additional complexity around node coverage.

4. **Candidate sampling efficiency:** We sample 10 negatives per positive per source node. With only 2,807 source nodes evaluated out of 7,115 total nodes (39%), a significant portion of the graph's source nodes are excluded. Nodes with no positive test edges are entirely absent from evaluation.

5. **No LLM integration:** This project demonstrates the retrieval module in isolation. Connecting the ranked candidates to an actual LLM and measuring downstream answer quality is the natural next step.

### 7.4 Future improvements

- **Combined scoring:** Ensemble the four scores (e.g., by normalising and summing) to produce a single rank. Hybrid methods often outperform any individual baseline.
- **Node features:** Incorporate additional node attributes (e.g., edit history, account age) as features in a GNN-based link predictor (e.g., GraphSAGE, GAT) for further improvement.
- **Time-aware split:** Implement a temporal split where edges before a cutoff date form training and edges after form the test set, more faithfully simulating a real GraphRAG deployment.
- **Recall@K and AUC-ROC:** Supplement Precision@K with recall-oriented metrics to capture whether the algorithm retrieves all relevant context, not just whether the top-K items are precise.

---

## 8. Conclusion

This project implemented and evaluated four link prediction algorithms as a retrieval module for a GraphRAG system, using the Wikipedia Vote Network as a proxy for a trust and relevance graph.

**Key findings:**

- All four methods substantially outperform random retrieval, confirming that graph structure carries meaningful information for context relevance.
- **Personalized PageRank** (Precision@10 = 0.337) is the strongest method, outperforming local similarity methods by ~11% in relative terms. Its ability to follow multi-hop directed paths closely mirrors the indirect reasoning that GraphRAG is designed to enable.
- Local similarity methods (Common Neighbors: 0.303, Adamic/Adar: 0.303, Jaccard: 0.294) perform comparably to each other, with the AA weighting providing negligible improvement over raw Common Neighbors in this particular graph.
- The wide standard deviation (≈0.23) across source nodes highlights that graph-based retrieval quality is uneven — a property that would motivate adaptive, node-specific retrieval strategies in a production system.

The experiment demonstrates concretely why GraphRAG is a promising approach: by structuring knowledge as a directed graph and ranking candidate nodes via personalized walks, a retrieval module can surface indirect relationships that flat keyword search would miss entirely — delivering up to 3.7× more relevant context to an LLM than a random baseline.

---

## References

1. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). *From Local to Global: A Graph RAG Approach to Query-Focused Summarization.* arXiv preprint arXiv:2404.16130.
2. Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., & Wu, X. (2024). *Unifying Large Language Models and Knowledge Graphs: A Roadmap.* IEEE Transactions on Knowledge and Data Engineering.
3. Leskovec, J., & Krevl, A. (2014). *SNAP Datasets: Stanford Large Network Dataset Collection.* [http://snap.stanford.edu/data](http://snap.stanford.edu/data)
4. Adamic, L. A., & Adar, E. (2003). *Friends and neighbors on the Web.* Social Networks, 25(3), 211–230.
5. Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). *The PageRank Citation Ranking: Bringing Order to the Web.* Technical Report, Stanford InfoLab.
