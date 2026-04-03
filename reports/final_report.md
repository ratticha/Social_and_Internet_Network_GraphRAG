# Final Report Template

## 1. Introduction

Briefly explain GraphRAG, the motivation for context retrieval, and why link prediction is a useful simplification for this assignment.

## 2. Dataset

Describe the Wikipedia Vote Network:

- Number of nodes
- Number of edges
- Directed graph properties
- Why the graph is relevant for trust or contextual relationships

## 3. Data Processing

Document:

- Raw data format
- Cleaning steps
- Duplicate and self-loop handling
- Split strategy used for training and testing

## 4. Methods

Describe each implemented method:

- Common Neighbors
- Jaccard
- Adamic/Adar
- Personalized PageRank

## 5. Evaluation Setup

Document:

- Negative sampling strategy
- Top-K ranking setup
- Precision@K definition
- Parameter settings

## 6. Results

Add a table comparing the methods and discuss which performed best.

## 7. Discussion

Explain:

- Why some methods work better than others
- Limitations of the simplified GraphRAG setup
- How this connects back to context retrieval in LLM systems

## 8. Conclusion

Summarize the main findings and suggest possible next steps.
