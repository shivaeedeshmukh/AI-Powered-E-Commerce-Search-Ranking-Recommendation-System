# 🛍️ AI-Powered E-Commerce Search, Ranking & Recommendation System

An end-to-end AI/ML system for intelligent e-commerce product discovery that combines **LLM-based query understanding, semantic search, TF-IDF retrieval, machine-learning ranking, content-based recommendations, and LLM-generated explanations**.

The goal is to improve product discovery by understanding what a user means rather than relying only on exact keyword matching.

---

## 🚀 Project Overview

Traditional e-commerce search systems often depend heavily on lexical keyword matching. This can fail when users describe their needs using natural language or vocabulary that differs from the product description.

For example:

> **"comfortable running shoes under ₹3000 for daily jogging"**

Instead of treating this as a simple keyword search, this system:

1. Understands the user's intent using an LLM.
2. Converts the intent into a structured representation.
3. Retrieves candidate products using semantic search.
4. Uses TF-IDF to capture lexical relevance.
5. Combines multiple signals using a machine-learning ranking model.
6. Returns the top-ranked products.
7. Recommends semantically similar products.
8. Generates a natural-language explanation for the top recommendation.

---

# 🎯 Problem Statement

The objective is to build an intelligent e-commerce search and recommendation system that can:

- Understand natural-language shopping queries.
- Retrieve products using semantic similarity.
- Preserve exact keyword relevance using lexical retrieval.
- Rank retrieved candidates using machine learning.
- Recommend similar products using product embeddings.
- Explain why a product is relevant using generative AI.

---

# 🧠 System Architecture

```text
                         USER
                           │
                           ▼
                  Natural Language Query
                           │
                           ▼
                  ┌─────────────────┐
                  │    Groq LLM     │
                  │ Query Understanding
                  └────────┬────────┘
                           │
                           ▼
                  Structured Search Intent
                           │
                           ▼
                  LLM-Enhanced Query
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
       Semantic Retrieval          TF-IDF Retrieval
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                   Candidate Products
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                  Random Forest Ranker
                           │
                           ▼
                      Top 10 Results
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
       Recommendation Engine    Groq Explanation
                 │                   │
                 ▼                   ▼
        Similar Products        Why this product?
