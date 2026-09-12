import os
import json

import joblib
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI E-Commerce Search",
    page_icon="🛍️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .result-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #e5e5e5;
        margin-bottom: 15px;
    }

    .product-title {
        font-size: 21px;
        font-weight: 650;
    }

    .product-id {
        color: #777;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD ML ARTIFACTS
# =========================================================

@st.cache_resource

def load_models():
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    ranking_model = joblib.load("models/ranking_model.pkl")
    products_catalog = joblib.load("models/products_catalog.pkl")
    product_embeddings = np.load("models/product_embeddings.npy")
    model_info = joblib.load("models/model_info.pkl")

    return (
        vectorizer,
        ranking_model,
        products_catalog,
        product_embeddings,
        model_info,
    )


try:
    with st.spinner("Loading AI search models..."):
        (
            vectorizer,
            ranking_model,
            products_catalog,
            product_embeddings,
            model_info,
        ) = load_models()
except Exception as e:
    st.error(f"Could not load the ML models: {e}")
    st.stop()


# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    embedding_model = load_embedding_model()
except Exception as e:
    st.error(f"Could not load the embedding model: {e}")
    st.stop()


# =========================================================
# GROQ CLIENT
# =========================================================

@st.cache_resource

def load_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


groq_client = load_groq_client()


# =========================================================
# GROQ — QUERY UNDERSTANDING
# =========================================================

def understand_query(user_query):
    """Use Groq to convert a natural-language query into structured intent."""

    if groq_client is None:
        return {
            "category": "Unknown",
            "use_case": "Unknown",
            "features": [],
            "max_price": None,
            "brand": None,
        }

    prompt = f"""
Analyze this e-commerce shopping query:

{user_query}

Return ONLY a valid JSON object with exactly these keys:

"category": string
"use_case": string
"features": array of strings
"max_price": number or null
"brand": string or null

Rules:
- category must be a string.
- use_case must be a string.
- features MUST be a JSON array of strings.
- max_price must be a number or null.
- brand must be a string or null.
- Do not invent information.
- Use an empty array when no features are mentioned.
- Use null when price or brand is not mentioned.
"""

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an e-commerce query understanding system. Return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=300,
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise ValueError("Groq returned an empty query-understanding response.")

        result = json.loads(content)

        # Normalize the structure so the UI never crashes.
        category = result.get("category") or "General"
        use_case = result.get("use_case") or "General"
        features = result.get("features") or []
        max_price = result.get("max_price")
        brand = result.get("brand")

        if not isinstance(features, list):
            features = [str(features)]

        return {
            "category": str(category),
            "use_case": str(use_case),
            "features": [str(x) for x in features],
            "max_price": max_price,
            "brand": brand if brand else None,
        }

    except Exception as e:
        st.warning(f"Groq query understanding unavailable: {e}")

        return {
            "category": "General",
            "use_case": "General",
            "features": [],
            "max_price": None,
            "brand": None,
        }


# =========================================================
# BUILD ENHANCED QUERY
# =========================================================

def build_enhanced_query(query, query_info):
    """Combine the original query with useful LLM-extracted intent."""

    parts = [
        query,
        query_info.get("category", ""),
        query_info.get("use_case", ""),
        " ".join(query_info.get("features", [])),
        query_info.get("brand") or "",
    ]

    return " ".join(
        str(part).strip()
        for part in parts
        if str(part).strip()
        and str(part).strip().lower() not in {"general", "unknown"}
    )


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(query, top_k=50):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=False,
    )

    similarities = cosine_similarity(
        query_embedding,
        product_embeddings,
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = products_catalog.iloc[top_indices].copy()
    results["semantic_similarity"] = similarities[top_indices]

    return results


# =========================================================
# FINAL ML SEARCH
# =========================================================

def search_products(query, final_k=10):
    candidates = semantic_search(query, top_k=50)

    query_vector = vectorizer.transform([query])

    candidate_texts = (
        candidates["product_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    candidate_vectors = vectorizer.transform(candidate_texts)

    tfidf_scores = cosine_similarity(
        query_vector,
        candidate_vectors,
    )[0]

    candidates["tfidf_similarity"] = tfidf_scores
    candidates["query_length"] = len(query.split())
    candidates["product_text_length"] = (
        candidates["product_text"]
        .fillna("")
        .astype(str)
        .apply(len)
    )

    feature_columns = [
        "tfidf_similarity",
        "semantic_similarity",
        "query_length",
        "product_text_length",
    ]

    candidates["predicted_relevance"] = ranking_model.predict(
        candidates[feature_columns]
    )

    candidates = candidates.sort_values(
        "predicted_relevance",
        ascending=False,
    )

    return candidates.head(final_k).reset_index(drop=True)


# =========================================================
# RECOMMENDATION ENGINE
# =========================================================

def recommend_products(selected_index, search_query, top_k=3):
    """Recommend products using the selected product + current query."""

    selected_embedding = product_embeddings[selected_index].reshape(1, -1)

    product_similarities = cosine_similarity(
        selected_embedding,
        product_embeddings,
    )[0]

    # Also compare the current query with every catalog product.
    query_embedding = embedding_model.encode(
        [search_query],
        normalize_embeddings=False,
    )

    query_similarities = cosine_similarity(
        query_embedding,
        product_embeddings,
    )[0]

    # Blend item similarity with query relevance.
    combined_scores = (
        0.6 * product_similarities
        + 0.4 * query_similarities
    )

    ranked_indices = np.argsort(combined_scores)[::-1]

    recommendations = []

    for idx in ranked_indices:
        if idx == selected_index:
            continue

        recommendations.append(idx)

        if len(recommendations) == top_k:
            break

    results = products_catalog.iloc[recommendations].copy()
    results["recommendation_score"] = combined_scores[recommendations]

    return results


# =========================================================
# GROQ — TOP PRODUCT EXPLANATION
# =========================================================

def generate_recommendation_explanation(query, product_text):
    """Generate one short explanation for the best result."""

    if groq_client is None:
        return "Groq is not connected."

    prompt = f"""
User query:
{query}

Product information:
{product_text}

Explain in exactly 2 short sentences why this product is relevant to the query.
Use ONLY facts present in the product information.
Do not invent price, rating, brand, features, or specifications.
Return plain text only.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise e-commerce recommendation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        content = response.choices[0].message.content

        if content and content.strip():
            return content.strip()

        return "Groq returned an empty explanation."

    except Exception as e:
        return f"AI explanation unavailable: {e}"


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛍️ AI-Powered E-Commerce Search</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">LLM-powered query understanding, semantic search, '
    'machine-learning ranking, and intelligent recommendations.</div>',
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("⚙️ System Information")

    st.write(f"Products: **{len(products_catalog):,}**")
    st.write("LLM: **Groq**")
    st.write("Query Model: **GPT-OSS-20B**")
    st.write("Explanation Model: **Llama 3.3 70B**")
    st.write("Embedding Model: **all-MiniLM-L6-v2**")
    st.write("Ranking Model: **Random Forest**")
    st.write("Retrieval: **Semantic + TF-IDF**")

    st.divider()

    if groq_client is not None:
        st.success("Groq LLM: Connected ✅")
    else:
        st.error("Groq LLM: Not Connected ❌")
        st.caption(
            "Set GROQ_API_KEY in the terminal before starting Streamlit."
        )

    st.divider()

    st.caption("AI E-Commerce Search & Recommendation System")


# =========================================================
# SEARCH BAR
# =========================================================

query = st.text_input(
    "🔍 What are you looking for?",
    placeholder="comfortable running shoes under ₹3000",
)

search_button = st.button(
    "🔎 Search Products",
    type="primary",
)


# =========================================================
# SEARCH PIPELINE
# =========================================================

if search_button and query.strip():

    # -----------------------------------------------------
    # 1. LLM QUERY UNDERSTANDING
    # -----------------------------------------------------

    with st.spinner("Understanding your query with AI..."):
        query_info = understand_query(query)

    enhanced_query = build_enhanced_query(
        query,
        query_info,
    )

    # -----------------------------------------------------
    # 2. SEARCH + ML RANKING
    # -----------------------------------------------------

    with st.spinner("Searching and ranking products..."):
        results = search_products(
            enhanced_query,
            final_k=10,
        )

    # -----------------------------------------------------
    # 3. QUERY UNDERSTANDING DISPLAY
    # -----------------------------------------------------

    st.subheader("🤖 AI Query Understanding")

    st.caption(
        f"🔎 Enhanced search query: **{enhanced_query}**"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Category",
            query_info.get("category", "General"),
        )

    with col2:
        st.metric(
            "Use Case",
            query_info.get("use_case", "General"),
        )

    with col3:
        features = query_info.get("features", [])
        st.metric(
            "Features",
            ", ".join(features) if features else "None",
        )

    with col4:
        max_price = query_info.get("max_price")
        st.metric(
            "Max Price",
            f"₹{max_price:,.0f}" if isinstance(max_price, (int, float)) else "No limit",
        )

    brand = query_info.get("brand")
    if brand:
        st.caption(f"🏷️ Brand detected by AI: **{brand}**")

    # -----------------------------------------------------
    # 4. SEARCH RESULTS
    # -----------------------------------------------------

    st.subheader("🛍️ Top Search Results")

    st.caption(
        f"Found {len(results)} top-ranked products from the semantic retrieval + ML ranking pipeline."
    )

    # -----------------------------------------------------
    # 5. PRODUCT CARDS
    # -----------------------------------------------------

    for i, product in results.iterrows():

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([7, 2])

        with col1:
            st.markdown(
                f'<div class="product-title">#{i + 1} Product {product["product_id"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="product-id">Product ID: {product["product_id"]}</div>',
                unsafe_allow_html=True,
            )

            st.write(product["product_text"])

        with col2:
            st.metric(
                "ML Relevance",
                f'{product["predicted_relevance"]:.3f}',
            )

            st.metric(
                "Semantic",
                f'{product["semantic_similarity"]:.3f}',
            )

            st.metric(
                "TF-IDF",
                f'{product["tfidf_similarity"]:.3f}',
            )

        # Only one LLM explanation per search to reduce API usage.
        if i == 0 and groq_client is not None:
            with st.spinner("Generating AI explanation..."):
                explanation = generate_recommendation_explanation(
                    query,
                    product["product_text"],
                )

            st.info(
                f"🤖 **AI Explanation:** {explanation}"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # 6. QUERY-AWARE RECOMMENDATIONS
    # -----------------------------------------------------

    st.subheader("✨ You May Also Like")

    best_product_id = results.iloc[0]["product_id"]

    matching_indices = products_catalog.index[
        products_catalog["product_id"] == best_product_id
    ]

    if len(matching_indices) > 0:
        best_index = matching_indices[0]

        recommendations = recommend_products(
            best_index,
            enhanced_query,
            top_k=3,
        )

        rec_columns = st.columns(3)

        for i, (_, product) in enumerate(
            recommendations.iterrows()
        ):
            with rec_columns[i]:
                st.markdown(
                    f"### 🛍️ Product {product['product_id']}"
                )

                st.write(product["product_text"])

                st.caption(
                    f"Recommendation score: {product['recommendation_score']:.3f}"
                )

    else:
        st.info("No additional recommendations available.")


# =========================================================
# INITIAL STATE
# =========================================================

else:
    st.info(
        "👆 Enter a natural-language shopping query to start searching."
    )

    st.markdown(
        """
        ### Try these searches

        - `comfortable running shoes`
        - `comfortable running shoes under 3000`
        - `black wireless headphones under 5000`
        - `lightweight shoes for running`
        - `durable waterproof shoes`
        - `premium wireless headphones`
        - `gaming laptop for students`
        - `waterproof shoes for trekking`

        ### 🧠 How the system works

        **1. LLM Query Understanding**  
        Groq extracts category, use case, features, price, and brand.

        **2. LLM-Enhanced Retrieval**  
        The extracted intent is combined with the original query.

        **3. Semantic Retrieval**  
        Sentence Transformers retrieve semantically similar products.

        **4. Lexical Retrieval**  
        TF-IDF captures important exact terms.

        **5. Machine-Learning Ranking**  
        Random Forest predicts product relevance from the search features.

        **6. Intelligent Recommendations**  
        Recommendations combine product similarity with query relevance.

        **7. Generative AI Explanation**  
        Groq generates a short explanation for the best-ranked product.
        """
    )
