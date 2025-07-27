import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("myntra_products_catalog.csv")
df.columns = [c.strip() for c in df.columns]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Focus on ProductName + Description for semantic search
texts = (df['ProductName'].fillna('') + " " + df['Description'].fillna('')).tolist()
product_embeddings = model.encode(texts, show_progress_bar=False)

def semantic_recommendation(user_profile, search_query, budget):
    # Create a semantic query combining search text and user profile
    query = (
        f"{search_query}. Recommend fashion products for a {user_profile['age_group']} {user_profile['gender']} "
        f"who is a {user_profile['occupation']}, prefers brands like {', '.join(user_profile['brands'])}, "
        f"has {user_profile['body_shape']} body shape, {user_profile['body_size']} body size, and {user_profile['skin_tone']} skin tone."
    )

    query_embedding = model.encode([query])
    sims = cosine_similarity(query_embedding, product_embeddings)[0]

    df_copy = df.copy()
    df_copy["similarity"] = sims

    # Gender filter from Gender column
    if user_profile['gender'] in ["Men", "Women"]:
        df_copy = df_copy[df_copy["Gender"].str.strip().str.lower() == user_profile['gender'].lower()]

    # Budget filter
    filtered = df_copy[df_copy["Price (INR)"] <= budget]

    # Sort by similarity score
    filtered = filtered.sort_values(by="similarity", ascending=False).head(3)
    return filtered

def generate_style_tip(product, user_profile):
    short_desc = product['Description'][:120] + "..." if isinstance(product['Description'], str) else ""
    return (
        f"For a {user_profile['age_group']} {user_profile['gender']} with {user_profile['body_shape']} body shape, "
        f"this {product['ProductName']} by {product['ProductBrand']} is ideal. {short_desc}"
    )

st.title("AI Fashion Stylist – Personalized")

# Central search bar
search_query = st.text_input("🔍 Describe what you are looking for (e.g., 'I need an elegant outfit for a wedding')", "")

st.sidebar.header("Tell Us About Yourself")
age_group = st.sidebar.selectbox("Age Group", ["<18", "18-25", "26-35", "36-45", "45+"])
gender = st.sidebar.selectbox("Identify Yourself", ["Men", "Women", "Others"])
occupation = st.sidebar.selectbox("What Do You Do?", ["College Student", "Working Professional", "School Student", "Other"])

st.sidebar.header("Body & Skin")
body_shape = st.sidebar.selectbox("Body Shape", ["Apple", "Pear", "Hourglass", "Triangle", "Rectangle"])
body_size = st.sidebar.selectbox("Body Size", ["Petite", "Skinny", "Average", "Athletic", "Plus-size"])
skin_tone = st.sidebar.selectbox("Skin Tone", ["Very Fair", "Fair", "Light", "Medium", "Tan", "Deep", "Dark"])

st.sidebar.header("Preferred Brands")
brand_options = sorted(df['ProductBrand'].dropna().unique().tolist())
brands = st.sidebar.multiselect("Choose Brands", brand_options, default=[brand_options[0]] if brand_options else [])

budget = st.sidebar.slider("Budget (₹)", 500, 20000, 5000, 500)

if st.button("Get Recommendations"):
    user_profile = {
        "age_group": age_group,
        "gender": gender,
        "occupation": occupation,
        "body_shape": body_shape,
        "body_size": body_size,
        "skin_tone": skin_tone,
        "brands": brands
    }

    recs = semantic_recommendation(user_profile, search_query, budget)

    st.subheader("Top Personalized Recommendations for You")

    if recs.empty:
        st.warning("No relevant matches found! Showing top similar products within your budget.")

    for _, row in recs.iterrows():
        st.write(f"**{row['ProductName']}** | ₹{row['Price (INR)']} | {row['ProductBrand']}")
        st.write(generate_style_tip(row, user_profile))
        if 'image_url' in row and isinstance(row['image_url'], str):
            st.image(row['image_url'], width=200)
