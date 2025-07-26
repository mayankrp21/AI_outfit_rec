import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv("fashion_products_india1.csv")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute product embeddings
texts = (df['name'] + " " + df['style'] + " " + df['category']).tolist()
product_embeddings = model.encode(texts)

# Recommend products using similarity ranking
def recommend_products(user_query, budget, style):
    query_embedding = model.encode([user_query])
    sims = cosine_similarity(query_embedding, product_embeddings)[0]

    df_copy = df.copy()
    df_copy = df_copy.assign(similarity=sims)

    filtered = df_copy[df_copy["price"] <= budget]
    filtered = filtered[filtered["style"].str.lower().str.strip() == style.lower().strip()]

    if filtered.empty:
        filtered = df_copy[df_copy["price"] <= budget]

    filtered = filtered.sort_values(by="similarity", ascending=False).head(3)
    return filtered

# Generate personalized style tip
def generate_style_tip(product, profile):
    return (
        f"Hey {profile['name']}, since you are {profile['age']} years old from {profile['city']} and love {', '.join(profile['preferred_brands'])}, "
        f"this {product['name']} would be perfect for {profile['occasion_pref']}! Pair it with accessories to match your style."
    )

# Streamlit UI
st.title("AI Fashion Stylist – Personalized")

st.sidebar.header("User Profile")
name = st.sidebar.text_input("Your Name", "Aisha")
age = st.sidebar.slider("Your Age", 16, 60, 25)
city = st.sidebar.text_input("City", "Delhi")

brand_options = sorted(df['brand'].unique().tolist())
preferred_brands = st.sidebar.multiselect("Preferred Brands", brand_options, default=["Biba", "H&M"])

occasion_pref = st.sidebar.selectbox("Favourite Occasion Type", ["Wedding", "Festive", "Casual", "Office", "Party", "Brunch"])

st.sidebar.header("Search Preferences")
user_input = st.sidebar.text_input("What are you looking for?", "Wedding outfit")
style = st.sidebar.selectbox("Style", ["Ethnic", "Western", "Footwear", "Fusion", "Formal", "Streetwear"])
occasion = st.sidebar.selectbox("Occasion", ["Wedding", "Festive", "Casual", "Office", "Party", "Brunch"])
budget = st.sidebar.slider("Budget (₹)", 500, 20000, 5000, 500)

if st.sidebar.button("Get Recommendations"):
    profile = {
        "name": name,
        "age": age,
        "city": city,
        "preferred_brands": preferred_brands,
        "occasion_pref": occasion_pref
    }

    brands_text = ", ".join(preferred_brands) if preferred_brands else "any brands"
    full_query = f"{user_input} for a {age} year old in {city} who likes {brands_text} and wants a {style} {occasion} outfit under {budget}"

    recs = recommend_products(full_query, budget, style)

    st.subheader("Top Personalized Recommendations for You")

    if recs.empty:
        st.warning("No exact matches found! Showing top similar products under your budget.")

    for _, row in recs.iterrows():
        st.image(row["image_url"], width=200)
        st.write(f"**{row['name']}** | ₹{row['price']} | {row['brand']}")
        st.write(generate_style_tip(row, profile))
        st.markdown(f"[View Product]({row['image_url']})")
