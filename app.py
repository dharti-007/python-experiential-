import difflib
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Mobile View Configuration ---
st.set_page_config(
    page_title="Content & Plagiarism Scanner",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom CSS for Mobile Optimization
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3.2em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        font-size: 1rem;
    }
    .stTextArea textarea {
        font-size: 14px;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Mock Reference Database (For Plagiarism Checking) ---
REFERENCE_DATABASE = {
    "Document A (Tech History)": (
        "Artificial intelligence and machine learning have evolved rapidly over the "
        "past decade. Data science plays a crucial role in modern applications."
    ),
    "Document B (Environmental Science)": (
        "Climate change is one of the most pressing challenges of our time. "
        "Renewable energy sources like solar and wind power are essential for sustainability."
    ),
    "Document C (Education & Growth)": (
        "Education is the foundation of personal and societal growth. Lifelong learning "
        "empowers individuals to adapt to changing economic landscapes."
    ),
}

# --- Prohibited Keywords for Content Filtering ---
INAPPROPRIATE_WORDS = ["badword1", "offensive", "spamword", "hateword"]


def check_content_filter(text):
  """Scans text for prohibited/inappropriate words."""
  found_words = []
  lower_text = text.lower()
  for word in INAPPROPRIATE_WORDS:
    if re.search(r"\b" + re.escape(word) + r"\b", lower_text):
      found_words.append(word)
  return found_words


def check_plagiarism(input_text, database):
  """Calculates cosine similarity between input text and reference database."""
  docs = list(database.values())
  doc_names = list(database.keys())

  # Append user input text to corpus for vectorization
  corpus = docs + [input_text]
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(corpus)

  # Compute cosine similarity
  input_vector = tfidf_matrix[-1]
  db_vectors = tfidf_matrix[:-1]
  similarities = cosine_similarity(input_vector, db_vectors).flatten()

  results = []
  for name, sim in zip(doc_names, similarities):
    results.append({"doc": name, "similarity": float(sim * 100)})

  # Sort by highest similarity score
  results = sorted(results, key=lambda x: x["similarity"], reverse=True)
  return results


# --- Mobile UI Header ---
st.markdown("### 🛡️ Mobile Content & Plagiarism Shield")
st.markdown(
    "<p style='font-size: 0.85rem; color: gray; margin-top: -10px;'>Instant"
    " automated text screening & similarity analysis.</p>",
    unsafe_allow_html=True,
)

# Text Input Area optimized for mobile typing/pasting
user_text = st.text_area(
    "Paste your text below:",
    placeholder="Type or paste your paragraph here...",
    height=160,
)

# Action Button
if st.button("Run Security & Plagiarism Scan"):
  if not user_text.strip():
    st.warning("⚠️ Please provide some text to scan.")
  else:
    with st.spinner("Analyzing document..."):
      # Run checks
      flagged_words = check_content_filter(user_text)
      plagiarism_results = check_plagiarism(user_text, REFERENCE_DATABASE)
      top_match = plagiarism_results[0]
      max_sim = top_match["similarity"]

    # --- Results Dashboard ---
    st.markdown("---")
    st.markdown("#### 📋 Inspection Report")

    # 1. Content Filter Result
    if flagged_words:
      st.error(
          "❌ **Inappropriate Content Flagged!**\nDetected terms:"
          f" `{', '.join(flagged_words)}`"
      )
    else:
      st.success("✅ **Content Filter:** Clean (No inappropriate terms found)")

    # 2. Plagiarism Risk Status
    if max_sim > 70:
      st.error(
          f"🔴 **High Plagiarism Risk ({max_sim:.1f}%)**\nMatched with:"
          f" *{top_match['doc']}*"
      )
    elif max_sim > 30:
      st.warning(
          f"🟡 **Moderate Similarity ({max_sim:.1f}%)**\nMatched with:"
          f" *{top_match['doc']}*"
      )
    else:
      st.success(
          f"🟢 **Original Content**\nLowest similarity score detected"
          f" ({max_sim:.1f}%)."
      )

    # 3. Detailed Database Match Breakdown
    with st.expander("📊 Detailed Match Breakdown"):
      for res in plagiarism_results:
        st.write(
            f"- **{res['doc']}**: `Similarity: {res['similarity']:.1f}%`"
        )

    # 4. Text Diff Comparison View
    if max_sim > 15:
      with st.expander("📝 Side-by-Side Diff Preview"):
        ref_text = REFERENCE_DATABASE[top_match["doc"]]
        st.caption(f"Comparison with: {top_match['doc']}")
        diff = list(
            difflib.ndiff(user_text.splitlines(), ref_text.splitlines())
        )
        st.code("\n".join(diff[:12]), language="text")