
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="AI Patent Prior Art Search",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 AI-Based Patent Prior Art Search")

st.write(
    "Enter a description of your invention below. "
    "The AI will find the 5 most semantically similar "
    "patents in the database."
)

@st.cache_resource
def load_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

model = load_model()

@st.cache_data
def load_patents():

    return pd.read_pickle("patent_database.pkl")


@st.cache_resource
def load_faiss():

    return faiss.read_index("patent_faiss.index")


df = load_patents()
index = load_faiss()

title_col = None
abstract_col = None
patent_id_col = None

for col in df.columns:

    name = str(col).lower()

    if title_col is None and "title" in name:
        title_col = col

    if abstract_col is None and "abstract" in name:
        abstract_col = col

    if patent_id_col is None:

        if (
            "publication_number" in name
            or "patent_number" in name
            or name == "id"
        ):
            patent_id_col = col

invention = st.text_area(
    "💡 Describe your invention:",
    height=200,
    placeholder=(
        "Example: A smart water bottle that "
        "monitors water consumption and sends "
        "hydration reminders to a mobile phone."
    )
)

if st.button("🔎 Find Similar Patents"):

    if invention.strip() == "":

        st.warning(
            "Please enter an invention description."
        )

    else:

        query_embedding = model.encode(
            [invention],
            convert_to_numpy=True
        )

        query_embedding = query_embedding / np.linalg.norm(
            query_embedding,
            axis=1,
            keepdims=True
        )

        scores, indices = index.search(
            query_embedding,
            min(5, index.ntotal)
        )

        st.subheader("📄 Top Similar Patents")

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1
        ):

            row = df.iloc[int(idx)]

            if title_col is not None:
                st.markdown(
                    f"### {rank}. {row[title_col]}"
                )
            else:
                st.markdown(
                    f"### {rank}. Patent"
                )

            if patent_id_col is not None:
                st.write(
                    "**Patent Number:**",
                    row[patent_id_col]
                )

            st.write(
                f"**Semantic Similarity:** "
                f"{score * 100:.2f}%"
            )

            if abstract_col is not None:
                st.write(
                    "**Abstract:**",
                    row[abstract_col]
                )

            st.divider()
