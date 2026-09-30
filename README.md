
# AI-Based Patent Prior Art Search Using Semantic Similarity

## Project Overview

This project is an AI-based patent prior art search system that uses Natural Language Processing and semantic similarity to identify patents related to a new invention description.

The system converts patent text and the user's invention description into numerical embeddings using a Sentence Transformer model. FAISS is then used to efficiently search the patent embeddings and return the top 5 semantically similar patents.

## How It Works

User enters an invention description
        ↓
Sentence Transformer
        ↓
Text embedding
        ↓
FAISS similarity search
        ↓
Top 5 similar patents
        ↓
Streamlit interface

## Technologies Used

- Python
- Pandas
- NumPy
- Sentence Transformers
- all-MiniLM-L6-v2
- FAISS
- Streamlit
- PyArrow
- Google Colab

## Dataset

The project uses genuine patent records obtained from the USPTO Explainable AI dataset on Kaggle.

The working dataset contains patent publication numbers, titles, abstracts, claims, and descriptions.

Due to dataset size and computational limitations, a subset of the available patent records is used for the demonstration.

## AI Method

The Sentence Transformer model `all-MiniLM-L6-v2` converts patent text into 384-dimensional embeddings.

The embeddings are normalized and searched using FAISS Inner Product similarity, which corresponds to cosine similarity for normalized vectors.

## Output

For a given invention description, the system returns:

- Top 5 similar patents
- Patent publication number
- Patent title
- Semantic similarity score
- Patent abstract

## Important Limitation

The similarity score represents semantic similarity between text documents. It is not a legal determination of patent infringement, novelty, or patentability.

The system is intended as a prior-art retrieval prototype and not as a replacement for professional patent examination or legal analysis.

## Future Improvements

- Larger patent database
- Patent-specific language models
- IPC/CPC classification filtering
- Hybrid keyword and semantic search
- Claims-focused similarity
- Date and classification filters
- Explainable similarity results

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
