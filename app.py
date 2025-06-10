import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import time

# Existing SentenceTransformer model and Qdrant settings
QDRANT_URL = "https://ba0f9774-1b9e-4b0b-bb05-db8fadfe122c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sMVFQwd_dg3z89uIih5r5olFlbXLAjl_Gcx0V5IJG-U"
COLLECTION_NAME = "arxiv_papers_titles"

# SPECTER2 Qdrant collection
SPECTER_COLLECTION = "arxiv_specter2_recommendations"
SPECTER_QDRANT_URL = "https://d09a5111-2452-49a5-b3f8-6a488ca728da.us-east-1-0.aws.cloud.qdrant.io"
SPECTER_QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4xolrSNFliLnWhb7i1Tw1CMbs2pPWJjKu-RgOlQGZTI"

# Initialize SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/find_similar', methods=['POST'])
def find_similar():
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = 10

    if not query_text.strip():
        return jsonify({"error": "Missing or empty query_text"}), 400

    query_embedding = model.encode([query_text], normalize_embeddings=True)[0].astype(np.float16)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(exact=False)
    )

    response = [
        {
            "arxiv_id": r.payload.get("arxiv_id", "N/A"),
            "score": r.score
        }
        for r in results
    ]
    return jsonify(response)


# ------------------ SPECTER2 Integration ------------------ #

def setup_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

    if torch.cuda.is_available():
        model = model.to('cuda')
        model.half()
    return model, tokenizer


specter_model, specter_tokenizer = setup_specter_model()
specter_client = QdrantClient(
    url=SPECTER_QDRANT_URL, 
    api_key=SPECTER_QDRANT_API_KEY, 
    timeout=120
)


@app.route('/specter_search', methods=['POST'])
def specter_search():
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = 10

    if not query_text.strip():
        return jsonify({"error": "Missing or empty query_text"}), 400

    inputs = specter_tokenizer(
        query_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        return_token_type_ids=False
    )

    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = specter_model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].astype(np.float16)

    results = specter_client.search(
        collection_name=SPECTER_COLLECTION,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )

    response = [
        {
            "arxiv_id": r.payload.get("arxiv_id", "N/A"),
            "score": r.score
        }
        for r in results
    ]
    return jsonify(response)

# ------------------ End SPECTER2 Integration ------------------ #

if __name__ == '__main__':
    app.run(debug=True)
