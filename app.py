import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

QDRANT_URL = "https://ba0f9774-1b9e-4b0b-bb05-db8fadfe122c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sMVFQwd_dg3z89uIih5r5olFlbXLAjl_Gcx0V5IJG-U"  # Replace with real one
COLLECTION_NAME = "arxiv_papers_titles"

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

# Set up Flask app
app = Flask(__name__)
CORS(app)


@app.route('/')
def home():

    return render_template('index.html')



@app.route('/find_similar', methods=['POST'])
def find_similar():

    print("check check ..")
    data = request.get_json()

    # Get query_text and optional top_k from request
    query_text = data.get("query_text", "")
    top_k = 10  # default top_k = 10

    if not query_text.strip():
        return jsonify({"error": "Missing or empty query_text"}), 400

    # Encode the query
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0].astype(np.float16)

    # Search Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(exact=False)
    )

    # Return just titles and scores
    response = [
        {
            "arxiv_id": r.payload.get("arxiv_id", "N/A"),
            "score": r.score
        }
        for r in results
    ]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
