import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the dataset
file_path = "arxiv-metadata-oai-snapshot.json"
df = pd.read_json(file_path, lines=True)

print("First 5 records:", df.head())

# Set up Flask app
app = Flask(__name__)
CORS(app)

@app.route('/print_transcript', methods=['POST'])
def print_transcript():
    # Example logic: return the first abstract in JSON
    return jsonify({"abstract": df.iloc[0]["abstract"]})

if __name__ == '__main__':
    app.run(debug=True)
