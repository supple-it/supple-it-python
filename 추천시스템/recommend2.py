import os
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

try:
    df = pd.read_csv('추천시스템/preprocessed_비타민.csv')
    df = df.dropna(subset=['기능성', '제품명'])
    df['기능성_제품명'] = df['기능성'] + " " + df['제품명']
except Exception as e:
    print(f"Error loading or preprocessing data: {e}")

if os.path.exists("tfidf_vectorizer.pkl"):
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
        tfidf_matrix = tfidf.transform(df['기능성_제품명'])
else:
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['기능성_제품명'])
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf,f)

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend(keyword, limit=9):
    try:
        keyword_vec = tfidf.transform([keyword])
        sim_scores = cosine_similarity(keyword_vec, tfidf_matrix)[0]
        sorted_indices = sim_scores.argsort()[::-1]
        recommended_products = set()
        result_indices = []
        for idx in sorted_indices:
            product_name = df['제품명'].iloc[idx]
            if product_name not in recommended_products:
                recommended_products.add(product_name)
                result_indices.append(idx)
                if len(recommended_products) >= limit:
                    break
        return df[['제품명', '기능성']].iloc[result_indices]
    except Exception as e:
        print(f"Error in recommendation function: {e}")
        return pd.DataFrame()

@app.route('/recommend', methods=['GET'])
def get_recommendations():
    keyword = request.args.get('keyword', '')
    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400
    recommendations = recommend(keyword)
    return jsonify({
        "keyword": keyword,
        "count": len(recommendations),
        "recommendations": recommendations['제품명'].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)