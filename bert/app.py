"""
Flask 预测服务：三分类 + 方面级情感分析
启动: python app.py
测试: curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"text\":\"手机很好用\"}"
"""
from flask import Flask, request, jsonify, make_response
from predict import predict, predict_batch

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/predict", methods=["OPTIONS"])
@app.route("/predict/batch", methods=["OPTIONS"])
def handle_options():
    return make_response("", 204)


@app.route("/predict", methods=["POST"])
def api_predict():
    """单条预测"""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "缺少 text 字段"}), 400

    result = predict(data["text"])
    return jsonify(result)


@app.route("/predict/batch", methods=["POST"])
def api_predict_batch():
    """批量预测"""
    data = request.get_json()

    if not data or "texts" not in data:
        return jsonify({"error": "缺少 texts 字段（列表）"}), 400

    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "texts 必须是列表"}), 400

    results = predict_batch(texts)
    return jsonify({"results": results})


@app.route("/", methods=["GET"])
def index():
    """托管前端页面"""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.route("/health", methods=["GET"])
def api_health():
    """健康检查"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 50)
    print("电商评论情感分析服务")
    print("=" * 50)
    print("POST /predict       单条预测")
    print("POST /predict/batch 批量预测")
    print("GET  /health        健康检查")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
