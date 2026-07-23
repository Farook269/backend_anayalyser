# # flask_api.py
# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from backend import create_pdf_retriever, generate_answer

# # ⚙️ Configuration
# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# app = Flask(__name__)
# CORS(app)  # Enable CORS for Postman and frontend use

# retriever = None  # Global retriever object (in-memory)

# # 🩺 Health Check Endpoint
# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({"message": "✅ Flask RAG Chatbot API is running!"})


# # 📁 Upload PDF (Override Mode)
# @app.route("/upload", methods=["POST"])
# def upload_pdf():
#     """
#     Upload a new PDF file.
#     Deletes old files and creates a new retriever.
#     """
#     global retriever

#     if "file" not in request.files:
#         return jsonify({"error": "❌ No file part in request."}), 400

#     file = request.files["file"]

#     if file.filename == "":
#         return jsonify({"error": "❌ No file selected."}), 400

#     # 🧹 Delete old PDFs
#     for old_file in os.listdir(UPLOAD_DIR):
#         try:
#             os.remove(os.path.join(UPLOAD_DIR, old_file))
#         except Exception as e:
#             print(f"⚠️ Could not delete {old_file}: {e}")

#     # 💾 Save new PDF
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     file.save(file_path)

#     # 🧠 Build retriever
#     retriever = create_pdf_retriever([file_path])

#     return jsonify({"status": "success", "file_uploaded": file.filename})


# # -----------------------------
# # 💬 Chat Endpoint
# # -----------------------------
# @app.route("/chat", methods=["POST"])
# def chat_with_pdf():
#     """
#     Ask a question about the uploaded PDF.
#     """
#     global retriever

#     if retriever is None:
#         return jsonify({"error": "❌ No PDF uploaded yet. Please upload a file first."}), 400

#     data = request.get_json()
#     if not data or "question" not in data:
#         return jsonify({"error": "❌ Missing 'question' in request body."}), 400

#     question = data["question"]
#     answer = generate_answer(question, retriever)

#     return jsonify({"question": question, "answer": answer})


# if __name__ == "__main__":
#     app.run(debug=True, port=5000)
