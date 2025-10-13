# server.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from vector import retriever

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, origins=["*"])

# Initialize mental health LLM and chain
print("[INFO] Initializing mental health model...")
model = Ollama(model="gemma3:1b")

template = """
You are a compassionate mental health therapist.

Here are some relevant previous Q&A excerpts:
{reviews}

Here is the user's question:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
print("[INFO] Model and chain ready.")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer": "Please provide a question."}), 400

    try:
        # Retrieve Q&A context
        reviews = retriever.invoke(question)
        # Generate answer
        result = chain.invoke({"reviews": reviews, "question": question})
        return jsonify({"answer": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return send_from_directory("templates", "chatbot.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
