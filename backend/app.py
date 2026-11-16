
# Import necessary libraries
from flask import Flask, jsonify, request
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Import RAG utility functions
from rag_utils import rag, retrieve
from validation import ValidationError, parse_rag_request


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
VECTORSTORE_PATH = "faiss_index"


# ---------------------------------------------------------
# Initialize the Flask application
# ---------------------------------------------------------

rag_api = Flask("Aurora Stay API")


# ---------------------------------------------------------
# Load the embedding model
# ---------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# ---------------------------------------------------------
# Load the saved FAISS vector store
# ---------------------------------------------------------

vectorstore = FAISS.load_local(
    VECTORSTORE_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Vectors stored: {vectorstore.index.ntotal}")


# ---------------------------------------------------------
# Default route
# ---------------------------------------------------------

@rag_api.get("/")
def home():
    """
    Handles GET requests to the root URL.
    """
    return "Welcome to the Aurora Stay RAG API!"


# ---------------------------------------------------------
# JSON error handlers
# ---------------------------------------------------------

@rag_api.errorhandler(404)
def handle_not_found(error):
    """
    Returns a JSON response for unknown routes.
    """
    return jsonify({"error": {"message": "Not found."}}), 404


@rag_api.errorhandler(405)
def handle_method_not_allowed(error):
    """
    Returns a JSON response for unsupported HTTP methods.
    """
    return jsonify({"error": {"message": "Method not allowed."}}), 405


# ---------------------------------------------------------
# Relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    """
    Handles POST requests to /v1/relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 4
    }

    Returns the top-k relevant chunks retrieved from
    the vector store.
    """

    # Validate the JSON request body
    try:
        params = parse_rag_request(
            request.get_json(silent=True),
            require_generation=False
        )
    except ValidationError as error:
        return jsonify(error.to_dict()), 400

    # Retrieve relevant chunks
    try:
        chunks = retrieve(
            query=params.query,
            k=params.k,
            vectorstore=vectorstore
        )
    except Exception:
        rag_api.logger.exception("Chunk retrieval failed")
        return jsonify({
            "error": {"message": "Unable to retrieve relevant chunks."}
        }), 502

    # Return the retrieved chunks
    return jsonify({
        "query": params.query,
        "k": params.k,
        "relevant_chunks": chunks
    })


# ---------------------------------------------------------
# Answer with relevant chunks endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    """
    Handles POST requests to /v1/answer_with_relevant_chunks.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 4,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns both the generated answer and the relevant
    chunks used to generate the answer.
    """

    # Validate the JSON request body
    try:
        params = parse_rag_request(request.get_json(silent=True))
    except ValidationError as error:
        return jsonify(error.to_dict()), 400

    # Run the complete RAG pipeline
    try:
        answer, chunks = rag(
            query=params.query,
            k=params.k,
            model_name=params.model_name,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            vectorstore=vectorstore
        )
    except Exception:
        rag_api.logger.exception("RAG generation failed")
        return jsonify({
            "error": {"message": "Unable to generate an answer."}
        }), 502

    # Return answer, chunks, and parameters used
    return jsonify({
        "query": params.query,
        "k": params.k,
        "model_name": params.model_name,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_tokens": params.max_tokens,
        "relevant_chunks": chunks,
        "answer": answer
    })


# ---------------------------------------------------------
# Answer endpoint
# ---------------------------------------------------------

@rag_api.post("/v1/answer")
def answer():
    """
    Handles POST requests to /v1/answer.

    Example request:
    {
        "query": "What is the cancellation policy?",
        "k": 4,
        "model_name": "gpt-4o-mini",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 512
    }

    Returns only the generated answer.
    """

    # Validate the JSON request body
    try:
        params = parse_rag_request(request.get_json(silent=True))
    except ValidationError as error:
        return jsonify(error.to_dict()), 400

    # Run the complete RAG pipeline
    try:
        generated_answer, _ = rag(
            query=params.query,
            k=params.k,
            model_name=params.model_name,
            temperature=params.temperature,
            top_p=params.top_p,
            max_tokens=params.max_tokens,
            vectorstore=vectorstore
        )
    except Exception:
        rag_api.logger.exception("RAG generation failed")
        return jsonify({
            "error": {"message": "Unable to generate an answer."}
        }), 502

    # Return only the answer
    return jsonify({
        "answer": generated_answer
    })


# ---------------------------------------------------------
# Run the Flask application
# ---------------------------------------------------------

if __name__ == "__main__":
    rag_api.run(
        host="0.0.0.0",
        port=7860,
        debug=True
    )
