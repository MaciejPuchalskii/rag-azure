import os
from dotenv import load_dotenv

from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery


load_dotenv()


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "databricks-docs")


def validate_env():
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "CHAT_DEPLOYMENT": CHAT_DEPLOYMENT,
        "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
        "AZURE_SEARCH_KEY": AZURE_SEARCH_KEY,
        "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
    }

    missing = [name for name, value in required_vars.items() if not value]

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )


def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )


def create_embedding(openai_client: AzureOpenAI, text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text,
    )

    return response.data[0].embedding


def search_similar_chunks(
    search_client: SearchClient,
    question_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    vector_query = VectorizedQuery(
        vector=question_embedding,
        k_nearest_neighbors=top_k,
        fields="contentVector",
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "content", "source"],
        top=top_k,
    )

    chunks = []

    for result in results:
        chunks.append(
            {
                "id": result.get("id"),
                "content": result.get("content"),
                "source": result.get("source"),
                "score": result.get("@search.score"),
            }
        )

    return chunks


def build_context(chunks: list[dict]) -> str:
    context_parts = []

    for index, chunk in enumerate(chunks, start=1):
        source = chunk["source"]
        content = chunk["content"]

        context_parts.append(
            f"[Source {index}: {source}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def generate_answer(
    openai_client: AzureOpenAI,
    question: str,
    chunks: list[dict],
) -> str:
    context = build_context(chunks)

    system_prompt = """
You are a helpful technical assistant.
Answer the user's question using only the provided context.
If the answer is not present in the context, say that you cannot find the answer in the provided documentation.
Keep the answer clear, technical and practical.
At the end, include a "Sources" section with the source file names used.
"""

    user_prompt = f"""
Context:

{context}

Question:

{question}
"""

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ]
    )

    return response.choices[0].message.content


def ask(question: str):
    openai_client = get_openai_client()
    search_client = get_search_client()

    print("Creating question embedding...")
    question_embedding = create_embedding(openai_client, question)

    print("Searching Azure AI Search...")
    chunks = search_similar_chunks(
        search_client=search_client,
        question_embedding=question_embedding,
        top_k=5,
    )

    if not chunks:
        print("No chunks found.")
        return

    print("\nTop chunks found:")
    for i, chunk in enumerate(chunks, start=1):
        print(f"{i}. Source: {chunk['source']} | Score: {chunk['score']}")

    print("\nGenerating answer with chat model...")
    answer = generate_answer(
        openai_client=openai_client,
        question=question,
        chunks=chunks,
    )

    print("\nAnswer:")
    print(answer)


def main():
    validate_env()

    question = input("Ask a question about Databricks documentation: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    ask(question)


if __name__ == "__main__":
    main()