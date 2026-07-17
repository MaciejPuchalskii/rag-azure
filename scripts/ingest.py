import os
import re
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from openai import AzureOpenAI

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

DOCS_DIR = Path("docs")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "databricks-docs")


def validate_env():
    required_vars = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_KEY": AZURE_OPENAI_KEY,
        "AZURE_OPENAI_API_VERSION": AZURE_OPENAI_API_VERSION,
        "EMBEDDING_DEPLOYMENT": EMBEDDING_DEPLOYMENT,
        "AZURE_SEARCH_ENDPOINT": AZURE_SEARCH_ENDPOINT,
        "AZURE_SEARCH_KEY": AZURE_SEARCH_KEY,
        "AZURE_SEARCH_INDEX_NAME": AZURE_SEARCH_INDEX_NAME,
    }

    missing = [name for name, value in required_vars.items() if not value]

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")


def read_pdf_text(pdf_path: Path) -> str:
    print(f"Reading PDF: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages_text.append(text)
        else:
            print(f"Warning: no text extracted from page {page_number} in {pdf_path.name}")

    return "\n".join(pages_text)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    return splitter.split_text(text)


def create_document_id(source: str, chunk_index: int, chunk_text: str) -> str:
    raw_id = f"{source}_{chunk_index}_{chunk_text[:50]}"
    hash_id = hashlib.md5(raw_id.encode("utf-8")).hexdigest()
    return f"{source.replace('.', '_')}_{chunk_index}_{hash_id}"


def create_embedding(client: AzureOpenAI, text: str):
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text,
    )

    return response.data[0].embedding


def upload_in_batches(search_client: SearchClient, documents, batch_size: int = 50):
    total_uploaded = 0

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]

        result = search_client.upload_documents(documents=batch)

        failed = [r for r in result if not r.succeeded]

        if failed:
            print("Some documents failed to upload:")
            for item in failed:
                print(item)

            raise RuntimeError("Upload failed for some documents.")

        total_uploaded += len(batch)
        print(f"Uploaded {total_uploaded}/{len(documents)} documents")


def main():
    validate_env()

    if not DOCS_DIR.exists():
        raise FileNotFoundError("Folder 'docs' does not exist. Create it and put PDF files inside.")

    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in 'docs' folder.")

    openai_client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY),
    )

    all_documents = []

    for pdf_path in pdf_files:
        raw_text = read_pdf_text(pdf_path)
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            print(f"Skipping empty PDF: {pdf_path.name}")
            continue

        chunks = create_chunks(cleaned_text)

        print(f"Created {len(chunks)} chunks from {pdf_path.name}")

        for chunk_index, chunk in enumerate(chunks):
            print(f"Generating embedding for {pdf_path.name}, chunk {chunk_index + 1}/{len(chunks)}")

            embedding = create_embedding(openai_client, chunk)

            document = {
                "id": create_document_id(pdf_path.name, chunk_index, chunk),
                "content": chunk,
                "source": pdf_path.name,
                "contentVector": embedding,
            }

            all_documents.append(document)

    if not all_documents:
        print("No documents to upload.")
        return

    print(f"Uploading {len(all_documents)} chunks to Azure AI Search index: {AZURE_SEARCH_INDEX_NAME}")

    upload_in_batches(search_client, all_documents)

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    main()