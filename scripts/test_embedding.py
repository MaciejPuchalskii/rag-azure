from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

response = client.embeddings.create(
    model=os.getenv("EMBEDDING_DEPLOYMENT"),
    input="What is Delta Lake?"
)

print(len(response.data[0].embedding))