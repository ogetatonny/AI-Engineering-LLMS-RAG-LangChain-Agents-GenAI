# pip install langchain
# pip install langchain-text-splitter
# pip install langchain-community
# pip install langchain-anthropic
# pip install sentence-transformers
# pip install chromadb

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Read the key
api_key = os.getenv('API_KEY')

os.environ["ANTHROPIC_API_KEY"] =api_key

# Step 1: Sample document
print("\nStep 1: Preparing our document")
print("-" * 50)

document = """
Artificial Intelligence (AI) is transforming the way we live and work. Machine learning, 
a subset of AI, enables computers to learn from data without explicit programming. 
Deep learning, a type of machine learning, uses neural networks inspired by the human brain.

Natural Language Processing (NLP) is a branch of AI that helps computers understand and 
process human language. It's used in applications like translation, chatbots, and text analysis.

Computer Vision is another important field in AI. It enables machines to understand and 
process visual information from the world, like images and videos. Applications include 
facial recognition, autonomous vehicles, and medical image analysis.

Reinforcement Learning is a type of machine learning where agents learn by interacting 
with an environment. They receive rewards for good actions and penalties for bad ones. 
This is used in game playing, robotics, and autonomous systems.
"""

print("Document loaded. Length:", len(document), "characters")
print("\nPreview of the document:")
print(document[:200], "...\n")


# Step 2: Text Chunking
print("\nStep 2: Chunking the document")
print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = text_splitter.split_text(document)

print(f"Document has been split into {len(chunks)} chunks.")
print("\nExample chunks:")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-" * 30)
