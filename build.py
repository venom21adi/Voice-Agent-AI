from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

# Use Ollama for embeddings
Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434"
)

docs = SimpleDirectoryReader(
    r"C:\Users\adity\Documents\VoiceAgentLab\data",  # folder path
    recursive=True
).load_data()

docs = SimpleDirectoryReader("data", recursive=True).load_data()

print("===== LOADED DOCUMENTS =====")
for d in docs:
    print(d.metadata)
print("===== END =====")


index = VectorStoreIndex.from_documents(docs)

index.storage_context.persist(persist_dir="index_store")
print("Index built and saved.")
