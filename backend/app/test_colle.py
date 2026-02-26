from backend.app.vectorstore.chroma_client import get_chroma_client

client = get_chroma_client()
print(client.list_collections())
print(client.get_collection("aegis_collection").count())
print(client.get_collection("test_collection").count())