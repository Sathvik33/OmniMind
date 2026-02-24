import uuid
from backend.app.vectorstore.collection_manager import CollectionManager


def test_add_and_query():
    cm = CollectionManager(collection_name="test_collection")

    doc_id = str(uuid.uuid4())

    cm.add_documents(
        documents=["Aegis is a powerful multi-model RAG system."],
        ids=[doc_id],
        metadata=[{"source": "unit_test"}]
    )

    result = cm.query("What is Aegis?", n_results=1)

    assert result["documents"] is not None
    assert len(result["documents"][0]) > 0