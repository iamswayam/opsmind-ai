from fastapi.testclient import TestClient

from app import main


def test_chat_retrieves_and_persists_with_postgres(monkeypatch):
    embedding = [1.0] + [0.0] * 767
    question = "CI integration test question"
    filename = "ci-integration-test.txt"
    answer = "Answer from the deterministic test stub."
    document_id = None
    conversation_id = None

    monkeypatch.setattr(main, "embed_text", lambda _: embedding)
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, chunks, history, images: answer,
    )

    connection = main.get_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents (filename, doc_type) VALUES (%s, %s) RETURNING id",
                (filename, "sop"),
            )
            document_id = cursor.fetchone()["id"]
            cursor.execute(
                "INSERT INTO chunks "
                "(document_id, content, embedding, chunk_index, metadata, modality) "
                "VALUES (%s, %s, %s, %s, %s, 'text')",
                (document_id, "Known CI source content.", embedding, 0, main.Json({"page": 1})),
            )
        connection.commit()
    finally:
        connection.close()

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/chat",
                json={"question": question, "document_ids": [document_id]},
            )

        assert response.status_code == 200, response.text
        body = response.json()
        conversation_id = body["conversation_id"]
        assert body["answer"] == answer
        assert body["sources"] == [
            {
                "filename": filename,
                "snippet": "Known CI source content.",
                "page": 1,
                "modality": "text",
            }
        ]

        connection = main.get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT role, content FROM messages "
                    "WHERE conversation_id = %s ORDER BY id",
                    (conversation_id,),
                )
                messages = cursor.fetchall()
            assert messages == [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        finally:
            connection.close()
    finally:
        connection = main.get_connection()
        try:
            with connection.cursor() as cursor:
                if conversation_id is not None:
                    cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
                if document_id is not None:
                    cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            connection.commit()
        finally:
            connection.close()