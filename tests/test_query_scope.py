"""Session-scoped artifact resolution and cross-user isolation."""

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.app.services.query_scope import (
    get_owned_session,
    resolve_session_artifact_ids,
)


class TestQueryScope:
    def test_get_owned_session_404_for_other_user(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        with pytest.raises(HTTPException) as ei:
            get_owned_session(db, session_id=1, user_id=99)
        assert ei.value.status_code == 404

    def test_resolve_session_artifact_ids_filters_user_and_session(self):
        chat = MagicMock()
        db = MagicMock()

        # First query: get_owned_session; second: artifact ids
        owned_q = MagicMock()
        owned_q.filter.return_value.first.return_value = chat

        arts_q = MagicMock()
        arts_q.join.return_value.filter.return_value.distinct.return_value.order_by.return_value.all.return_value = [
            (10,),
            (11,),
        ]

        db.query.side_effect = [owned_q, arts_q]

        ids = resolve_session_artifact_ids(db, user_id=5, session_id=3)
        assert ids == [10, 11]
        # ensure ownership check was applied
        owned_q.filter.assert_called()
        arts_q.join.assert_called()

    def test_resolve_empty_when_no_embeddings(self):
        chat = MagicMock()
        db = MagicMock()
        owned_q = MagicMock()
        owned_q.filter.return_value.first.return_value = chat
        arts_q = MagicMock()
        arts_q.join.return_value.filter.return_value.distinct.return_value.order_by.return_value.all.return_value = []
        db.query.side_effect = [owned_q, arts_q]

        assert resolve_session_artifact_ids(db, user_id=1, session_id=2) == []
