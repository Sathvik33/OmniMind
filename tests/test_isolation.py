"""Cross-tenant isolation: User A cannot resolve User B's session artifacts."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from backend.app.api.query import _scope
from backend.app.services.query_scope import get_owned_session, resolve_session_artifact_ids


class TestCrossTenantIsolation:
    def test_other_users_session_not_found(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        with pytest.raises(HTTPException) as ei:
            get_owned_session(db, session_id=5, user_id=1)
        assert ei.value.status_code == 404

    def test_scope_rejects_foreign_session(self):
        user = MagicMock()
        user.id = 1
        db = MagicMock()
        with patch(
            "backend.app.api.query.resolve_session_artifact_ids",
            side_effect=HTTPException(status_code=404, detail="Chat not found"),
        ):
            with pytest.raises(HTTPException) as ei:
                _scope(db, user, session_id=999)
            assert ei.value.status_code == 404

    def test_resolve_never_returns_other_session_ids_without_ownership(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        with pytest.raises(HTTPException):
            resolve_session_artifact_ids(db, user_id=2, session_id=1)
