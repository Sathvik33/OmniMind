from backend.app.core.secrets_check import validate_production_secrets


class TestSecretsCheck:
    def test_dev_skips_checks(self):
        assert (
            validate_production_secrets(
                app_env="development",
                jwt_secret="aegis-dev-change-me",
                database_url="postgresql://postgres:postgres@localhost/db",
                postgres_password="postgres",
                minio_user="minioadmin",
                minio_password="minioadmin",
            )
            == []
        )

    def test_production_rejects_defaults(self):
        problems = validate_production_secrets(
            app_env="production",
            jwt_secret="change-me-in-production",
            database_url="postgresql://postgres:CHANGE_ME@localhost/db",
            postgres_password="CHANGE_ME",
            minio_user="minioadmin",
            minio_password="minioadmin",
        )
        assert len(problems) >= 2
        assert any("JWT" in p for p in problems)

    def test_production_accepts_strong_secrets(self):
        problems = validate_production_secrets(
            app_env="production",
            jwt_secret="x" * 48,
            database_url="postgresql://app:S3cure!Pass@localhost/db",
            postgres_password="S3cure!Pass",
            minio_user="aegis-minio",
            minio_password="S3cure!Minio99",
        )
        assert problems == []
