import os
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv()

MINIO_URL = os.getenv("MINIO_URL")
if not MINIO_URL:
    raise ValueError("MINIO_URL environment variable is not set")
    
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER")
if not MINIO_ACCESS_KEY:
    raise ValueError("MINIO_ROOT_USER environment variable is not set")
    
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD")
if not MINIO_SECRET_KEY:
    raise ValueError("MINIO_ROOT_PASSWORD environment variable is not set")
    
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() == "true"

class MinioClient:
    def __init__(self):
        self.client = Minio(
            MINIO_URL,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )
        self.bucket_name = "omnimind-raw"
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except S3Error as err:
            print(f"MinIO bucket error: {err}")

    def upload_file(self, object_name: str, file_path: str, content_type: str = "application/octet-stream") -> str:
        """Upload a file to MinIO and return the object path."""
        try:
            self.client.fput_object(
                self.bucket_name,
                object_name,
                file_path,
                content_type=content_type
            )
            return f"{self.bucket_name}/{object_name}"
        except S3Error as err:
            print(f"Upload error: {err}")
            raise

    def get_presigned_url(self, object_name: str) -> str:
        """Get a temporary download URL."""
        return self.client.presigned_get_object(self.bucket_name, object_name)

minio_client = MinioClient()
