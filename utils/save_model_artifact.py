from google.cloud import storage
import os
import json
from config import GCP_SERVICE_ACCOUNT_JSON
from google.oauth2 import service_account
from datetime import datetime


def upload_to_gcs(bucket_name: str, source_path: str, dest_path: str, check_exists: bool = True) -> bool:
    print("Uploading model artifact to GPC Bucket")
    try:

        credentials_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(bucket_name)
        
        # Формируем путь с датой
        timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M')
        base_dest_path = f"{timestamp}/{dest_path}"
        
        if check_exists:
            blobs = bucket.list_blobs(prefix=base_dest_path)
            if list(blobs):
                print(f"Директория {base_dest_path} уже существует")
                return False
        
        # Рекурсивная загрузка
        for root, dirs, files in os.walk(source_path):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, source_path)
                blob_path = os.path.join(base_dest_path, relative_path)
                
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                
        return True
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return False

# def main():
#     success = upload_to_gcs(
#         config.GCP_BUCKET_MODEL_ARTIFACT,
#         "data/model/",  # Директория для загрузки
#         "model/"   # Имя папки в бакете
#     )

# if __name__ == "__main__":
#     main()
