from google.cloud import storage
import os
import json
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from config import GCP_SERVICE_ACCOUNT_JSON, GCP_BUCKET_MODEL_ARTIFACT
from google.oauth2 import service_account
from datetime import datetime


class GCPWorker:
    def __init__(self, bucket_name, source_path, dest_path, check_exists: bool = True):
        self.source_path = source_path
        self.dest_path = dest_path
        self.credentials_dict = json.loads(GCP_SERVICE_ACCOUNT_JSON)
        self.credentials = service_account.Credentials.from_service_account_info(self.credentials_dict)
        self.storage_client = storage.Client(credentials=self.credentials)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.check_exists = check_exists

    def upload_to_gcs(self) -> bool:
        print("Uploading model artifact to GPC Bucket")
        try:
            timestamp = datetime.now().strftime('%d-%m-%Y-%H-%M')
            base_dest_path = f"{timestamp}/{self.dest_path}"
            
            if self.check_exists:
                blobs = self.bucket.list_blobs(prefix=base_dest_path)
                if list(blobs):
                    print(f"Dir {base_dest_path} already exist")
                    return False
            
            # Рекурсивная загрузка
            for root, dirs, files in os.walk(self.source_path):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, self.source_path)
                    blob_path = os.path.join(base_dest_path, relative_path)
                    
                    blob = self.bucket.blob(blob_path)
                    blob.upload_from_filename(local_path)

            print("GCP uploaded")        
            return True
        except Exception as e:
            print(f"GCP upload error: {e}")
            return False

    def download_from_gcp(self, source_path: str, dest_path: str) -> bool:
        print(f"Downloading from GCS: {source_path} to {dest_path}")
        try:
            if source_path.endswith('/'):
                blobs = list(self.bucket.list_blobs(prefix=source_path))
                if not blobs:
                    print(f"Directory {source_path} is empty or does not exist")
                    return False
                for blob in blobs:
                    relative_path = blob.name.replace(source_path,'', 1)
                    target_path = os.path.join(dest_path, relative_path)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    blob.download_to_filename(target_path)
            else:
                blob = self.bucket.blob(source_path)
                if not blob.exists():
                    print(f"File {source_path} does not exist in bucket")
                    return False
                
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                blob.download_to_filename(dest_path)

            return True
        except Exception as e:
            print(f"Error downloading from GCS: {str(e)}")
            return False

def main():
    gcp_worker = GCPWorker(GCP_BUCKET_MODEL_ARTIFACT, "data/model/test", "model/")
    #gcp_worker.upload_to_gcs()
    gcp_worker.download_from_gcp("10-02-2025-20-37/model/logs/", "/Users/max/PycharmProjects/Topic/test123/")

if __name__ == "__main__":
    main()
