from typing import Any, Optional, List
import pandas as pd
from minio import Minio
from io import BytesIO
from minio.error import S3Error
import pickle
import yaml
import io
import random
import time


class MinioClient:
    def __init__(self, access_key: str, secret_access_key: str):
        self.config = yaml.safe_load(open("config.yml"))
        self.access_key = access_key
        self.secret_key = secret_access_key
        self.client = Minio(
            self.config.get("minio_client_endpoint"),
            access_key=access_key,
            secret_key=secret_access_key,
            secure=False)

    def get_storage_options(self) -> dict:
        return {"key": self.access_key,
                "secret": self.secret_key,
                "endpoint_url": self.config.get("minio_endpoint_url")}

    def put_obj(self, obj: Any, file_name: str, bucket: str):
        raise NotImplementedError

    def get_obj(self, bucket: str, file_name: str):
        obj = self.client.get_object(
            bucket,
            file_name)
        return obj

    def get_model(self, bucket_name: str, obj_name: str, path: str):
        self.client.fget_object(bucket_name, obj_name, path)


    def read_csv(self, bucket: str, file_name: str) -> pd.DataFrame:
        df = pd.read_csv(
            f"s3://{bucket}/{file_name}",
            storage_options=self.get_storage_options)
        return df

    def read_df_parquet(self, bucket: str, file_name: str) -> pd.DataFrame:
        df = pd.read_parquet(
            f"s3://{bucket}/{file_name}",
            storage_options=self.get_storage_options())
        return df

    # def save_df_parquet(self, bucket: str, file_name: str, df: pd.DataFrame) -> None:
    #     file_name = f"s3://{bucket}/{file_name}.parquet"
    #     storage = self.get_storage_options()
    #     df.to_parquet(
    #         file_name,
    #         index=False,
    #         storage_options=storage)
    #     print(f"{file_name} saved on bucket {bucket}")

    def save_df_parquet(self, bucket: str, file_name: str, df: pd.DataFrame) -> None:
        # Convert DataFrame to Parquet file in memory
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        # Define the retry parameters
        max_retries = 5

        for attempt in range(max_retries):
            try:
                self.client.put_object(
                    bucket_name=bucket,
                    object_name=f"{file_name}.parquet",
                    data=parquet_buffer,
                    length=parquet_buffer.getbuffer().nbytes,
                    content_type="application/octet-stream"
                )
                print(f"{file_name}.parquet saved on bucket {bucket}")
                break
            except S3Error as e:
                if e.code == 'SlowDown':
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Throttling error on attempt {attempt + 1}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error occurred: {e}")
                    raise
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise

    def store_image(self, image: Any, file_name: str, length: int, bucket_name: str):
        self.client.put_object(
            bucket_name,
            file_name,
            data=image,
            length=length,
        )

    def check_obj_exists(self, bucket: str, file_path: str):
        try:
            return self.client.stat_object(bucket, file_path) is not None
        except (S3Error, ValueError) as err:
            if "empty bucket name" in str(err):
                # Object doesn't exist or empty bucket name error
                return False
            if isinstance(err, S3Error) and err.code == 'NoSuchKey':
                # Object doesn't exist
                return False

    def get_bloom_filter(self, file_name: str):
        response = self.client.get_object("bloom-filter", file_name)
        file_obj = BytesIO()
        for d in response.stream(32 * 1024):
            file_obj.write(d)
        # Reset the file-like object's position to the beginning
        file_obj.seek(0)
        bloom_filter = pickle.load(file_obj)
        return bloom_filter

    def save_bloom(self, bloom, file_name: str):
        pickle_data = pickle.dumps(bloom)
        file_obj = BytesIO(pickle_data)
        self.client.put_object("bloom-filter", file_name, file_obj, len(pickle_data))

    def list_objects_names(self, bucket: str, date: Optional[str]) -> List[str]:
        objects = self.client.list_objects(bucket, recursive=True)
        file_names = []
        if date:
            for obj in objects:
                files = obj.object_name.split("/")
                if files[0] == date:
                    file_names.append(obj.object_name)
        for obj in objects:
            file_names.append(obj.object_name)
        return file_names
