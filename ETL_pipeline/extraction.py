from typing import Any, Optional, List, Dict
import zlib
import json
import pandas as pd
import os
from bloom_filter import BloomFilter
from process_data import ProcessData
from urllib.parse import urlparse
import arrow
from bs4 import BeautifulSoup
from mlscraper.html import Page
import chardet
import logging
import pybase64
import requests
from bs4 import BeautifulSoup


from create_metadata import (
    get_sintax_opengraph,
    get_sintax_dublincore,
    get_dict_json_ld,
    get_dict_microdata
)

class ExtractionDisk(ProcessData):
    def __init__(self, bucket: str, minio_client: Any, path: str, save_image: Optional[bool], task: Optional[str], column: str,
                 model: str, bloom_filter: Optional[BloomFilter]):
        super().__init__(bloom_filter=bloom_filter, minio_client=minio_client, bucket=bucket, task=task, column=column,
                         model=model)
        self.bucket = bucket
        self.path = path
        self.save_image = save_image
        self.task = task

    def get_files(self):
        try:
            files = os.listdir(self.path)
            logging.info(f"{len(files)} files to be processed")
        except FileNotFoundError:
            logging.error(f"No files on {self.path}")
            return None
        return files

    def run(self, folder_name: str) -> None:
        # Run ETL for all the files on the given path on disk
        files = self.get_files()

        if files:
            for file in files:
                logging.info(f"Starting processing file {file}")
                final_filename = file.split(".")[0]
                final_filename = f"{folder_name}{final_filename}"
                if self.minio_client:
                    checked_obj = self.minio_client.check_obj_exists(self.bucket, final_filename + ".parquet")
                else:
                    checked_obj = False
                if not checked_obj:
                # if True:
                    cached = []
                    processed = []
                    count = 0
                    decompressed_data = self.get_decompressed_file(file)

                    # Processing decompressed data in batch size of 5000 records
                    for line in decompressed_data.splitlines():
                        json_doc = json.loads(line)
                        cached.append(json_doc)
                        count += 1
                        if count % 5000 == 0:
                            processed_df = self.make_extraction(cached)
                            processed.append(processed_df)
                            count = 0
                            cached = []
                    if len(cached) > 0:
                        processed_df = self.make_extraction(cached)
                        processed.append(processed_df)
                    if len(processed) > 0:
                        processed_df = pd.concat(processed).reset_index(drop=True)
                        # processed_df = processed_df[processed_df["title"].notnull()]
                        if self.minio_client:
                            self.load_file_to_minio(final_filename, processed_df)
                        else:
                            processed_df.to_csv(final_filename, index=False)
                else:
                    logging.info(f"file {final_filename} already indexed")
            logging.info("ETL Job run completed")

    def load_file_to_minio(self, file_name, df):
        self.minio_client.save_df_parquet(self.bucket, file_name, df)
        self.bloom_filter.save()
        logging.info("Document successfully indexed on minio")

    def maybe_check_bloom(self, text):
        if self.bloom_filter:
            self.bloom_filter.check_bloom_filter(text)
        else:
            return False

    def create_df(self, ads: list) -> pd.DataFrame:
        seller_username=None
        shipping_location=None
        seller_rating=None
        seller_url =None

        final_dict = []
        for ad in ads:
            domain = ExtractionDisk.get_domain(ad["url"])
            if "ebay.com" in domain and not "/sch/" in ad["url"]:
                html_content = ExtractionDisk.get_decoded_html_from_bytes(ad["content"])
                if html_content:
                    # content_type = ad["content_type"]
                    # parser = ProcessData.get_parser(content_type)
                    soup = BeautifulSoup(html_content, 'html.parser')
                    seller_username, shipping_location, seller_rating, seller_url = self.extract_seller_and_shipping_info(soup)
                    # if seller_username:
                    #     print(f"url: {ad['url']}")
                    #     print(f"Seller Username: {seller_username}")
                dict_df = {
                    "url": ad["url"],
                    "domain": domain,
                    "retrieved": ExtractionDisk.get_time(ad["fetch_time"]),
                    "seller_username": seller_username,
                    "shipping_location": shipping_location,
                    "seller_rating": seller_rating,
                    "seller_url" : seller_url,
                }
                final_dict.append(dict_df)

        df = pd.DataFrame(final_dict)
        return df

    def extract_seller_and_shipping_info(self, soup):
        # Send an HTTP GET request to the URL and retrieve the HTML content
        seller_username=None
        shipping_location=None
        seller_rating=None
        seller_url =None
        # seller_info=None
        # print(soup)
        # raise()
        # Find the div element with the specified class
        seller_info_div = soup.find('div', class_='x-sellercard-atf__info__about-seller')
        seller_ratings_div = seller_ratings = soup.find('span', class_='fdbk-detail-seller-rating__value')
        shipping_location_element = soup.find('div', {'class': 'ux-labels-values col-12 ux-labels-values--itemLocation'})
        # print(soup)
        # raise ValueError()
        if seller_info_div:
            try:
                # Extract the seller's usernametry
                seller_username = seller_info_div.a.span.get_text(strip=True)
                print(f"Seller Username: {seller_username}")
                seller_url = seller_info_div.a.get('href', None)
                # if seller_url:
                #     seller_info = self.get_seller_info(seller_url)
            except Exception:
                print("seller not find")


        if shipping_location_element:
            try:
                # Extract the shipping location text
                shipping_location_text = shipping_location_element.get_text()
                shipping_location = shipping_location_text.split(':')[-1].strip()
                #print("Shipping Location:", shipping_location)
            except Exception:
                print("shiping not find")

        if seller_ratings_div:
            try:
                seller_rating = seller_ratings_div.text
            except Exception:
                print("text rating not find")

        return seller_username, shipping_location, seller_rating, seller_url


    def get_seller_info(self, seller_url):
        import re
        url = seller_url + "&_tab=about"
        try:
            response = requests.get(url, timeout=30)
        except requests.exceptions.Timeout:
            return None, None

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            try:
                span_elements = soup.find_all('span', attrs={'class': 'str-text-span BOLD'})
                info = [span.text for span in span_elements]
                return info
            except Exception:
                return None
        return None

    def get_decompressed_file(self, file):
        with open(f"{self.path}{file}", "rb") as f:
            decompressor = zlib.decompressobj()
            decompressed_data = decompressor.decompress(f.read())
            logging.info(f"file {file} decompressed")
            file_size = len(decompressed_data)
            logging.info(f"The size of the decompressed file is {file_size} bytes")
        return decompressed_data

    @staticmethod
    def get_decoded_html_from_bytes(content):
        try:
            # Attempt decoding with utf-8 encoding
            decoded_bytes = pybase64.b64decode(content, validate=True)
            html_content = decoded_bytes.decode('utf-8')

        except UnicodeDecodeError:
            try:
                # Attempt decoding with us-ascii encoding
                html_content = decoded_bytes.decode('ascii')
                print("us-ascii worked")
            except UnicodeDecodeError:
                # If both utf-8 and us-ascii decoding fail, use chardet for detection
                detection = chardet.detect(decoded_bytes)
                try:
                    html_content = decoded_bytes.decode(detection["encoding"])
                except UnicodeDecodeError as e:
                    logging.error("Error while decoding HTML from bytes due to " + str(e))
                    html_content = None

        except Exception as e:
            html_content = None
            logging.error("Error while decoding HTML from bytes due to " + str(e))

        return html_content

    @staticmethod
    def get_domain(url):
        parsed_url = urlparse(url)
        host = parsed_url.netloc.replace("www.", "")
        return host

    @staticmethod
    def get_time(time):
        # Example epoch timestamp - 1676048703245
        timestamp = arrow.get(time / 1000).format('YYYY-MM-DDTHH:mm:ss.SSSZ')
        return timestamp

    @staticmethod
    def get_text_title(soup):
        if soup:
            try:
                title = soup.title.string if soup.title else None
                text = soup.get_text()
            except Exception as e:
                text = ""
                title = ""
                logging.warning(e)
                logging.warning("Neither title or text")
            return text, title
        else:
            return None, None


    def make_extraction(self, result: List[Dict]) -> pd.DataFrame:
        def log_processed(
                raw_count: int,
                processed_count: int) -> None:
            logging.info(f"{pd.Timestamp.now()}: received {raw_count} articles, total: "
                         f"{processed_count} unique processed")

        cache = []
        count = 0
        hits = len(result)
        # print(hits)
        for val in result:
            # print(val)
            processed = val.get("_source")
            if processed:
                if not ProcessData.remove_text(processed["text"]) and not self.bloom_filter.check_bloom_filter(
                        processed["text"]):
                    count += 1
                    cache.append(processed)
            elif val["content"]:
                count += 1
                cache.append(val)
        log_processed(hits, count)
        df = pd.DataFrame()
        if count > 0:
            df = self.create_df(cache)
            # if not df.empty:
                # df["id"] = df.apply(lambda _: str(uuid.uuid4()), axis=1)
                # df = self.get_location_info(df)
        return df
