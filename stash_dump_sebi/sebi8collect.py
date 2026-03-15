import requests
from bs4 import BeautifulSoup
import json
import os
import time
import random
from urllib.parse import urljoin, urlparse

INPUT_JSON = "sebi_circulars_full_1.json"
DOWNLOAD_DIR = "sebi_pdfs"

headers = {
    "User-Agent": "Mozilla/5.0"
}

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_sebi_pdf(page_url):

    try:
        r = requests.get(page_url, headers=headers, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")

        iframe = soup.find("iframe")

        if not iframe:
            print("No iframe found:", page_url)
            return

        iframe_src = iframe.get("src")

        iframe_url = urljoin(page_url, iframe_src)

        if "file=" in iframe_url:
            pdf_url = iframe_url.split("file=")[1]
        else:
            pdf_url = iframe_url

        filename = os.path.basename(urlparse(pdf_url).path)
        filepath = os.path.join(DOWNLOAD_DIR, filename)

        if os.path.exists(filepath):
            print("Already downloaded:", filename)
            return

        print("Downloading:", filename)

        pdf_response = requests.get(pdf_url, headers=headers, timeout=60)

        with open(filepath, "wb") as f:
            f.write(pdf_response.content)

    except Exception as e:
        print("Error:", page_url, e)


# load JSON
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    circulars = json.load(f)

print("Total circulars:", len(circulars))

for i, item in enumerate(circulars):

    print(f"\n{i+1}/{len(circulars)}")

    download_sebi_pdf(item["url"])

    # polite delay
    sleep_time = random.uniform(1.5, 3.5)
    time.sleep(sleep_time)

print("\nAll downloads attempted.")