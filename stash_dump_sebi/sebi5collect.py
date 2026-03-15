import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse

def download_sebi_pdf(page_url):

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(page_url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")

    iframe = soup.find("iframe")

    if not iframe:
        print("No iframe found")
        return

    iframe_src = iframe.get("src")

    # convert relative URL to absolute
    iframe_url = urljoin(page_url, iframe_src)

    # extract actual pdf link from query
    if "file=" in iframe_url:
        pdf_url = iframe_url.split("file=")[1]
    else:
        pdf_url = iframe_url

    print("PDF URL:", pdf_url)

    pdf_response = requests.get(pdf_url, headers=headers)

    filename = os.path.basename(urlparse(pdf_url).path)

    with open(filename, "wb") as f:
        f.write(pdf_response.content)

    print("Downloaded:", filename)


# Example usage
download_sebi_pdf(
"https://www.sebi.gov.in/legal/circulars/jan-2026/specification-of-the-consequential-requirements-with-respect-to-amendment-of-securities-and-exchange-board-of-india-merchant-bankers-regulations-1992_98831.html"
)