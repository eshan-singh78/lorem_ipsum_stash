import requests
from bs4 import BeautifulSoup
import json
import time
import random

url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"

headers = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.sebi.gov.in",
    "Referer": "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0",
    "User-Agent": "Mozilla/5.0"
}

base_data = {
    "nextValue": "0",
    "next": "n",
    "search": "",
    "fromDate": "",
    "toDate": "",
    "fromYear": "",
    "toYear": "",
    "deptId": "-1",
    "sid": "1",
    "ssid": "7",
    "smid": "0",
    "ssidhidden": "7",
    "intmid": "-1",
    "sText": "Legal",
    "ssText": "Circulars",
    "smText": "",
    "doDirect": "1"
}

results = []

for page in range(111):   # 0 → 110 pages

    print(f"Fetching page {page+1}/111")

    data = base_data.copy()
    data["nextValue"] = str(page)

    try:
        response = requests.post(url, headers=headers, data=data, timeout=30)

        soup = BeautifulSoup(response.text, "html.parser")

        for row in soup.select("table tr"):
            date_cell = row.find("td")
            link = row.find("a")

            if date_cell and link:
                results.append({
                    "date": date_cell.get_text(strip=True),
                    "title": link.get_text(strip=True),
                    "url": link["href"]
                })

    except Exception as e:
        print("Error on page", page, e)

    # polite delay to avoid rate limiting
    sleep_time = random.uniform(1.5, 3.5)
    print(f"Sleeping {sleep_time:.2f}s")
    time.sleep(sleep_time)

print("Total records collected:", len(results))

with open("sebi_circulars_full_1.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved to sebi_circulars_full_1.json")