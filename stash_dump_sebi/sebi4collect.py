import requests
from bs4 import BeautifulSoup
import json

url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"

headers = {
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.sebi.gov.in",
    "Referer": "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0",
    "User-Agent": "Mozilla/5.0"
}

data = {
    "nextValue": "1",
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

response = requests.post(url, headers=headers, data=data)

soup = BeautifulSoup(response.text, "html.parser")

results = []

for row in soup.select("table tr"):
    date_cell = row.find("td")
    link = row.find("a")

    if date_cell and link:
        results.append({
            "date": date_cell.get_text(strip=True),
            "title": link.get_text(strip=True),
            "url": link["href"]
        })

with open("sebi_circulars.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Saved to sebi_circulars.json")