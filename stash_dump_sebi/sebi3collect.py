import requests

url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"

headers = {
    "Accept": "*/*",
    "Accept-Language": "en-GB,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://www.sebi.gov.in",
    "Pragma": "no-cache",
    "Referer": "https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&ssid=7&smid=0",
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36",
}

cookies = {
    "JSESSIONID": "517F2C7D85568A90E6565F993052602F"
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

response = requests.post(url, headers=headers, cookies=cookies, data=data)

with open("sebi_raw.html", "w", encoding="utf-8") as f:
    f.write(response.text)

print("Raw response saved to sebi_raw.html")