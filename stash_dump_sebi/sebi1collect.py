import requests
import pandas as pd

base_url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslist.jsp"

params = {
    "next": "n",
    "deptId": "-1",
    "sid": "1",
    "ssText": "",
    "smid": "0",
    "cid": "1",
    "year": "",
    "page": 1
}

data = []

for page in range(1,200):
    params["page"] = page
    r = requests.get(base_url, params=params)

    if r.status_code != 200:
        break

    json_data = r.json()

    if len(json_data) == 0:
        break

    data.extend(json_data)

df = pd.DataFrame(data)
df.to_csv("sebi_circulars_dump.csv", index=False)
