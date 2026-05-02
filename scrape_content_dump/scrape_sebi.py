#!/usr/bin/env python3
"""Scraper for SEBI registered Investment Advisers (FPI category intmId=13).
Fetches fresh session automatically, paginates through all records, and saves to JSON.
"""
import requests
import re
import json
import time
import sys
from html import unescape

BASE_URL = 'https://www.sebi.gov.in/sebiweb/ajax/other/getintmfpiinfo.jsp'
LANDING_URL = 'https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId=13'

HEADERS = {
    'Accept': '*/*',
    'Accept-Language': 'en-GB,en;q=0.9',
    'Cache-Control': 'no-cache',
    'Content-type': 'application/x-www-form-urlencoded',
    'Origin': 'https://www.sebi.gov.in',
    'Pragma': 'no-cache',
    'Referer': LANDING_URL,
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-GPC': '1',
    'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1',
    'sec-ch-ua': '"Brave";v="147", "Not.A/Brand";v="8", "Chromium";v="147"',
    'sec-ch-ua-mobile': '?1',
    'sec-ch-ua-platform': '"iOS"',
}

FIELD_MAP = {
    'Name': 'name',
    'Registration No.': 'registration_no',
    'E-mail': 'email',
    'Telephone': 'phone',
    'Fax No.': 'fax',
    'Address': 'address',
    'Contact Person': 'contact_person',
    'Correspondence Address': 'correspondence_address',
    'Validity': 'validity',
}

RECORD_END_FIELD = 'Validity'

DELAY_BETWEEN_PAGES = 1.5  # seconds


def create_session():
    """Visit the landing page to get a fresh JSESSIONID, return configured session."""
    session = requests.Session()
    session.headers.update(HEADERS)
    resp = session.get(LANDING_URL, timeout=30)
    resp.raise_for_status()
    if 'JSESSIONID' not in session.cookies:
        raise RuntimeError("Failed to obtain JSESSIONID from landing page")
    return session


def fetch_page(session, page_idx):
    """Fetch a single page of results. Returns raw HTML."""
    data = {
        'nextValue': str(page_idx),
        'next': 'n',
        'intmId': '13',
        'contPer': '',
        'name': '',
        'regNo': '',
        'email': '',
        'location': '',
        'exchange': '',
        'affiliate': '',
        'alp': '',
        'language': '2',
        'model': '',
        'esgCategory': '',
        'doDirect': '0',
        'intmIds': '',
    }
    resp = session.post(BASE_URL, data=data, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_total_pages(html):
    """Extract total page count from the pagination hidden field."""
    m = re.search(r"name='totalpage'\s+value=(\d+)", html)
    if m:
        return int(m.group(1))
    # Fallback: count from pagination links
    pages = re.findall(r"searchFormFpi\('n',\s*'(\d+)'\)", html)
    if pages:
        return int(max(pages, key=int)) + 1
    return None


def parse_records(html):
    """Parse all records from the HTML response."""
    records = []
    title_pattern = re.compile(r"<div class='title'><span>(.*?)</span></div>", re.DOTALL)
    value_pattern = re.compile(r"<div class='value[^>]*><span>(.*?)</span></div>", re.DOTALL)

    titles = title_pattern.findall(html)
    raw_values = value_pattern.findall(html)
    values = [unescape(re.sub(r'<[^>]+>', '', v).strip()) for v in raw_values]

    current = {}
    for title, value in zip(titles, values):
        if title in FIELD_MAP:
            current[FIELD_MAP[title]] = value
            if title == RECORD_END_FIELD:
                if current.get('name'):
                    records.append(current)
                current = {}

    return records


def scrape_all(output_file='sebi_fpi_records.json'):
    session = create_session()
    print(f"Session established (JSESSIONID={session.cookies['JSESSIONID'][:8]}...)")

    # First page to get total count
    html = fetch_page(session, 0)
    total_pages = extract_total_pages(html)
    if total_pages is None:
        print("ERROR: Could not determine total pages from first response")
        sys.exit(1)

    all_records = parse_records(html)
    print(f"Page 1/{total_pages}: {len(all_records)} records")

    for page_idx in range(1, total_pages):
        time.sleep(DELAY_BETWEEN_PAGES)

        for attempt in range(3):
            try:
                html = fetch_page(session, page_idx)
                records = parse_records(html)
                if not records:
                    # Possibly session expired — refresh
                    print(f"\n  Empty response on page {page_idx + 1}, refreshing session...")
                    session = create_session()
                    html = fetch_page(session, page_idx)
                    records = parse_records(html)
                break
            except requests.RequestException as e:
                if attempt < 2:
                    print(f"\n  Request failed ({e}), retrying ({attempt + 2}/3)...")
                    time.sleep(3)
                    session = create_session()
                else:
                    raise

        all_records.extend(records)
        print(f"Page {page_idx + 1}/{total_pages}: {len(records)} records (total: {len(all_records)})")

    # Add sequential counter
    for i, record in enumerate(all_records, 1):
        record['counter'] = i

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(all_records)} records saved to {output_file}")
    return all_records


if __name__ == '__main__':
    output = sys.argv[1] if len(sys.argv) > 1 else 'sebi_fpi_records.json'
    scrape_all(output)