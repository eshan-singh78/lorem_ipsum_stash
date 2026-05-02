#!/usr/bin/env python3
"""Debug script to test API pagination"""

import requests
from bs4 import BeautifulSoup
import re

def test_with_session():
    session = requests.Session()
    url = "https://www.sebi.gov.in/sebiweb/ajax/other/getintmfpiinfo.jsp"
    
    headers = {
        'Accept': '*/*',
        'Content-type': 'application/x-www-form-urlencoded',
        'Origin': 'https://www.sebi.gov.in',
        'Referer': 'https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId=13',
        'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1'
    }
    
    # First, visit the main page to get a session
    print("Step 1: Getting session from main page...")
    main_page = session.get('https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId=13')
    print(f"  Session cookies: {session.cookies.get_dict()}")
    
    def test_page(next_value, next_direction='n'):
        data = {
            'nextValue': str(next_value),
            'next': next_direction,
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
            'intmIds': ''
        }
        
        response = session.post(url, headers=headers, data=data)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Get pagination info
        pagination_text = soup.find('p')
        if pagination_text:
            print(f"nextValue={next_value}, next={next_direction} -> {pagination_text.get_text(strip=True)}")
        
        # Get first record name
        first_card = soup.find('div', class_='fixed-table-body card-table')
        if first_card:
            name_div = first_card.find('div', class_='value varun-text')
            if name_div:
                print(f"  First record: {name_div.get_text(strip=True)}")
    
    print("\nStep 2: Testing pagination with session:\n")
    test_page(0, 'n')
    test_page(1, 'n')
    test_page(2, 'n')

test_with_session()
