#!/usr/bin/env python3
"""
SEBI FPI Intermediary Scraper
Scrapes Foreign Portfolio Investor intermediary data from SEBI website
"""

import requests
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from bs4 import BeautifulSoup
import re


class SEBIFPIScraper:
    """Scraper for SEBI FPI intermediary information"""
    
    BASE_URL = "https://www.sebi.gov.in/sebiweb/ajax/other/getintmfpiinfo.jsp"
    
    def __init__(self, intm_id: int = 13, output_dir: str = "scrape_content_dump"):
        self.intm_id = intm_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self._setup_headers()
        self._initialize_session()
    
    def _initialize_session(self):
        """Initialize session by visiting the main page"""
        try:
            main_url = f'https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId={self.intm_id}'
            response = self.session.get(main_url)
            response.raise_for_status()
            print(f"Session initialized with cookies: {list(self.session.cookies.keys())}")
        except Exception as e:
            print(f"Warning: Failed to initialize session: {e}")
    
    def _setup_headers(self):
        """Setup request headers"""
        self.session.headers.update({
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-type': 'application/x-www-form-urlencoded',
            'Origin': 'https://www.sebi.gov.in',
            'Pragma': 'no-cache',
            'Referer': f'https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes&intmId={self.intm_id}',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1'
        })
    
    def fetch_page(self, next_value: int = 0, direction: str = 'n', do_direct: int = 0) -> Optional[str]:
        """
        Fetch a single page of data
        
        Args:
            next_value: Pagination page number
            direction: 'n' for next, 'p' for previous
            do_direct: 0 for first page, 1 for subsequent pages
        
        Returns:
            HTML response or None if failed
        """
        data = {
            'nextValue': str(next_value),
            'next': direction,
            'intmId': str(self.intm_id),
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
            'doDirect': str(do_direct),
            'intmIds': ''
        }
        
        try:
            response = self.session.post(self.BASE_URL, data=data)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching page at offset {next_value}: {e}")
            return None
    
    def parse_html_response(self, html: str) -> tuple[List[Dict], Optional[int], Optional[int]]:
        """
        Parse HTML response to extract records and pagination info
        
        Returns:
            (records, total_records, next_page)
        """
        soup = BeautifulSoup(html, 'html.parser')
        records = []
        
        # Extract pagination info
        total_records = None
        next_page = None
        current_range = None
        
        # Look for "X to Y of Z records" pattern
        pagination_text = soup.find('p')
        if pagination_text:
            match = re.search(r'(\d+)\s+to\s+(\d+)\s+of\s+(\d+)\s+records', pagination_text.text)
            if match:
                start_rec = int(match.group(1))
                end_rec = int(match.group(2))
                total_records = int(match.group(3))
                current_range = f"{start_rec}-{end_rec}"
        
        # Find the active page and next page link
        # Look for pagination links like: <a href="javascript: searchFormFpi('n', '1');">2</a>
        pagination_div = soup.find('div', class_='pagination_outer')
        if pagination_div:
            # Find the active page
            active_link = pagination_div.find('a', class_='active')
            if active_link:
                current_page_num = active_link.get_text(strip=True)
                print(f"  Current page number: {current_page_num}")
                
                # Find all page links
                all_links = pagination_div.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href', '')
                    # Look for the next page after active
                    # Pattern: javascript: searchFormFpi('n', '1');
                    match = re.search(r"searchFormFpi\('n',\s*'(-?\d+)'\)", href)
                    if match:
                        page_value = int(match.group(1))
                        # The link text shows the page number
                        link_text = link.get_text(strip=True)
                        if link_text.isdigit() and int(link_text) == int(current_page_num) + 1:
                            next_page = page_value
                            break
        
        print(f"  Page shows records: {current_range}, next page value: {next_page}")
        
        # Find all card-table containers (each contains one record)
        card_tables = soup.find_all('div', class_='fixed-table-body card-table')
        
        for card_table in card_tables:
            record = {}
            # Find all card-view divs within this card-table
            card_views = card_table.find_all('div', class_='card-view')
            
            for card_view in card_views:
                # Extract title (field name)
                title_div = card_view.find('div', class_='title')
                value_div = card_view.find('div', class_='value')
                
                if title_div and value_div:
                    field_name = title_div.get_text(strip=True)
                    field_value = value_div.get_text(strip=True)
                    
                    # Decode HTML entities (e.g., &#64; -> @, &#46; -> .)
                    import html
                    field_value = html.unescape(field_value)
                    
                    record[field_name] = field_value
            
            if record:  # Only add non-empty records
                records.append(record)
        
        return records, total_records, next_page
    
    def scrape_all(self, start_page: int = 0, delay: float = 1.0, 
                   max_pages: Optional[int] = None) -> List[Dict]:
        """
        Scrape all pages of data
        
        Args:
            start_page: Starting page number (0-indexed)
            delay: Delay between requests in seconds
            max_pages: Maximum number of pages to scrape (None for all)
        
        Returns:
            List of all records
        """
        all_records = []
        page_count = 0
        
        print(f"Starting scrape for intmId={self.intm_id}")
        
        while True:
            if max_pages and page_count >= max_pages:
                print(f"Reached max pages limit: {max_pages}")
                break
            
            # Correct pagination pattern: doDirect controls the page
            do_direct = page_count  # 0 for page 1, 1 for page 2, etc.
            next_value = 1  # Can be any value, doesn't matter
            
            print(f"Fetching page {page_count + 1} (nextValue={next_value}, doDirect={do_direct})...")
            html = self.fetch_page(next_value=next_value, do_direct=do_direct)
            
            if not html:
                print("Failed to fetch data, stopping")
                break
            
            # Parse HTML response
            records, total_records, next_page = self.parse_html_response(html)
            
            if not records:
                print("No more records found")
                break
            
            # Check for duplicates by comparing with last batch
            if all_records and len(all_records) >= 25:
                # Compare first record of new batch with records we already have
                first_new_record = records[0] if records else {}
                first_new_reg_no = first_new_record.get("Registration No.", "")
                
                # Check if this registration number already exists
                existing_reg_nos = {r.get("Registration No.", "") for r in all_records}
                if first_new_reg_no in existing_reg_nos:
                    print(f"  Detected duplicate records (first record: {first_new_reg_no}), stopping")
                    break
            
            all_records.extend(records)
            print(f"  Retrieved {len(records)} records (total: {len(all_records)})")
            
            if total_records:
                print(f"  Progress: {len(all_records)}/{total_records}")
                
                # Stop if we've got all records
                if len(all_records) >= total_records:
                    print("Reached end of data")
                    break
            
            page_count += 1
            
            # Be polite to the server
            time.sleep(delay)
        
        print(f"\nScraping complete! Total records: {len(all_records)}")
        return all_records
    
    def _extract_records(self, data: Dict) -> List[Dict]:
        """Extract records from API response - deprecated, using HTML parsing now"""
        return []
    
    def _has_more_pages(self, data: Dict) -> bool:
        """Check if there are more pages to fetch - deprecated, using HTML parsing now"""
        return False
    
    def save_results(self, records: List[Dict], filename: str = "sebi_fpi_records.json"):
        """Save scraped records to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")


def main():
    """Main execution"""
    scraper = SEBIFPIScraper(intm_id=13)
    
    # Scrape all data
    records = scraper.scrape_all(
        start_page=0,
        delay=1.0,
        max_pages=None  # Set to a number to limit pages for testing
    )
    
    # Save results
    scraper.save_results(records)


if __name__ == "__main__":
    main()
