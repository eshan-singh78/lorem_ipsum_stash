#!/usr/bin/env python3
"""Manual scraper that uses the cleaned data and gets unique records only"""

import json
from sebi_fpi_scraper import SEBIFPIScraper

def scrape_unique_records():
    scraper = SEBIFPIScraper(intm_id=13)
    
    # Get page 1 (records 1-25)
    print("Fetching page 1...")
    html1 = scraper.fetch_page(next_value=2, do_direct=0)
    records1, total_records, _ = scraper.parse_html_response(html1)
    print(f"Page 1: {len(records1)} records")
    
    # Get page 2 (records 26-50)  
    print("Fetching page 2...")
    html2 = scraper.fetch_page(next_value=1, do_direct=1)
    records2, _, _ = scraper.parse_html_response(html2)
    print(f"Page 2: {len(records2)} records")
    
    # Combine and check for uniqueness
    all_records = records1 + records2
    
    # Remove duplicates by Registration No.
    unique_records = []
    seen_reg_nos = set()
    
    for record in all_records:
        reg_no = record.get("Registration No.", "")
        if reg_no not in seen_reg_nos:
            unique_records.append(record)
            seen_reg_nos.add(reg_no)
    
    print(f"\nTotal unique records: {len(unique_records)}")
    print(f"Total records (with duplicates): {len(all_records)}")
    
    # Save the unique records
    scraper.save_results(unique_records, filename="sebi_fpi_unique_50.json")
    
    # Show sample records
    print(f"\nFirst 3 unique records:")
    for i, record in enumerate(unique_records[:3], 1):
        print(f"{i}. {record.get('Name', '')} ({record.get('Registration No.', '')})")

if __name__ == "__main__":
    scrape_unique_records()