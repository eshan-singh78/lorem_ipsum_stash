#!/usr/bin/env python3
"""Test the updated SEBI scraper with correct pagination"""

from sebi_fpi_scraper import SEBIFPIScraper

def main():
    scraper = SEBIFPIScraper(intm_id=13)
    
    # Test with first 5 pages to verify pagination works
    records = scraper.scrape_all(
        start_page=0,
        delay=1.0,
        max_pages=5
    )
    
    print(f"\n{'='*60}")
    print(f"Total records scraped: {len(records)}")
    print(f"{'='*60}")
    
    # Show sample records from different pages
    sample_indices = [0, 25, 50, 75, 100]  # First record from each page
    
    for i, idx in enumerate(sample_indices, 1):
        if idx < len(records):
            record = records[idx]
            name = record.get("Name", "")
            reg_no = record.get("Registration No.", "")
            print(f"Page {i} first record: {name} ({reg_no})")
    
    scraper.save_results(records, filename="sebi_fpi_test_5pages.json")
    print(f"\n✓ Saved {len(records)} records to sebi_fpi_test_5pages.json")

if __name__ == "__main__":
    main()
