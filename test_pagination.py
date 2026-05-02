#!/usr/bin/env python3
"""Test pagination patterns with different doDirect values"""

from sebi_fpi_scraper import SEBIFPIScraper

def test_pagination():
    scraper = SEBIFPIScraper(intm_id=13)
    
    # Test different combinations with varying doDirect values
    test_cases = [
        (2, 0),  # Page 1
        (1, 1),  # Page 2
        (2, 1),  # Try page 3
        (1, 2),  # Try doDirect=2
        (2, 2),  # Try doDirect=2
        (3, 2),  # Try doDirect=2
        (1, 3),  # Try doDirect=3
        (2, 3),  # Try doDirect=3
        (3, 3),  # Try doDirect=3
        (4, 3),  # Try doDirect=3
        (1, 4),  # Try doDirect=4
        (2, 4),  # Try doDirect=4
    ]
    
    for next_val, do_direct in test_cases:
        print(f"\nTesting nextValue={next_val}, doDirect={do_direct}")
        html = scraper.fetch_page(next_value=next_val, do_direct=do_direct)
        if html:
            records, total_records, next_page = scraper.parse_html_response(html)
            if records:
                first_record = records[0].get("Name", "")
                reg_no = records[0].get("Registration No.", "")
                print(f"  First record: {first_record} ({reg_no})")
                print(f"  Total records on page: {len(records)}")
            else:
                print("  No records found")
        else:
            print("  Failed to fetch page")

test_pagination()