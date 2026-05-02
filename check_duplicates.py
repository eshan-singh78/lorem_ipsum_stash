#!/usr/bin/env python3
"""Check for and remove duplicates in scraped FPI data"""

import json
import sys
from collections import defaultdict

def check_duplicates(filename="scrape_content_dump/sebi_fpi_records.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        records = json.load(f)
    
    print(f"Total records: {len(records)}")
    
    # Check for duplicates by Registration No.
    reg_nos = defaultdict(list)
    for i, record in enumerate(records):
        reg_no = record.get("Registration No.", "")
        reg_nos[reg_no].append(i)
    
    duplicates = {reg_no: indices for reg_no, indices in reg_nos.items() if len(indices) > 1}
    
    if duplicates:
        print(f"\nFound {len(duplicates)} duplicate registration numbers:")
        for reg_no, indices in duplicates.items():
            print(f"  {reg_no}: appears at positions {indices}")
            # Show the names for these duplicates
            names = [records[i].get("Name", "") for i in indices]
            print(f"    Names: {names}")
    else:
        print("\nNo duplicates found by Registration No.")
    
    # Check for duplicates by Name
    names = defaultdict(list)
    for i, record in enumerate(records):
        name = record.get("Name", "")
        names[name].append(i)
    
    name_duplicates = {name: indices for name, indices in names.items() if len(indices) > 1}
    
    if name_duplicates:
        print(f"\nFound {len(name_duplicates)} duplicate names:")
        for name, indices in name_duplicates.items():
            print(f"  {name}: appears at positions {indices}")
    else:
        print("\nNo duplicates found by Name.")
    
    # Remove duplicates (keep first occurrence)
    unique_records = []
    seen_reg_nos = set()
    
    for record in records:
        reg_no = record.get("Registration No.", "")
        if reg_no not in seen_reg_nos:
            unique_records.append(record)
            seen_reg_nos.add(reg_no)
    
    print(f"\nAfter removing duplicates: {len(unique_records)} unique records")
    
    if len(unique_records) < len(records):
        # Save cleaned data
        output_file = filename.replace('.json', '_cleaned.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_records, f, indent=2, ensure_ascii=False)
        print(f"Cleaned data saved to: {output_file}")
    
    return unique_records

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "scrape_content_dump/sebi_fpi_records.json"
    check_duplicates(filename)