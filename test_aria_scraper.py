#!/usr/bin/env python3
"""Test the ARIA scraper with limited pages"""

from aria_members_scraper import ARIAMembersScraper

def main():
    scraper = ARIAMembersScraper()
    
    # Test with first 3 pages
    members = scraper.scrape_all(
        delay=1.0,
        max_pages=3
    )
    
    print(f"\n{'='*60}")
    print(f"Total members scraped: {len(members)}")
    print(f"{'='*60}")
    
    # Show detailed sample records
    for i, member in enumerate(members[:3], 1):
        print(f"\nMember {i}:")
        for key, value in member.items():
            print(f"  {key}: {value}")
    
    scraper.save_results(members, filename="aria_members_test.json")
    print(f"\n✓ Saved {len(members)} members to aria_members_test.json")

if __name__ == "__main__":
    main()