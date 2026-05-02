#!/usr/bin/env python3
"""Check for and remove duplicates in ARIA members data"""

import json
import sys
from collections import defaultdict

def check_duplicates(filename="scrape_content_dump/aria_members.json"):
    with open(filename, 'r', encoding='utf-8') as f:
        members = json.load(f)
    
    print(f"Total members: {len(members)}")
    
    # Check for duplicates by member_id
    member_ids = defaultdict(list)
    for i, member in enumerate(members):
        member_id = member.get("member_id", "")
        member_ids[member_id].append(i)
    
    id_duplicates = {member_id: indices for member_id, indices in member_ids.items() if len(indices) > 1}
    
    if id_duplicates:
        print(f"\nFound {len(id_duplicates)} duplicate member IDs:")
        for member_id, indices in id_duplicates.items():
            print(f"  Member ID {member_id}: appears at positions {indices}")
            # Show the names for these duplicates
            names = [members[i].get("name", "") for i in indices]
            print(f"    Names: {names}")
    else:
        print("\nNo duplicates found by Member ID.")
    
    # Check for duplicates by name
    names = defaultdict(list)
    for i, member in enumerate(members):
        name = member.get("name", "")
        names[name].append(i)
    
    name_duplicates = {name: indices for name, indices in names.items() if len(indices) > 1}
    
    if name_duplicates:
        print(f"\nFound {len(name_duplicates)} duplicate names:")
        for name, indices in name_duplicates.items():
            print(f"  {name}: appears at positions {indices}")
            # Show member IDs for these duplicates
            member_ids_for_name = [members[i].get("member_id", "") for i in indices]
            print(f"    Member IDs: {member_ids_for_name}")
    else:
        print("\nNo duplicates found by Name.")
    
    # Check for duplicates by email
    emails = defaultdict(list)
    for i, member in enumerate(members):
        email = member.get("email", "")
        if email:  # Only check non-empty emails
            emails[email].append(i)
    
    email_duplicates = {email: indices for email, indices in emails.items() if len(indices) > 1}
    
    if email_duplicates:
        print(f"\nFound {len(email_duplicates)} duplicate emails:")
        for email, indices in email_duplicates.items():
            print(f"  {email}: appears at positions {indices}")
            names_for_email = [members[i].get("name", "") for i in indices]
            print(f"    Names: {names_for_email}")
    else:
        print("\nNo duplicates found by Email.")
    
    # Remove duplicates (keep first occurrence by member_id)
    unique_members = []
    seen_member_ids = set()
    
    for member in members:
        member_id = member.get("member_id", "")
        if member_id not in seen_member_ids:
            unique_members.append(member)
            seen_member_ids.add(member_id)
    
    print(f"\nAfter removing duplicates: {len(unique_members)} unique members")
    
    if len(unique_members) < len(members):
        # Save cleaned data
        output_file = filename.replace('.json', '_cleaned.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unique_members, f, indent=2, ensure_ascii=False)
        print(f"Cleaned data saved to: {output_file}")
    
    # Show some statistics
    print(f"\nData Statistics:")
    categories = defaultdict(int)
    has_website = 0
    has_phone = 0
    has_email = 0
    
    for member in unique_members:
        category = member.get("category", "Unknown")
        categories[category] += 1
        
        if member.get("website"):
            has_website += 1
        if member.get("phone"):
            has_phone += 1
        if member.get("email"):
            has_email += 1
    
    print(f"  Categories: {dict(categories)}")
    print(f"  Members with website: {has_website}")
    print(f"  Members with phone: {has_phone}")
    print(f"  Members with email: {has_email}")
    
    return unique_members

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "scrape_content_dump/aria_members.json"
    check_duplicates(filename)