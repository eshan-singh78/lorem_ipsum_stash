#!/usr/bin/env python3
"""
ARIA Members Directory Scraper
Scrapes Registered Investment Advisers data from ARIA website
"""

import requests
import json
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
from bs4 import BeautifulSoup
import html


class ARIAMembersScraper:
    """Scraper for ARIA members directory"""
    
    BASE_URL = "https://aria.org.in/wp-admin/admin-ajax.php"
    
    def __init__(self, output_dir: str = "scrape_content_dump"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self._setup_headers()
    
    def _setup_headers(self):
        """Setup request headers"""
        self.session.headers.update({
            'Accept': '*/*',
            'Accept-Language': 'en-GB,en;q=0.8',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Origin': 'https://aria.org.in',
            'Pragma': 'no-cache',
            'Referer': 'https://aria.org.in/members-directory/',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Mobile/15E148 Safari/604.1',
            'X-Requested-With': 'XMLHttpRequest'
        })
    
    def fetch_page(self, page: int = 1) -> Optional[Dict]:
        """
        Fetch a single page of members data
        
        Args:
            page: Page number (1-indexed)
        
        Returns:
            JSON response or None if failed
        """
        data = {
            'action': 'load_members_filter',
            'page': str(page),
            'member_category': '',
            'product_services': '',
            'sebi_reg_date': '',
            'full_name': '',
            'firm_name': '',
            'address': '',
            'city': '',
            'pincode': '',
            'country': 'IND',
            'state': ''
        }
        
        try:
            response = self.session.post(self.BASE_URL, data=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            return None
    
    def parse_member_card(self, card_html: str) -> Dict:
        """Parse a single member card HTML to extract member data"""
        soup = BeautifulSoup(card_html, 'html.parser')
        member = {}
        
        try:
            # Extract name
            name_elem = soup.find('h6', class_='itemtitle')
            if name_elem:
                member['name'] = name_elem.get_text(strip=True)
            
            # Extract member category
            category_elem = soup.find('div', class_='membercategory')
            if category_elem:
                member['category'] = category_elem.get_text(strip=True)
            
            # Extract member ID
            member_id_elem = soup.find('div', {'data-member-id': True})
            if member_id_elem:
                member['member_id'] = member_id_elem.get('data-member-id')
            
            # Extract details from list items
            list_items = soup.find_all('li', class_='member-listgroup-item')
            
            for item in list_items:
                icon = item.find('i')
                if not icon:
                    continue
                
                icon_class = ' '.join(icon.get('class', []))
                
                if 'bi-briefcase' in icon_class:
                    # Company/Firm name
                    title_elem = item.find('h6', class_='title')
                    if title_elem:
                        member['firm_name'] = title_elem.get_text(strip=True)
                
                elif 'bi-phone' in icon_class:
                    # Phone number
                    title_elem = item.find('h6', class_='title')
                    if title_elem:
                        member['phone'] = title_elem.get_text(strip=True)
                
                elif 'bi-envelope' in icon_class:
                    # Email
                    link_elem = item.find('a', class_='title-link')
                    if link_elem:
                        email = link_elem.get_text(strip=True)
                        member['email'] = email
                
                elif 'bi-globe2' in icon_class:
                    # Website
                    link_elem = item.find('a', class_='title-link')
                    if link_elem:
                        website = link_elem.get('href', '').strip()
                        member['website'] = website
            
        except Exception as e:
            print(f"Error parsing member card: {e}")
        
        return member
    
    def parse_members_html(self, html_content: str) -> List[Dict]:
        """Parse the members HTML content to extract all member data"""
        # Decode HTML entities
        html_content = html.unescape(html_content)
        
        soup = BeautifulSoup(html_content, 'html.parser')
        members = []
        
        # Find all member cards
        member_cards = soup.find_all('div', class_='card member-card')
        
        for card in member_cards:
            member_data = self.parse_member_card(str(card))
            if member_data:
                members.append(member_data)
        
        return members
    
    def get_total_pages(self, response_data: Dict) -> int:
        """Extract total pages from pagination HTML"""
        pagination_html = response_data.get('pagination', '')
        if not pagination_html:
            return 1
        
        # Decode HTML entities
        pagination_html = html.unescape(pagination_html)
        soup = BeautifulSoup(pagination_html, 'html.parser')
        
        # Find all page links
        page_links = soup.find_all('a', {'data-page': True})
        max_page = 1
        
        for link in page_links:
            try:
                page_num = int(link.get('data-page', '1'))
                max_page = max(max_page, page_num)
            except (ValueError, TypeError):
                continue
        
        return max_page
    
    def scrape_all(self, delay: float = 1.0, max_pages: Optional[int] = None) -> List[Dict]:
        """
        Scrape all pages of members data
        
        Args:
            delay: Delay between requests in seconds
            max_pages: Maximum number of pages to scrape (None for all)
        
        Returns:
            List of all member records
        """
        all_members = []
        page = 1
        total_pages = None
        
        print(f"Starting ARIA members scrape...")
        
        while True:
            if max_pages and page > max_pages:
                print(f"Reached max pages limit: {max_pages}")
                break
            
            print(f"Fetching page {page}...")
            response_data = self.fetch_page(page)
            
            if not response_data:
                print("Failed to fetch data, stopping")
                break
            
            # Get total pages from first response
            if total_pages is None:
                total_pages = self.get_total_pages(response_data)
                print(f"Total pages detected: {total_pages}")
            
            # Extract members from HTML
            members_html = response_data.get('members', '')
            if not members_html:
                print("No members HTML found")
                break
            
            members = self.parse_members_html(members_html)
            
            if not members:
                print("No members found on this page")
                break
            
            all_members.extend(members)
            print(f"  Retrieved {len(members)} members (total: {len(all_members)})")
            
            # Check if we've reached the last page
            if page >= total_pages:
                print("Reached last page")
                break
            
            page += 1
            
            # Be polite to the server
            time.sleep(delay)
        
        print(f"\nScraping complete! Total members: {len(all_members)}")
        return all_members
    
    def save_results(self, members: List[Dict], filename: str = "aria_members.json"):
        """Save scraped members to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(members, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")


def main():
    """Main execution"""
    scraper = ARIAMembersScraper()
    
    # Scrape all data
    members = scraper.scrape_all(
        delay=1.0,
        max_pages=None  # Set to a number to limit pages for testing
    )
    
    # Save results
    scraper.save_results(members)
    
    # Show sample data
    if members:
        print(f"\nSample member data:")
        sample_member = members[0]
        for key, value in sample_member.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()