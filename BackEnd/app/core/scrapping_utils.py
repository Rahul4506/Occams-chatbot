"""Web scraping utilities using Playwright - Enhanced for complete navigation coverage."""

import asyncio
import json
import logging
from typing import List, Dict, Set
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright, Page
from bs4 import BeautifulSoup
import time
from pathlib import Path

from ..config import TARGET_WEBSITE_URL, DATA_DIR, SCRAPING_DELAY, MAX_PAGES_TO_SCRAPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OccamsWebScraper:
    """Enhanced web scraper for Occam's Advisory website using Playwright."""
    
    def __init__(self, base_url: str = TARGET_WEBSITE_URL):
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.scraped_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        self.navigation_links: Set[str] = set()  # Store navigation links separately
        
    async def scrape_website(self) -> List[Dict]:
        """Scrape the entire website with enhanced navigation coverage."""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            
            try:
                # First, extract all navigation links from homepage
                logger.info("Extracting navigation structure...")
                await page.goto(self.base_url, wait_until="networkidle")
                await page.wait_for_timeout(3000)  # Wait for dynamic content
                
                # Extract navigation links first
                nav_links = await self._extract_navigation_links(page)
                logger.info(f"Found {len(nav_links)} navigation links: {nav_links}")
                
                # Scrape homepage first
                await self._scrape_single_page(page, self.base_url)
                
                # Then scrape all navigation pages and their subsections
                for nav_url in nav_links:
                    if nav_url not in self.scraped_urls:
                        await asyncio.sleep(SCRAPING_DELAY)
                        await self._scrape_page_with_subsections(page, nav_url)
                
                # Finally, do a recursive scrape for any remaining internal links
                await self._scrape_remaining_pages(page)
                
                # Save scraped data
                await self._save_scraped_data()
                
                logger.info(f"Scraping completed. Total pages scraped: {len(self.scraped_data)}")
                return self.scraped_data
                
            except Exception as e:
                logger.error(f"Error during scraping: {str(e)}")
                raise
            finally:
                await browser.close()
    
    async def _extract_navigation_links(self, page: Page) -> List[str]:
        """Extract all navigation links including main nav and dropdowns."""
        nav_links = []
        
        try:
            # Common navigation selectors
            nav_selectors = [
                'nav a[href]',
                '.navigation a[href]',
                '.nav a[href]',
                '.menu a[href]',
                '.navbar a[href]',
                'header a[href]',
                '.main-nav a[href]',
                '.primary-nav a[href]'
            ]
            
            # Try to interact with dropdown menus first
            dropdown_triggers = await page.query_selector_all('nav .dropdown, nav .has-dropdown, .menu-item-has-children')
            for trigger in dropdown_triggers:
                try:
                    await trigger.hover()
                    await page.wait_for_timeout(500)  # Wait for dropdown to appear
                except:
                    pass
            
            # Extract all navigation links
            for selector in nav_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        href = await element.get_attribute('href')
                        if href:
                            full_url = urljoin(self.base_url, href)
                            if self._is_valid_url(full_url) and self._is_main_section(href):
                                nav_links.append(full_url)
                                logger.info(f"Found navigation link: {full_url}")
                except Exception as e:
                    logger.debug(f"Error with selector {selector}: {e}")
                    continue
            
            # Also look for specific section keywords in link text
            all_links = await page.query_selector_all('a[href]')
            for element in all_links:
                try:
                    link_text = (await element.inner_text()).lower().strip()
                    href = await element.get_attribute('href')
                    
                    # Check for main section keywords
                    section_keywords = ['about', 'services', 'team', 'resources', 'contact', 'portfolio', 'blog']
                    if any(keyword in link_text for keyword in section_keywords) and href:
                        full_url = urljoin(self.base_url, href)
                        if self._is_valid_url(full_url):
                            nav_links.append(full_url)
                            logger.info(f"Found section link by text '{link_text}': {full_url}")
                except:
                    continue
            
            # Remove duplicates while preserving order
            unique_links = list(dict.fromkeys(nav_links))
            return unique_links
            
        except Exception as e:
            logger.error(f"Error extracting navigation links: {str(e)}")
            return []
    
    def _is_main_section(self, href: str) -> bool:
        """Check if link appears to be a main section."""
        href_lower = href.lower()
        main_sections = [
            'about', 'services', 'team', 'resources', 'contact', 
            'portfolio', 'blog', 'news', 'careers', 'clients'
        ]
        
        # Check if href contains main section keywords
        return any(section in href_lower for section in main_sections)
    
    async def _scrape_page_with_subsections(self, page: Page, url: str):
        """Scrape a page and all its subsections."""
        if url in self.scraped_urls or len(self.scraped_data) >= MAX_PAGES_TO_SCRAPE:
            return
        
        try:
            logger.info(f"Scraping main section: {url}")
            
            # Scrape the main page
            await self._scrape_single_page(page, url)
            
            # Look for subsections on this page
            subsection_links = await self._extract_subsection_links(page, url)
            
            for subsection_url in subsection_links[:5]:  # Limit subsections per main section
                if subsection_url not in self.scraped_urls:
                    await asyncio.sleep(SCRAPING_DELAY)
                    await self._scrape_single_page(page, subsection_url)
                    
        except Exception as e:
            logger.error(f"Error scraping section {url}: {str(e)}")
    
    async def _extract_subsection_links(self, page: Page, parent_url: str) -> List[str]:
        """Extract subsection links from the current page."""
        subsection_links = []
        
        try:
            # Look for links that seem to be subsections of the current page
            all_links = await page.query_selector_all('a[href]')
            
            for element in all_links:
                href = await element.get_attribute('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    
                    # Check if this could be a subsection
                    if (self._is_valid_url(full_url) and 
                        self._is_likely_subsection(parent_url, full_url)):
                        subsection_links.append(full_url)
            
            return list(dict.fromkeys(subsection_links))
            
        except Exception as e:
            logger.error(f"Error extracting subsections from {parent_url}: {str(e)}")
            return []
    
    def _is_likely_subsection(self, parent_url: str, child_url: str) -> bool:
        """Check if child_url is likely a subsection of parent_url."""
        parent_path = urlparse(parent_url).path.strip('/')
        child_path = urlparse(child_url).path.strip('/')
        
        # If child path starts with parent path, it's likely a subsection
        return child_path.startswith(parent_path) and child_path != parent_path
    
    async def _scrape_single_page(self, page: Page, url: str):
        """Scrape a single page."""
        if url in self.scraped_urls:
            return
        
        try:
            logger.info(f"Scraping page: {url}")
            
            response = await page.goto(url, wait_until="networkidle")
            
            if response.status != 200:
                logger.warning(f"Failed to load {url}: HTTP {response.status}")
                return
            
            # Wait for content to load
            await page.wait_for_timeout(2000)
            
            # Handle dynamic content loading
            try:
                await page.wait_for_load_state("domcontentloaded")
            except:
                pass
            
            content = await page.content()
            page_data = await self._extract_page_data(page, url, content)
            
            if page_data and page_data['content'].strip():
                self.scraped_data.append(page_data)
                self.scraped_urls.add(url)
                logger.info(f"Successfully scraped: {url}")
            else:
                logger.warning(f"No content extracted from: {url}")
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
    
    async def _scrape_remaining_pages(self, page: Page):
        """Scrape any remaining internal links not already covered."""
        if len(self.scraped_data) >= MAX_PAGES_TO_SCRAPE:
            return
        
        # Get all unique internal links from already scraped pages
        all_internal_links = set()
        
        for scraped_page in self.scraped_data:
            try:
                await page.goto(scraped_page['url'], wait_until="networkidle")
                page_links = await self._extract_internal_links(page)
                all_internal_links.update(page_links)
            except:
                continue
        
        # Scrape unscraped internal links
        remaining_links = all_internal_links - self.scraped_urls
        
        for link in list(remaining_links)[:10]:  # Limit additional pages
            if len(self.scraped_data) >= MAX_PAGES_TO_SCRAPE:
                break
            await asyncio.sleep(SCRAPING_DELAY)
            await self._scrape_single_page(page, link)
    
    async def _extract_page_data(self, page: Page, url: str, html_content: str) -> Dict:
        """Extract structured data from a page."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", ".advertisement"]):
            element.decompose()
        
        # Extract title
        title = ""
        try:
            title_element = await page.query_selector('h1, title')
            if title_element:
                title = await title_element.inner_text()
            elif soup.title:
                title = soup.title.string
        except:
            pass
        
        # Extract main content with multiple strategies
        content = ""
        content_selectors = [
            'main', 'article', '.content', '#content', '.main-content', 
            '.page-content', '.entry-content', '.post-content', '[role="main"]'
        ]
        
        for selector in content_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    content = await element.inner_text()
                    if content.strip():
                        break
            except:
                continue
        
        # Fallback to body text if no main content found
        if not content.strip():
            try:
                body_element = await page.query_selector('body')
                if body_element:
                    content = await body_element.inner_text()
                else:
                    content = soup.get_text()
            except:
                content = soup.get_text()
        
        # Clean up content
        content = self._clean_text(content)
        
        # Extract meta description
        meta_desc = ""
        meta_element = soup.find('meta', attrs={'name': 'description'})
        if meta_element:
            meta_desc = meta_element.get('content', '')
        
        # Extract headings for better structure
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if heading.get_text().strip():
                headings.append(heading.get_text().strip())
        
        return {
            'url': url,
            'title': title.strip(),
            'content': content,
            'headings': headings,
            'meta_description': meta_desc,
            'scraped_at': time.time(),
            'word_count': len(content.split()) if content else 0
        }
    
    async def _extract_internal_links(self, page: Page) -> List[str]:
        """Extract internal links from the current page."""
        links = []
        try:
            anchor_elements = await page.query_selector_all('a[href]')
            
            for element in anchor_elements:
                href = await element.get_attribute('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    if self._is_valid_url(full_url):
                        links.append(full_url)
            
            return list(dict.fromkeys(links))
            
        except Exception as e:
            logger.error(f"Error extracting links: {str(e)}")
            return []
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for scraping."""
        parsed = urlparse(url)
        
        # Must be same domain
        if parsed.netloc != self.domain:
            return False
        
        # Skip unwanted file types
        unwanted_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.ico', '.xml', '.zip'}
        if any(url.lower().endswith(ext) for ext in unwanted_extensions):
            return False
        
        # Skip unwanted paths
        unwanted_patterns = {'#', 'mailto:', 'tel:', 'javascript:', 'login', 'admin', 'wp-admin'}
        if any(pattern in url.lower() for pattern in unwanted_patterns):
            return False
        
        return True
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Split into lines and clean each
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip very short lines and navigation items
            if line and len(line) > 3 and not line.lower() in ['home', 'menu', 'skip']:
                cleaned_lines.append(line)
        
        # Join lines and clean up extra whitespace
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove multiple consecutive newlines
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text.strip()
    
    async def _save_scraped_data(self):
        """Save scraped data to JSON file with enhanced metadata."""
        output_file = DATA_DIR / "scraped_data.json"
        
        # Add summary information
        summary = {
            'scraping_summary': {
                'total_pages': len(self.scraped_data),
                'total_urls_found': len(self.scraped_urls),
                'scraping_timestamp': time.time(),
                'base_url': self.base_url
            },
            'scraped_pages': self.scraped_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Scraped data saved to {output_file}")
        
        # Also save a simple text summary
        summary_file = DATA_DIR / "scraping_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Web Scraping Summary\n")
            f.write(f"===================\n\n")
            f.write(f"Base URL: {self.base_url}\n")
            f.write(f"Total pages scraped: {len(self.scraped_data)}\n")
            f.write(f"Scraped URLs:\n")
            for i, data in enumerate(self.scraped_data, 1):
                f.write(f"{i}. {data['url']} - {data.get('title', 'No title')}\n")


async def scrape_occams_website() -> List[Dict]:
    """Main function to scrape Occam's Advisory website with complete coverage."""
    scraper = OccamsWebScraper()
    return await scraper.scrape_website()


if __name__ == "__main__":
    # Run scraping
    asyncio.run(scrape_occams_website())    