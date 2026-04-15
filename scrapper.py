import requests
from bs4 import BeautifulSoup as bs
import csv
import time
import random


def scrape_page(url, session, headers, retries=3):
    """Download HTML content with retry + delay + session"""

    for attempt in range(retries):
        try:
            response = session.get(url, headers=headers)

            if response.status_code == 200:
                # Random delay to mimic human behavior
                time.sleep(random.uniform(1.5, 4))
                return response.text

            elif response.status_code > 500:
                if "To discuss automated access to Amazon data please contact" in response.text:
                    print(f"Blocked by Amazon: {url}")
                else:
                    print(f"Error {response.status_code} for {url}")

            # Exponential backoff
            sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
            print(f"Retrying in {sleep_time:.2f} sec...")
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Request failed for {url}: {e}")
            time.sleep(2 ** attempt)

    return None


def extract_product_links(search_urls, session, headers, base_url):
    """Extract product review links from search pages"""
    extracted_links = []

    for url in search_urls:
        print(f"Scraping search page: {url}")
        page = scrape_page(url, session, headers)

        if not page:
            continue

        soup = bs(page, "html.parser")
        a_tags = soup.find_all("a", class_="a-link-normal")

        for tag in a_tags:
            link = tag.get("href")
            if link:
                extracted_links.append(link)

    # Filter review links
    product_links = [link for link in extracted_links if "#customerReviews" in link]

    # Convert to full URLs
    full_links = [
        link if link.startswith("https:") else base_url + link
        for link in product_links
    ]

    return full_links


def extract_reviews(product_links, session, headers, max_reviews=200):
    """Extract reviews from product pages"""
    reviews = []

    for url in product_links:
        if len(reviews) >= max_reviews:
            break

        try:
            print(f"Scraping reviews: {url}")
            page = scrape_page(url, session, headers)

            if not page:
                continue

            soup = bs(page, "html.parser")
            div_reviews = soup.find_all('div', {'class': 'a-section celwidget'})

            for review_div in div_reviews:
                if len(reviews) >= max_reviews:
                    break

                #Rating extraction (FIXED)
                star = review_div.find('i', {'data-hook': 'review-star-rating'})
                star_rating = 'N/A'

                if star:
                    raw_rating = star.find('span', class_='a-icon-alt').get_text(strip=True)
                    try:
                        star_rating = int(float(raw_rating.split()[0]))
                    except:
                        star_rating = 'N/A'

                #Review text
                body = review_div.find('span', {'data-hook': 'review-body'})
                review_text = body.get_text(strip=True) if body else 'N/A'

                if all(x != 'N/A' for x in [star_rating, review_text]):
                    reviews.append((star_rating, review_text))

        except Exception as e:
            print(f"Error processing {url}: {e}")

    return reviews


def save_to_csv(reviews, output_file):
    """Save reviews to CSV"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Star Rating', 'Review Description'])
        writer.writerows(reviews)

    print(f"Saved {len(reviews)} reviews to {output_file}")


def run_scraper(search_urls, output_file="amazon_reviews.csv", max_reviews=200):
    """Main pipeline function"""

    headers = {
        'authority': 'www.amazon.com',
        'pragma': 'no-cache',
        'cache-control': 'no-cache',
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'sec-fetch-site': 'none',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-dest': 'document',
        'accept-language': 'en-US,en;q=0.9',
    }

    base_url = "https://www.amazon.in"

    #Session handling
    session = requests.Session()
    session.headers.update(headers)

    product_links = extract_product_links(search_urls, session, headers, base_url)
    reviews = extract_reviews(product_links, session, headers, max_reviews)

    save_to_csv(reviews, output_file)

    return reviews