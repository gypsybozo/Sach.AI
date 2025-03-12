from googlesearch import search
from newspaper import Article
import json
import time
import nltk
# nltk.download('punkt_tab')
import tldextract

def google_search_links(query, num_results=3):
    """Fetch top Google search links."""
    try:
        links = search(query + " site:news", num_results=num_results, lang="en")
        return list(links)
    except Exception as e:
        print(f"Error: {e}")
        return []

def scrape_article(url):
    """Scrape article content and metadata using Newspaper3k."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()  # Enables keyword and summary extraction

        # Extract the media source (domain name)
        extracted = tldextract.extract(url)
        media_source = f"{extracted.domain}.{extracted.suffix}"

        return {
            "url": url,
            "media_source": media_source,  # Add media source
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date.strftime('%Y-%m-%d') if article.publish_date else None,
            "top_image": article.top_image,
            "keywords": article.keywords,
            "summary": article.summary,
            "content": article.text
        }
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return None

def main():
    # query = input("Enter the event to search: ")  # User enters a news event
    news_links = google_search_links("koodalmanikyam temple")

    articles = []
    for url in news_links:
        print(f"Scraping: {url}")
        article_data = scrape_article(url)
        if article_data:
            articles.append(article_data)
        time.sleep(2)  # Delay to avoid rate limits

    with open("news_articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=4)

    print("Scraping complete! Articles saved to news_articles.json")

if __name__ == "__main__":
    main()
