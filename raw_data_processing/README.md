# Raw Data Processing Pipeline

This folder contains the Jupyter notebooks used to collect and process the NYT terrorism news corpus (1851–2019). The notebooks should be reviewed in the order listed below.

## Pipeline Order

```
Step 1: NYT_WEB_SCRAPER.ipynb
           Scrape article metadata from nytimes.com/search
           → 1,478 CSVs in nyt_terrorism_1985-2019_news/

Step 2: scrape_information_of_news_under_category_of_terrorism_on_NYT_website.ipynb
           Consolidate 1,478 CSVs into one master list
           Filter by news_desk, clean URLs, remove duplicates
           Scrape full article text for post-1980 articles via Selenium
           → total_nyt_terrorism_news1850-2019.csv
           → nyt_online_texts.csv

Step 3: combine_all_terrorism_news_and_scrape_texts_1850_1980(ProQuestNYU).ipynb
           For pre-1981 articles: search ProQuest Historical NYT (NYU proxy),
           download PDFs, OCR with Tesseract at 500 DPI, spell-correct with TextBlob
           → nyt_terrorism_pdftexts_proquest2.csv

Step 4: 1850_1980_combine_test1.ipynb
           Merge article metadata with ProQuest OCR texts (pre-1981 only)
           → NYT_text_1851_1980.csv

Step 5: combine_all_terrorism_news_and_scrape_texts_1850_2019.ipynb
           Three-way merge: metadata + ProQuest texts + online texts
           → Full 1851–2019 corpus

Step 6: nyt_main_training.py (in code/ folder)
           Text cleaning, lemmatization, phrase detection
           → combined_processed_df.parquet
```

## Data Sources

- **nytimes.com/search**: Article metadata (title, date, abstract, link, author, news desk) for the search query "terrorism", 1851–2019
- **ProQuest Historical NYT** (via NYU library proxy): Full article text for pre-1981 articles, retrieved as PDFs and processed with Tesseract OCR at 500 DPI
- **nytimes.com article pages**: Full article text for post-1980 articles, scraped via Selenium and BeautifulSoup

## Key Methodological Notes

1. **Date windowing**: The web scraper uses adaptive window sizes to stay under the NYT search result display cap — full multi-year windows for sparse periods (pre-1900), monthly for 1969–2000, daily for September 11–30 2001, and 7-day windows for 2002–2019.

2. **News desk filter**: Only sections relevant to terrorism coverage were retained: ARCHIVES, U.S., 9/11 ANNIVERSARY, WORLD, POLITICS, AFRICA, EUROPE, NEW YORK, ASIA PACIFIC, MIDDLE EAST, AUSTRALIA, CANADA.

3. **Pre/post-1981 split**: Articles at index < 22855 in the master list correspond to 1851–1980 and were retrieved from ProQuest PDFs. Articles at index >= 22855 correspond to 1981–2019 and were scraped directly from nytimes.com.

4. **OCR post-processing**: TextBlob spell correction was applied to Tesseract OCR output to improve text quality for pre-1981 articles.

## Requirements

These notebooks require: selenium, beautifulsoup4, requests, pdf2image, pytesseract, opencv-python, textblob, Pillow, pandas, numpy. The ProQuest scraping additionally requires NYU library proxy access.

## Note

These notebooks were run interactively over multiple sessions and are provided as-is to document the data collection methodology. They cannot be re-run without the original environment (Selenium ChromeDriver, live NYT website access, NYU ProQuest proxy credentials, Tesseract-OCR installation).
