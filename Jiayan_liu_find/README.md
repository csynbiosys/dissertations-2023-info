# Jiayan_Liu_find

Running Jupyter notebooks and the script: orgs_extraction.ipynb

Final output is stored in data/org_info_final.csv

## Installation
1. Download the Crunchbase dataset.

2. The evaluation dataset is from Crunchbase, then you need to use the dataset path to replace '/Users/northarbour/Downloads/bulk_export/organizations.csv'

## Running
1. Open the Jupyter Notebook
2. Run the org_extraction.ipynb

## Configuration Options
The configuration options are all in configuration.py, including the specified number of crawls per source.

## Data Source
<pre>
```
data/
│
├── attribute # Including extracted product infomation
│
├── news_company # Including extracted orgs from news
│   ├── links.csv # News links
│   ├── articles.csv # News articles from links
│   └── info.csv # Orgs from News articles
│
├── website_company # Including extracted orgs from structured websites
│   └── info.csv # Orgs from Websites
│
├── org_info.csv # Extracted orgs info
│ 
├── org_info_searched.csv # Extracted orgs info with additional info
│ 
├── org_info_wait.csv # Org info wait for next turn
│ 
└── org_info_final.csv # The final org info to store

```
</pre>

## File Structure
<pre>
```
project/
│
├── attribute_scrape # Extract product infomation
│   ├── scraper.py # Extract product links
│   └── extract_content.py # Extract product info from links
│
├── company_search # Search additional orgs info through identifier
│   └── searcher.py # Extract orgs info
│
├── news_company_scrape # Extracted orgs from news
│   └── news_ner_spacy.py # Extract orgs entities from text
│
├── website_comapny_scrape # Extracted orgs from structured websites
│ 
├── configuration.py # Including scraper configuration
│ 
├── data_processing.py # Solve different sources collision
│ 
├── orgs_extraction.ipynb # Script that extract and evaluate the whole dataset
│ 
├── evaluation.py # Including evaluation methods
│ 
├── text_similarity_model.py # Provide text similarity model and similarity calculation method
│ 
├── org_submit.py # Submit qualified data
│ 
├── competitor_ranking.py # Search and rank competitors for orgs
│ 
└── dynamic_crawl.py # An example of dynamically crawling datasets
```
</pre>

