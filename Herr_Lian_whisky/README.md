
# Using Machine Learning to Predict asset pricing in Alternative Investments



data_scraping
=============
Run the data_scraping file and you can get different data sources and sizes by modifying the variables selected_website and selected_type.
If selected_website is 'whiskyauction', we will get the auction history data from the website 'https://whiskyauction.com/'.
If selected_website is "whiskyhammer", it will get the auction history data of "https://www.whiskyhammer.com/".
When selected_type is 'test', only a small amount (one page) of data will be fetched. The rest of the cases will fetch global data.
The raw data file generated will be stored in the whisky_data.csv file in the run directory, while the processed data will be stored in processed_data.csv.

The crawl_website1 function:
-------------
This function serves the job of fetching data for 'https://whiskyauction.com/'.
Generates URLs for a whisky auction website to scrape past auction results.
Collects the relevant data by navigating through the URLs using requests and BeautifulSoup.
Processes the data in concurrent batches using multithreading.
Finally, saves the extracted data to a CSV file.
The data_transfer function:
Performs preprocessing and data cleaning tasks on a CSV file, such as:
Renaming columns and detecting dates.
Filling missing date columns based on other columns' information.
Handling missing values and dropping unnecessary columns.
Saving the processed data to a new CSV file.

The crawl_website2: function:
-------------
This function serves the job of fetching data for 'https://www.whiskyhammer.com/'.
Define the range of auction URLs and prepare the list.
Define the process_lot function to extract desired attributes from individual auction lot pages.
Parse the main auction list page to get individual lot URLs.
Process the lot URLs in batches using multithreading.
Store the scraped data in a CSV file.

The data_transfer: function:
-------------

Load existing data, skipping the first row, and save it back to the CSV with proper headers.
Use spacy to detect and format dates in the columns.
Fill missing auction dates.
Calculate age based on bottled and distilled dates if age is missing.
Fill missing bottled and distilled dates using auction date and age.
Drop rows where vital information is missing and handle other missing values.
Drop the 'Series' column and save the processed data to a CSV file.
Perform additional data cleaning.
Drop the 'Distilled Date' and 'Bottled Date' columns and save the final processed data to a new CSV file.
This is a comprehensive ETL (Extract, Transform, Load) process where data is scraped, cleaned, transformed, and finally saved in a structured manner.

XGBoost Regression Analysis
=============
This script conducts regression analysis using XGBoost.

#### Key Features
Data Preprocessing: Handles date columns, categorical data, etc.
Model Definition: A regression model based on XGBoost is defined.
Hyperparameter Search: Uses RepeatedKFold cross-validation combined with GridSearch for hyperparameter tuning.
Model Evaluation: Computes and outputs metrics like Mean Squared Error (MSE), Coefficient of Determination (R^2), and Root Mean Squared Error (RMSE).
Visualization Analysis: Includes scatter plots of actual vs predicted values, residual plots, feature importance plots, and more.
SHAP Value Analysis: Uses the SHAP library for model interpretability.
Dependencies

####  **Ensure you have the following Python libraries installed:**

**pandas
xgboost
scikit-learn
shap
matplotlib**

#### How to Run
Make sure your data files (like train_data.csv and test_data.csv) are in the same directory. Then, execute the script.



### End
