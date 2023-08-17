# Whisky Price Prediction

## Data Collection

Data scraping, post-processing and visualization codes and Intermediate run results are demonstrated in `data_scraping.ipynb` file. In jupyter notebook file, codes are splited in below parts including `1. Data Scraping`, `2. Post Process`and `3. Visualization`.

### 1 Data Scraping

Because of large scale data need to be scrapped from different subpages in one websites, we run python code with multi-processing and multi-threading speeding skills. In notebook file, running codes in command are shown, such as `python3 crawl_auction.py --run crawl --st 0 --ed 10 &`. Data are scrapped from Two websites(https://whiskyauction.com/,https://www.scotchwhiskyauctions.com/), with `--run` indicating running mode, `--st, --ed` standing for starting and ending indexes for multi-processing running. For crawling all current data, `--ed 150` and `--ed 490000` are enough for two source website, respectively.

## 2. Post Processing

These codes are used for processing data to clean version for model training, prediction and evaluation. Cells in the part 2.1 and 2.2 process data from [website1](https://www.scotchwhiskyauctions.com/) and cells in the part 2.3 for [website2](https://whiskyauction.com/). The part 2.4 further remove useless columns and change feature type to convert processed csv data into model-friendly format. In order to reproduce outputs and generate dataset for model training, running notebook cells step by step is sufficient.

## 3. Visualization

In this part, we remove nan values and only keep 5%~95% percentile datas for different feature based on dataframe before post process stage2.4. Hist plot and density plot are utilized for observing data distribution for continuous numerical features, discrete category features and regression prediction  target `Winning bid`. Furthermore, the history timing information `his_bid_times` and `his_bid_prices` are auction years(2010-2023) and history auction prices(with simiar range to `Winning bid`).



# Modelling

Price prediction regression modelling codes are show in `model.ipynb`. The part1 `1. ML Models` and part2 `DNN Models` are separately used for traditional machine learning models(such as LR, Lasso, SVM, Decision Tree, Random Forest and GBDT) and deep learning models(such as LSTM&DNN, CNN&DNN and pure DNN) with timing information extraction. For each part, data load and scale(standard scale, minmax scale or orginal format) are utilized in preprocessing the part1.1 and 2.1. The prefix 'LSTM&' and 'CNN&' stand for model types chosen for modelling historical auction time prices for time series information and the suffix 'DNN' indicates the MLP network for normal numerical features. Model training, prediction and evaluation are shown in the part1.2 and 2.2. Furthermore, the comparison experiments can be reproduce using cell codes in the parts1.2 and 2.2 for different scale methods, datasets with distribution gap and  variant models both suitable for regression task. The deep learning model structures and parameters scale can be found in the part2.3.