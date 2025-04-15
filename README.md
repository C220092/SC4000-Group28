# SC4000: Machine Learning - Elo Merchant Category Recommendation

## Project Overview
This repository contains the code for the [Elo Merchant Category Recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation) competition from Kaggle. The goal is to predict the loyalty score for each card_id in the test dataset, which helps Elo better understand customer preferences and improve their recommendation system.

## About Elo
Elo is one of the largest payment brands in Brazil, partnering with merchants to offer promotions and discounts to cardholders. By understanding customer preferences, Elo can deliver more personalized promotions, enhancing customer satisfaction and loyalty. This project aims to build a machine learning model that predicts customer loyalty scores based on their transaction patterns.

## Repository Structure
- `SC4000.ipynb`: Main Jupyter notebook containing data preprocessing, feature engineering, model training, and evaluation
- `data/`: Directory containing all the datasets.
  - `train.csv`: Training data with target loyalty scores
  - `test.csv`: Test data for prediction
  - `historical_transactions.csv`: Historical transaction data for each card
  - `new_merchant_transactions.csv`: New merchant transaction data for each card
  - `merchants.csv`: Information about merchants
  - `Data_Dictionary.xlsx`: Detailed descriptions of all dataset fields
- `pyproject.toml`: Project configuration and dependencies

Note: The datasets are not included in this repository due to their size and licensing restrictions. You can download them from the Kaggle competition page.

## Approach
The approach taken in this project includes:
1. **Data Loading and Exploration**: Understanding the structure and characteristics of the datasets
2. **Data Preprocessing and Cleaning**: Handling missing values and converting data types
3. **Feature Engineering**: Creating meaningful features from transaction data
   - Temporal features from transaction dates
   - Statistical aggregations of transaction amounts
   - Behavioral patterns based on merchant interactions
4. **Aggregating Transaction Data**: Converting transaction-level data to card-level features
5. **Model Training**: Using LightGBM with K-fold cross-validation
6. **Evaluation**: Assessing model performance using Root Mean Squared Error (RMSE)
7. **Prediction**: Generating loyalty score predictions for the test set

## Evaluation Metric
The model performance is evaluated using Root Mean Squared Error (RMSE), which measures the average magnitude of the errors between predicted and actual values. Lower RMSE values indicate better model performance.

## Getting Started
To run this project:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/SC4000-project.git
   cd SC4000-project
   ```

2. Set up the Python environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   ```

3. Ensure you have the required datasets in the `data/` directory

4. Open and run the Jupyter notebook:
   ```
   jupyter notebook SC4000.ipynb
   ```

## Required Dependencies
- Python 3.12+
- pandas 2.2.3+
- numpy 2.2.4+
- scikit-learn 1.6.1+
- lightgbm 4.6.0+
- matplotlib 3.10.1+
- seaborn 0.13.2+
