# **House Sales Price Prediction in King County, Washington**  

## **Problem Description**  
This project aims to develop a supervised learning model to predict house sales prices in King County, Washington, from May 2014 to May 2015. The prediction will be based on various variables and housing characteristics contained in the dataset.  

## **Dataset**  
The dataset used in this project was obtained from the Kaggle platform. It contains information about properties sold in King County, Washington, during the specified period.  

- **Source:** [King County House Sales Dataset - Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction)  
- **Public:** Yes  
- **Description:** The dataset includes variables such as sale price, number of bedrooms, bathrooms, built area, lot size, year of construction, geographic location, among others.  

## **Adopted Solution**  
To address this problem, the following approach will be taken:  

1. **Exploratory Data Analysis (EDA):** Analyze variable distributions, identify outliers, and perform appropriate data preprocessing.  
2. **Preprocessing:** Handle missing values, transform categorical variables, and scale numerical variables if necessary.  
3. **Model Training:** Test different regression models, such as linear regression, regularized regression, and tree-based models (Random Forest, XGBoost) to evaluate their performance.  
4. **Evaluation and Tuning:** Measure model accuracy using metrics like RMSE and R², adjusting hyperparameters to optimize results.  
5. **Conclusions and Recommendations:** Interpret the results and assess the model’s applicability for future predictions.  

## **Repository Structure**  
```
/
├── data/                  # Contains the original dataset and processed versions
├── img/                   # Support images and visualizations generated during analysis
├── models/                # Trained models and configuration files
├── notebooks/             # Jupyter Notebooks for data cleaning, analysis, and model development
├── results_notebook/      # Notebook with the entire process and prediction evaluation results
├── utils/                 # Auxiliary functions and useful tools
├── README.md              # Project explanatory document (Spanish)
├── README_en.md           # Project explanatory document (Englilsh)
```

This repository provides all necessary files for replicating and improving the prediction model.  