# Unsupervised Machine Learning : Credit Card Analysis
## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [References](#references)

## Project Description

We will analyze a credit card database. The goal is to separate customers based on their characteristics, i.e., clusters that best describe financial and behavioral patterns.

## Data

Summary of Features:

| Column Name        | Description                                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------- |
| **Sl_No**          | Serial number or index of the row in the dataset                                                |
| **Customer Key**   | A unique identifier for each customer                                                           |
| **Avg_Credit_Limit** | Average credit limit across all the credit cards of the customer                                 |
| **Total_Credit_Cards** | Total number of credit cards possessed by the customer                                         |
| **Total_visits_bank** | Total number of visits made by the customer to the bank                                         |
| **Total_visits_online** | Total number of visits made by the customer to the bank's website                              |
| **Total_calls_made** | Total number of calls made by the customer to the bank's customer support                       |



## Methodology

In this project, we will apply hierarchical clustering, Meanshift, Gaussian Mixture, and KMeans algorithms to find the best one that describes the clusters based on our data. For this purpose, an exploratory analysis was conducted, which can be found in the notebook. As it is a dataset with multiple features, I used PCA for 2 components to visualize how the algorithms separated the clusters and evaluate them.

## Results

Before defining the profiles, it is important to mention that the hierarchical clustering, Meanshift, Gaussian Mixture, and KMeans algorithms had similar results. After analyzing the PCA plot, it was clear that the DBSCAN clustering was not satisfactory.

When observing the clusters and the description of each variable, we have some variables that show differences between populations, as seen in the boxplots. Based on this, we can create three customer profiles:

- ***Cluster 0 consists of customers with low to medium credit limits, a medium number of credit cards, high bank visits, low online visits, and low phone calls made.***
- ***Cluster 1 consists of customers with low credit limits, a low number of credit cards, low bank visits, high online visits, and phone calls made.***
- ***Cluster 2 consists of customers with high credit limits, a high number of credit cards, low bank visits, high online visits, and low phone calls made.***




## Usage

### Libraries Used
- `pandas`: data manipulation library used to read, clean and manipulate the dataset.
- `numpy`: numerical computing library used to perform mathematical operations on the data.
- `matplotlib`: plotting library used to create visualizations.
- `seaborn`: plotting library based on matplotlib used to create attractive visualizations.
- `sklearn`: machine learning library used to train and evaluate models.
- `scipy`: library used for scientific and technical computing.
- `scipy.cluster.hierarchy`: library used for hierarchical clustering.
- `sklearn.cluster`: library used for clustering algorithms.
- `sklearn.neighbors`: library used for unsupervised nearest neighbors algorithms.
- `sklearn.decomposition`: library used for feature dimensionality reduction.
- `metrics`: library used to evaluate models.
- `StandardScaler`: library used to standardize the data.

### Installation
To install the required libraries, please run the following command in your terminal:

pip install pandas numpy matplotlib seaborn scikit-learn scipy

## References
https://www.kaggle.com/datasets/aryashah2k/credit-card-customer-data
