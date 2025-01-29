# CryptoClusteringCryptoClustering Project
#Overview
In this project, we applied unsupervised learning techniques to analyze the relationship between cryptocurrency price changes over various time frames. Specifically, we used K-Means clustering to determine whether cryptocurrencies are affected by 24-hour and 7-day price changes. We then optimized the clusters using Principal Component Analysis (PCA) to reduce the dimensionality of the data, ultimately comparing clustering results before and after the PCA transformation.

#Project Structure
##This project is organized as follows:

Data Loading & Preparation: The raw cryptocurrency data from the crypto_market_data.csv file is loaded into a DataFrame, cleaned, and scaled using StandardScaler().
K-Means Clustering: The elbow method is applied to determine the optimal number of clusters (k) for the cryptocurrency data, followed by clustering using K-Means.
Visualization: We create scatter plots of clustered data using both original and PCA-transformed datasets, with detailed hover functionality to identify individual cryptocurrencies.
PCA Optimization: Dimensionality reduction is applied using PCA to optimize the clustering process.
Steps to Reproduce
1. Data Preparation
Load the crypto_market_data.csv file into a DataFrame.
Compute summary statistics and plot the data.
Normalize the data using StandardScaler() from scikit-learn.
Create a new DataFrame with the scaled data and set coin_id as the index.
2. Elbow Method for Optimal K
Implement the elbow method by iterating through k values from 1 to 11 and calculating the inertia for each.
Plot the elbow curve and determine the best value for k based on the curve's "elbow".
3. K-Means Clustering
Initialize and fit the K-Means model using the best value for k.
Predict the clusters and add a new column with cluster labels to the scaled data.
Create a scatter plot using hvPlot to visualize the clustering with 24h and 7d price changes.
4. Principal Component Analysis (PCA)
Perform PCA to reduce the data to three principal components.
Analyze the explained variance to understand how much information each principal component contributes.
Create a new DataFrame with the PCA data, with coin_id as the index.
5. Elbow Method on PCA Data
Apply the elbow method again to the PCA-transformed data to determine the best value for k.
Compare this result with the optimal k obtained from the original scaled data.
6. K-Means Clustering with PCA
Use the best k value from the PCA elbow method to fit the K-Means model.
Predict the clusters for the PCA-transformed data and create a scatter plot to visualize the clustering in the first two principal components (PC1 and PC2).
7. Impact of Feature Reduction
Discuss the impact of using fewer features (PCA transformation) on the K-Means clustering results.
Files in this Repository:
Crypto_Clustering.ipynb: Jupyter notebook containing all the code, analysis, and visualizations for this project.
crypto_market_data.csv: The cryptocurrency market data used for analysis.

#Key Insights

The K-Means algorithm was used to segment the cryptocurrencies based on their price changes over 24h and 7d periods.
PCA was applied to reduce dimensionality and optimize clustering.
The optimal number of clusters (k) was determined using the elbow method before and after PCA transformation.
Dependencies

#This project requires the following Python libraries:

pandas
numpy
matplotlib
hvplot
scikit-learn
How to Run This Project:
Clone this repository to your local machine.
Ensure that the required dependencies are installed (e.g., using pip install -r requirements.txt).
Open Crypto_Clustering.ipynb in Jupyter Notebook or JupyterLab and run the cells step by step.
Follow the analysis and visualize the results.

#Conclusion
By clustering cryptocurrencies using K-Means, we explored the influence of price changes over different timeframes and utilized PCA to optimize the process. This analysis provides insights into how clustering can reveal patterns in cryptocurrency price movements and how dimensionality reduction impacts the clustering results.