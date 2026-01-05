import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the dataset
df = pd.read_csv('Online_Retail.csv')

# Check for missing values and drop them
df.dropna(inplace=True)

# Convert categorical variables to dummy/indicator variables
df = pd.get_dummies(df, columns=['ProductCategory'])

# Selecting features for clustering
features = df[['PurchaseAmount', 'Frequency'] + [col for col in df.columns if 'ProductCategory' in col]]

# Ensure the features are numeric
features = features.select_dtypes(include=[np.number])

# Check if there are any non-numeric data types
print(features.dtypes)

# Scaling the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Manually set the optimal number of clusters based on the plot
optimal_clusters = 3  # Example: Replace with the number you determined

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)
df['Cluster'] = clusters

# Visualize the clusters (assuming you have 2 main features to plot)
plt.figure(figsize=(10, 6))
plt.scatter(df['PurchaseAmount'], df['Frequency'], c=df['Cluster'], cmap='rainbow')
plt.title('Customer Segments')
plt.xlabel('Purchase Amount')
plt.ylabel('Frequency')
plt.show()