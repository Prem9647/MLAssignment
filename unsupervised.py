# -----------------------------
# STREAMLIT UNSUPERVISED APP
# -----------------------------

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Title
st.title("🌱 Environmental Clustering Analysis")

# Load dataset
data = pd.read_csv("env.csv")

st.subheader("Dataset Preview")
st.write(data.head())

# Remove target column if exists
if "Pollution_Level" in data.columns:
    X = data.drop("Pollution_Level", axis=1)
else:
    X = data

# Sidebar controls
st.sidebar.header("Clustering Settings")

k = st.sidebar.slider("Number of Clusters (K-Means)", 2, 5, 3)

# -----------------------------
# K-Means Clustering
# -----------------------------
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Add cluster labels
data["Cluster"] = kmeans.labels_

st.subheader("K-Means Cluster Labels")
st.write(data[["Cluster"]].head(10))

# Plot clusters (2 features for visualization)
st.subheader("K-Means Visualization")

plt.figure()
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_)
plt.xlabel(X.columns[0])
plt.ylabel(X.columns[1])
plt.title("K-Means Clustering")
st.pyplot(plt)

# -----------------------------
# Hierarchical Clustering
# -----------------------------
from scipy.cluster.hierarchy import linkage, dendrogram

st.subheader("Hierarchical Clustering Dendrogram")

Z = linkage(X, method='ward')

fig, ax = plt.subplots()
dendrogram(Z, ax=ax)
ax.set_title("Dendrogram")
ax.set_xlabel("Data Points")
ax.set_ylabel("Distance")

st.pyplot(fig)