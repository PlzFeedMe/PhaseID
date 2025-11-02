# Mineralogical Tool for XY data analysis

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import re
import requests

# Constants
MINERAL_DATABASE_URL = "https://mineraldata.gov/api/records/1.0/data/mineral_database.json"

def get_mineral_database():
    # Fetch mineral data from the web API
    response = requests.get(MINERAL_DATABASE_URL)
    soup = BeautifulSoup(response.text, 'xml')

    # Extract mineral attributes and store them in a DataFrame
    mineral_data = []
    for record in soup.find_all('record'):
        attributes = {}
        for attribute in record.find_all('dataElement'):
            name = attribute['name']
            value = attribute.find('value').text
            attributes[name] = value

        mineral_data.append(attributes)

    minerals = pd.DataFrame(mineral_data)
    minerals = minerals[['mineral_formula', 'density', 'color', 'hardness', 'refractive_index', 'transparency']]

    return minerals

def preprocess_xy_data(xy_data):
    # Standardize the XY data and apply K-means clustering

    # Use the measured X and Y values as clustering features
    features = xy_data[['X', 'Y']]

    # Normalize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform K-means clustering (select an appropriate number of clusters)
    kmeans = KMeans(n_clusters=10)
    clusters = kmeans.fit_predict(scaled_features)

    xy_data['cluster'] = clusters

    return xy_data

def identify_minerals(xy_data):
    # Assign placeholder mineral names to each cluster based on their centroids

    cluster_centroids = xy_data.groupby('cluster')[['X', 'Y']].mean()
    cluster_labels = {cluster: f"Cluster {cluster}" for cluster in cluster_centroids.index}

    xy_data['mineral'] = xy_data['cluster'].map(cluster_labels)

    return xy_data

def load_xy_data(filename):
    # Load XY data from a file

    with open(filename, "r") as file:
        lines = file.readlines()
    xy_data = np.array([list(map(float, line.split())) for line in lines if line])

    xy_data = pd.DataFrame(data=xy_data, columns=['X', 'Y'])
    return xy_data

def plot_results(xy_data):
    # Visualize the results using a scatter plot

    unique_clusters = sorted(xy_data['cluster'].unique())
    palette = sns.color_palette("tab10", n_colors=len(unique_clusters))
    color_map = dict(zip(unique_clusters, palette))
    xy_data['color'] = xy_data['cluster'].map(color_map)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy_data['X'], xy_data['Y'], c=xy_data['color'], marker='o')

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_title('Identified Mineral Phases in XY File')

    return fig, ax

def main():
    # Prompt the user to enter the XY data file name
    filename = input("Enter the XY data file name: ")

    # Process and analyze the XY data
    xy_data = load_xy_data(filename)
    xy_data = preprocess_xy_data(xy_data)
    xy_data = identify_minerals(xy_data)

    # Visualize the results
    fig, ax = plot_results(xy_data)
    plt.show()

if __name__ == "__main__":
    main()
