# Galaxy10 DECals Anomaly Detection

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)

In this repository, I present an analysis of anomaly detection in the Galaxy10 DECals dataset, which is a collection of galaxy images captured over the long history of the Sky Survey.

## Motivation

Analyzing anomalous galaxies can potentially identify interesting samples worth further exploration, guiding researchers to potential discoveries.

This work uses a pretrained `ResNet50` model to extract image features as linear features with a dimension of $2048$. These features are then used to train fully connected autoencoders, perform K-means clustering, and apply DBSCAN. Outliers are identified using reconstruction losses from autoencoders, distances from centroids in K-means, and the `-1` class in DBSCAN.

## Updates

As of January 2025, I have built a public API using Flask for image analysis and anomaly detection with Autoencoders. The implementation details are provided in the sections below. The API is containerized with Docker and deployed on Google Cloud Run. To use it, follow these steps:

1. **Prepare Images**: Format your images similarly to the Galaxy10 DECals dataset. While not mandatory, this ensures better accuracy.
2. **Use the API**: The API is publicly accessible at [https://galaxy10-722997148167.asia-southeast1.run.app](https://galaxy10-722997148167.asia-southeast1.run.app). Note that rate limits may restrict large requests. Here’s an example of how to use it:

   ```bash
   curl -X POST "https://galaxy10-722997148167.asia-southeast1.run.app/detect-anomalies" \
      -H "Content-Type: application/json" \
      -d '{
         "image_paths": [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg"
         ]
      }'
   ```

   Ensure the image URL is publicly accessible on the internet. The output is returned in JSON format, as shown below, with "loss" representing the reconstruction error.

   ```json
   [
     {
       "filename": "https://example.com/image1.jpg",
       "loss": 0.04512,
       "anomaly": true
     },
     {
       "filename": "https://example.com/image2.jpg",
       "loss": 0.01234,
       "anomaly": false
     }
   ]
   ```

## Key Features

- Feature extraction using a pretrained `ResNet50` model (trained on ImageNet1K).
- Autoencoder-based anomaly detection leveraging reconstruction losses.
- Unsupervised anomaly detection using clustering methods (K-means and DBSCAN).
- t-SNE visualizations of latent spaces.
- Reproducible experimental pipelines implemented in PyTorch (refer to `src/main.py`).

## Quick Overview

For a comprehensive analysis, please refer to [this report](/reports/results.md). Overall, we identify a total of 14 anomalous galaxies out of the 1,774 galaxies in the testing set by combining the common anomalies detected by the three methods. You can explore the author's results as well as the documentation either through the [notebook](/galaxy10-anomaly-detection.ipynb) or on [Kaggle](https://www.kaggle.com/code/psrisukhawasu/galaxy10-anomaly-detection). Below are some of the identified anomalies:

![Galaxies Flagged As Anomalies](/reports/figures/sampled_anomalies.png)

## Getting Started

To get started with this project, follow the steps below:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/oadultradeepfield/galaxy10-anomaly-detection.git
   cd galaxy10-anomaly-detection
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. You can install the required packages using the following command:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the PyTorch pipeline**:
   Train the autoencoder, K-means clustering, and DBSCAN models using:

   ```bash
   python src/main.py
   ```

## Acknowledgment

I would like to express my gratitude to the creators and contributors sof the Galaxy10 DECals dataset for providing this invaluable resource. Their work in capturing and curating galaxy images has significantly advanced research in astronomy and machine learning applications.
