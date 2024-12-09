# Analysis of Galaxies Flagged as Anomalies

In my experiments, the galaxy images are split into training and testing sets in a 90:10 ratio. The training set is used to extract features and train autoencoders, K-means, and DBSCAN clustering models. The testing set is then used for anomaly detection. The training set contains 15,962 images, while the testing set contains 1,774 images.

## Feature Extraction

I use the pretrained `ResNet50` from `torchvision.models`, with the default weights trained on the ImageNet1K dataset, to establish the baseline for my experiments. The output from the feature extractor is a vector of length 2,048.

## Autoencoders

For simplicity, I build autoencoders to encode and decode the linear features extracted from `ResNet50`, rather than reconstructing the entire images. The model architecture is shown below:

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int = 2048, latent_dim: int = 64) -> None:
        """Initialize Autoencoder model"""
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
```

I evaluate the reconstruction losses using Mean Squared Error (MSE). The anomaly detection threshold is established based on the interquartile range (IQR), with samples having an MSE greater than $Q_3 + 1.5 \times \text{IQR}$ considered as anomalies.

## K-means Clustering and DBSCAN

I perform clustering on the extracted features. For K-means, I use a similar implementation to that of the reconstruction losses for anomaly detection, but with an allowance for the lower quartile (i.e., distance less than $Q_1 - 1.5 \times \text{IQR}$). DBSCAN flags anomalies as cluster number `-1`. I tune the epsilon values using the K-Nearest Neighbors algorithm to establish appropriate thresholds. The clustering space is visualized using t-SNE, as depicted below:

![t-SNE](/reports/figures/latent_space.png)

The color scale on the right demonstrates the classes, ranging from 0 to 9. The gradient in the color scale has no meaningful interpretation, as the values should be discrete. I observe some clustered patterns, with some class scattering, indicating that there are no obvious representations of the galaxy labels. In the Galaxy10 DECals dataset, the classes are labeled as follows:

```bash
Galaxy10 dataset (17736 images)
├── Class 0 (1081 images): Disturbed Galaxies
├── Class 1 (1853 images): Merging Galaxies
├── Class 2 (2645 images): Round Smooth Galaxies
├── Class 3 (2027 images): In-between Round Smooth Galaxies
├── Class 4 ( 334 images): Cigar Shaped Smooth Galaxies
├── Class 5 (2043 images): Barred Spiral Galaxies
├── Class 6 (1829 images): Unbarred Tight Spiral Galaxies
├── Class 7 (2628 images): Unbarred Loose Spiral Galaxies
├── Class 8 (1423 images): Edge-on Galaxies without Bulge
└── Class 9 (1873 images): Edge-on Galaxies with Bulge
```

## Common Anomalies

Using the reconstruction losses from the [autoencoders](/results/reconstruction_losses_anomalies.txt), I detect a total of 87 anomalies (with the reconstruction error cutoff threshold set at around 0.0352). Similarly, [K-means clustering](/results/kmeans_anomalies.txt) detects around 99 anomalies, though it may flag different samples. [DBSCAN](/results/dbscan_anomalies.txt), however, does not perform well for this task, as it flags 767 samples (almost 44%) as anomalies. Since DBSCAN is density-based and the number of clusters is not well established, I hypothesize that the algorithm may not appropriately identify data points outside the training set and may overfit, as illustrated in the plot above.

For reliable anomaly detection, I decide to use the common anomalies identified by all three methods, resulting in a total of 14 anomalies. Nine of these are shown in the images below:

![Galaxies Flagged as Anomalies](/reports/figures/sampled_anomalies.png)

There are some interesting interpretations here. Overall, galaxies flagged as anomalies tend to possess multiple bright spots in their field of view. In astrophotography, this scenario occurs when two bright objects appear close to each other (e.g., optical doubles). In the context of sky surveys, this suggests that there may be multiple galaxies located at similar distances or positions. However, redshift data is needed to confirm this hypothesis. Potentially, this machine learning tool could identify interacting galactic systems, where two or more galaxies are gravitationally bound (IDs: 1023, 1154, 1373, 1477, 1532, and 1673). Another key observation includes images with artifacts or corrupted signals (e.g., ID: 616, with bright spots in the middle right portion).

In a nutshell, I have demonstrated that this anomaly detection method is effective for identifying potential interacting galactic systems and other deep sky objects within the field. As ongoing sky surveys generate increasingly large image datasets, my tools, with further development, could serve as valuable screening tools for identifying potential objects that astronomers and astrophysicists can study further.
