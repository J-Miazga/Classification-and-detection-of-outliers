import clip
import torch
import os
import numpy as np
import shutil
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from PIL import Image

def get_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = model.encode_image(image_input)
    return features.cpu().numpy().flatten()


def load_images(directory):
    image_paths = []
    for filename in os.listdir(directory):
        image_paths.append(os.path.join(directory, filename))
    return image_paths


def perform_dbscan(features, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples,metric="euclidean")
    labels = dbscan.fit_predict(features)
    return labels, dbscan


model, preprocess = clip.load("ViT-L/14")

image_directory = 'Final_images_dataset'
image_paths = load_images(image_directory)
all_features = []

for image_path in image_paths:
    features = get_features(image_path)
    all_features.append(features)

all_features = np.array(all_features)
scaler = StandardScaler()
all_features_scaled = scaler.fit_transform(all_features)

labels, dbscan = perform_dbscan(all_features_scaled, eps=34.3, min_samples=2)
outliers = np.where(labels == -1)[0]

output_directory = 'output_clusters'
os.makedirs(output_directory, exist_ok=True)
outlier_folder = os.path.join(output_directory, 'outliers')
os.makedirs(outlier_folder, exist_ok=True)

for idx, image_path in enumerate(image_paths):
    image_name = os.path.basename(image_path)

    if idx in outliers:
        shutil.copy(image_path, os.path.join(outlier_folder, image_name))
    else:
        cluster_id = labels[idx]
        if cluster_id != -1:
            cluster_folder = os.path.join(output_directory, f'cluster_{cluster_id}')
            os.makedirs(cluster_folder, exist_ok=True)
            shutil.copy(image_path, os.path.join(cluster_folder, image_name))

print(f"Found {len(outliers)} outliers")