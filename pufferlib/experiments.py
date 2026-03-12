import numpy as np
import math
import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms


def load_spiral_data(n_samples_per_class=2000):
    features_list = []
    labels_list = []
    noise_scale = 0.15

    for class_id in range(4):
        for _ in range(n_samples_per_class):
            t = np.random.uniform(0, 4 * math.pi)
            offset = class_id * 2 * math.pi / 5
            radius = 0.5 + 0.25 * t
            x = radius * math.cos(t + offset) + noise_scale * np.random.randn()
            y = radius * math.sin(t + offset) + noise_scale * np.random.randn()
            z = 0.3 * t + noise_scale * np.random.randn()
            features_list.append([x, y, z])
            labels_list.append(class_id)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    perm = np.random.permutation(len(features))
    return torch.from_numpy(features[perm]), torch.from_numpy(labels[perm])


def load_image_data(fashion=False, n_samples_per_class=2000):
    transform = transforms.Compose([transforms.Resize((7, 7)), transforms.ToTensor()])
    cache_dir = os.path.join(os.path.dirname(__file__), 'pufferlib', 'ocean', 'repgrid', '.mnist_cache')

    if fashion:
        dataset = datasets.FashionMNIST(root=cache_dir, train=True, download=True, transform=transform)
        class_map = {0: 0, 2: 1, 4: 2, 3: 3}
    else:
        dataset = datasets.MNIST(root=cache_dir, train=True, download=True, transform=transform)
        class_map = {0: 0, 1: 1, 2: 2, 3: 3}

    data_by_class = {k: [] for k in range(4)}
    for img, label in dataset:
        if label in class_map:
            data_by_class[class_map[label]].append(img.numpy().flatten())

    all_data = np.concatenate([np.array(data_by_class[k], dtype=np.float32) for k in range(4)], axis=0)
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0) + 1e-8

    features_list = []
    labels_list = []
    for class_id in range(4):
        arr = (np.array(data_by_class[class_id], dtype=np.float32) - global_mean) / global_std
        if len(arr) > n_samples_per_class:
            idx = np.random.choice(len(arr), n_samples_per_class, replace=False)
            arr = arr[idx]
        features_list.append(arr)
        labels_list.append(np.full(len(arr), class_id, dtype=np.int64))

    features = np.concatenate(features_list)
    labels = np.concatenate(labels_list)
    perm = np.random.permutation(len(features))
    return torch.from_numpy(features[perm]), torch.from_numpy(labels[perm])


def train_and_eval(features, labels, num_epochs=100, lr=0.01, train_split=0.8):
    n = len(features)
    n_train = int(n * train_split)
    perm = torch.randperm(n)
    train_x, train_y = features[perm[:n_train]], labels[perm[:n_train]]
    val_x, val_y = features[perm[n_train:]], labels[perm[n_train:]]

    linear = nn.Linear(features.shape[1], 4)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = criterion(linear(train_x), train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        val_acc = (linear(val_x).argmax(dim=1) == val_y).float().mean().item()
    return val_acc


if __name__ == "__main__":
    print("Loading datasets...")
    spiral_f, spiral_l = load_spiral_data()
    mnist_f, mnist_l = load_image_data(fashion=False)
    fmnist_f, fmnist_l = load_image_data(fashion=True)

    print("Training linear classifiers...\n")
    spiral_acc = train_and_eval(spiral_f, spiral_l)
    mnist_acc = train_and_eval(mnist_f, mnist_l)
    fmnist_acc = train_and_eval(fmnist_f, fmnist_l)

    print(f"Spiral accuracy:       {spiral_acc:.4f}")
    print(f"MNIST accuracy:        {mnist_acc:.4f}")
    print(f"Fashion MNIST accuracy:{fmnist_acc:.4f}")
