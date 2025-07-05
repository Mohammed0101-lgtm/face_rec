import os
import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


positive_pairs_dir = 'positive_pairs'
negative_pairs_dir = 'negative_pairs'
max_pairs = 1000
res_height = 256
res_width = 256
input_shape = (3, res_height, res_width)

model_save_path = "siamese_model.pth"


def get_negative_pairs():
    pairs = []
    images = [os.path.join(negative_pairs_dir, f) for f in os.listdir(negative_pairs_dir) if f.endswith('.jpg')]

    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append([images[i], images[j]])

            if len(pairs) >= max_pairs:
                break

        if len(pairs) >= max_pairs:
            break

    random.shuffle(pairs)
    return pairs


def get_positive_pairs():
    pairs = []

    for subdir in os.listdir(positive_pairs_dir):
        subdir_path = os.path.join(positive_pairs_dir, subdir)
        if os.path.isdir(subdir_path):
            images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.jpg')]

            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    pairs.append([images[i], images[j]])

                    if len(pairs) >= max_pairs:
                        break

                if len(pairs) >= max_pairs:
                    break

    random.shuffle(pairs)
    return pairs


def get_pairs():
    negative_pairs = get_negative_pairs()
    positive_pairs = get_positive_pairs()
    min_pairs = min(len(positive_pairs), len(negative_pairs))
    negative_pairs = negative_pairs[:min_pairs]
    positive_pairs = positive_pairs[:min_pairs]
    pairs = positive_pairs + negative_pairs
    labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
    return pairs, labels


def preprocess_image(image_path, target_size):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    return transform(img)


def process_pairs(pairs, target_size):
    processed_pairs = []
    for pair in pairs:
        img1 = preprocess_image(pair[0], target_size)
        img2 = preprocess_image(pair[1], target_size)
        processed_pairs.append([img1, img2])
    return processed_pairs


def _get_flattened_size(input_shape):
    dummy = torch.zeros(1, *input_shape)
    model = create_base_network()
    return model(dummy).shape[1]


def create_base_network():
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 128, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Flatten(),
        nn.Linear(256 * 14 * 14, 4096),
        nn.ReLU()
    )
    return model


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_network = create_base_network()
        self.fc = nn.Linear(4096, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_a, input_b):
        out_a = self.base_network(input_a)
        out_b = self.base_network(input_b)
        diff = torch.abs(out_a - out_b)
        out = self.sigmoid(self.fc(diff))
        return out

    def get_embedding(self, input_tensor):
        return self.base_network(input_tensor)


class SiameseDataset(Dataset):
    def __init__(self, pairs, labels):
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], torch.tensor(self.labels[idx], dtype=torch.float32)


def train(model, train_loader, val_loader, num_epochs=50, patience=5):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    no_improve = 0
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for a, b, label in train_loader:
            optimizer.zero_grad()
            output = model(a, b).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * a.size(0)
            preds = (output > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)

        train_acc = correct / total
        train_loss /= total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for a, b, label in val_loader:
                output = model(a, b).squeeze()
                loss = criterion(output, label)
                val_loss += loss.item() * a.size(0)
                preds = (output > 0.5).float()
                val_correct += (preds == label).sum().item()
                val_total += label.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total

        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_save_path)
    return history


def visualize_predictions(model, val_dataset):
    model.eval()
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for ax in axes.flat:
        idx = random.randint(0, len(val_dataset) - 1)
        img1, img2, label = val_dataset[idx]

        with torch.no_grad():
            output = model(img1.unsqueeze(0), img2.unsqueeze(0)).item()

        prediction = 1 if output > 0.5 else 0
        img1 = transforms.ToPILImage()(img1)
        img2 = transforms.ToPILImage()(img2)
        combined = Image.new('RGB', (res_width * 2, res_height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (res_width, 0))
        ax.imshow(combined)
        ax.set_title(f"GT: {int(label)}, Pred: {prediction}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].plot(history['accuracy'], label='Train Accuracy')
    axs[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(loc='lower right')

    axs[1].plot(history['loss'], label='Train Loss')
    axs[1].plot(history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def export_embeddings(model, dataset):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for a, _, label in dataset:
            emb = model.get_embedding(a.unsqueeze(0)).squeeze().numpy()
            embeddings.append(emb)
            labels.append(int(label.item()))

    embeddings = np.array(embeddings)
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(embeddings)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='coolwarm', alpha=0.6)
    plt.title("t-SNE Embedding Visualization")
    plt.show()

if __name__ == '__main__':
    pairs, labels = get_pairs()
    
    if len(pairs) == 0:
        raise ValueError("No data found")

    processed_pairs = process_pairs(pairs, (res_height, res_width))
    train_pairs, val_pairs, train_labels, val_labels = train_test_split(processed_pairs, labels, test_size=0.2, random_state=42)

    train_dataset = SiameseDataset(train_pairs, train_labels)
    val_dataset = SiameseDataset(val_pairs, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = SiameseNetwork()
    history = train(model, train_loader, val_loader)

    visualize_predictions(model, val_dataset)
    export_embeddings(model, val_dataset)
    plot_training_history(history)

    print("Model saved to", model_save_path)