import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from preprocess import clean_data
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import models

BATCH_SIZE:int = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device is: {device}')

class SpaceshiTitanicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X.values, dtype=torch.float32)

        self.y = None
        if y is not None:
            self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]


def load_dataset(df):
    X = df.drop(columns=['Transported'])
    y = df['Transported']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = SpaceshiTitanicDataset(X_train, y_train)
    val_dataset = SpaceshiTitanicDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, X_train.shape[1]

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    df = clean_data(df)
    train_loader, val_loader, input_dim = load_dataset(df)

    model = models.SpacechipTitanicMLP(input_dim).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    best_val_loss = float('inf')
    train_avg_losses = []
    val_avg_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            prediction = model(batch_X)

            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_avg_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                prediction = model(batch_X)
                loss = criterion(prediction, batch_y)
                val_loss += loss.item()

                predicted_classes = (prediction > 0.5).float()
                correct_predictions += (predicted_classes == batch_y).sum().item()
                total_predictions += batch_y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_avg_losses.append(avg_val_loss)

        val_accuracy = correct_predictions / total_predictions
        val_accuracies.append(val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'spaceship_titanic_model.pth')
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}]: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val accuracy: {val_accuracy:.4f}')

    epochs = range(1, len(train_avg_losses)+1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_avg_losses, label="Training Loss", color='blue')
    axes[0].plot(epochs, val_avg_losses, label="Validation Loss", color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    axes[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('./training_val_metrics.png')
