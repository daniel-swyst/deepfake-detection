import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import psutil
from efficientnet_pytorch import EfficientNet
import gc

# Set directories for training and validation data
train_dir = '/train'
val_dir = '/valid'

# Data augmentation and normalization for different EfficientNet models
def get_transforms(model_name):
    resolution = {
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
    }[model_name]

    return {
        'train': transforms.Compose([
            transforms.RandomRotation(20),
            transforms.Resize((resolution, resolution)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# List of models to train
model_names = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2']

total_training_times = {}
ram_usages = {}

# Train and evaluate each model
for model_name in model_names:
    print(f'Training {model_name}...')

    # Load the model
    model = EfficientNet.from_pretrained(model_name)
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 1)

    data_transforms = get_transforms(model_name)

    train_dataset = datasets.ImageFolder(train_dir, data_transforms['train'])
    val_dataset = datasets.ImageFolder(val_dir, data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_epochs = 40
    best_model_wts = model.state_dict()
    best_acc = 0.0

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    epoch_times, ram_usage = [], []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        start_time = time.time()

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().unsqueeze(1)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds = (preds > 0.5).float()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.cpu())  
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.cpu())  

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        end_time = time.time()
        epoch_times.append(end_time - start_time)
        ram_usage.append(psutil.virtual_memory().used / (1024 ** 3))  # RAM usage in GB

        # Clear GPU cache
        torch.cuda.empty_cache()
        # Delete unused variables
        del inputs, labels, outputs, preds, loss
        gc.collect()
        print()

    model.load_state_dict(best_model_wts)

    # Print final training and validation accuracy and loss
    print(f'Final Training Loss: {train_losses[-1]:.4f}')
    print(f'Final Training Accuracy: {train_accuracies[-1]:.4f}')
    print(f'Final Validation Loss: {val_losses[-1]:.4f}')
    print(f'Final Validation Accuracy: {val_accuracies[-1]:.4f}')

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        with torch.no_grad():
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        preds = (preds > 0.5).float()
        val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)

    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # Convert accuracies to float before plotting
    train_accuracies = [acc.item() for acc in train_accuracies]
    val_accuracies = [acc.item() for acc in val_accuracies]

    # Plot training and validation accuracy
    plt.figure()
    plt.plot(train_accuracies, label='Accuracy (train set)')
    plt.plot(val_accuracies, label='Accuracy (valid set)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.savefig(f'/efficientnet/{model_name}_accuracy.png')
    plt.show()

    # Plot training and validation loss
    plt.figure()
    plt.plot(train_losses, label='Loss (train set)')
    plt.plot(val_losses, label='Loss (valid set)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.savefig(f'efficientnet/{model_name}_loss.png')
    plt.show()

    # Save the model
    torch.save(model.state_dict(), f'efficientnet/{model_name}_model.pth')

    # Total times and RAM usage
    total_time = sum(epoch_times)
    average_ram = sum(ram_usage) / len(ram_usage)

    total_training_times[model_name] = total_time
    ram_usages[model_name] = average_ram

# Plot total training times for all models
plt.figure()
bars = plt.bar(total_training_times.keys(), total_training_times.values())
plt.xlabel('Model')
plt.ylabel('Total training time [s]]')
plt.title('Total training time [s]')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')  
plt.savefig(f'efficientnet/total_training_time_all_models.png')
plt.show()

# Plot average RAM usage for all models
plt.figure()
bars = plt.bar(ram_usages.keys(), ram_usages.values())
plt.xlabel('Model')
plt.ylabel('RAM average usage [GB]')
plt.title('RAM average usage')
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', va='bottom', ha='center')  
plt.savefig(f'efficientnet/average_ram_usage_all_models.png')
plt.show()
