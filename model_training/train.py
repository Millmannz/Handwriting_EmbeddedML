import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import DigitClassifier

# --- Data ---
def main():

    torch.backends.cudnn.benchmark = True

    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # Load DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size = 256,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
        persistent_workers = True
    )
    test_loader = DataLoader(
        test_data,
        batch_size = 256,
        shuffle = False,
        num_workers = 2,
        pin_memory = True,
        persistent_workers = True
    )

    # --- Model, loss, optimizer ---
    model = DigitClassifier(784, 64, 10).to(device, non_blocking=True)
    criterion = nn.CrossEntropyLoss()
    # What optimizer did you use in your earlier PyTorch-from-scratch project?
    # torch.optim has several options (SGD, Adam, ...) — pick one and give it
    # model.parameters() and a learning rate.
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0008)

    # --- Training loop ---
    num_epochs = 40 

    for epoch in range(num_epochs):
        model.train()
        running_loss = torch.tensor(0.0, device=device)
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Reset Gradients
            optimizer.zero_grad()

            # Forward Pass
            outputs = model(images)

            # compute the loss between outputs and labels
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()

            # optimizer step
            optimizer.step()

            # --- bookkeeping for display ---
            running_loss += loss.detach() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        test_loss, test_accuracy = evaluate(model=model, loader=test_loader, criterion=criterion, device=device)

        epoch_loss = running_loss.item() / total.item()
        epoch_acc = correct.item() / total.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Acc.: {epoch_acc:.4f} | Test : Loss: {test_loss:.4f} - Acc: {test_accuracy:.4f}")

    torch.save(model.state_dict(), "weights.pth")




def evaluate(model, loader, criterion, device):
    
    model.eval()

    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    # Test
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # --- bookkeeping for display ---
            total_loss += loss.detach() * images.size(0)

           
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return total_loss.item() / total.item(), correct.item() / total.item()

if __name__ == "__main__":
    main()