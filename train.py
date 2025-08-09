import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import argparse
from dataset import get_data_loaders
from model import CA_SAUNet
from utils import pixel_accuracy, mean_iou, dice_coefficient, f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path, best_model_path, dataset_name):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 20
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        best_val_loss = checkpoint['best_val_loss']
        early_stop_counter = checkpoint['early_stop_counter']
        print(f"Checkpoint loaded, resuming from epoch {start_epoch}.")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)[0]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)[0]
                loss_val = criterion(outputs, masks)
                val_running_loss += loss_val.item() * images.size(0)
        val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss - 0.0001:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model updated.")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered!")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'early_stop_counter': early_stop_counter,
                }, checkpoint_path)
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'early_stop_counter': early_stop_counter,
        }, checkpoint_path)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(val_losses, label="Val Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss over Epochs")
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader, device, dataset_name):
    model.load_state_dict(torch.load(f'best_model_{dataset_name}.pth'))
    model.eval()
    total_pa = 0.0
    total_miou = 0.0
    total_dice = 0.0
    total_f1 = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)[0]
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            img = images[0].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            true_mask = masks[0].cpu().numpy()
            pred_mask = preds[0]

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.title("Input Image")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(img, cmap='gray')
            plt.imshow(true_mask, cmap="gray", alpha=0.3)
            plt.title("Ground Truth with Input Image")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(img, cmap='gray')
            plt.imshow(pred_mask, cmap="gray", alpha=0.3)
            plt.title("Predicted Mask")
            plt.axis('off')
            plt.show()

            batch_size = images.shape[0]
            for j in range(batch_size):
                gt_mask = masks[j].cpu().numpy()
                pred_mask = preds[j]
                pa = pixel_accuracy(pred_mask, gt_mask)
                miou = mean_iou(pred_mask, gt_mask, num_classes=2)
                dice = dice_coefficient(pred_mask, gt_mask, num_classes=2)
                f1 = f1_score(pred_mask, gt_mask, num_classes=2)
                total_pa += pa
                total_miou += miou
                total_dice += dice
                total_f1 += f1
                total_samples += 1

    avg_pa = total_pa / total_samples
    avg_miou = total_miou / total_samples
    avg_dice = total_dice / total_samples
    avg_f1 = total_f1 / total_samples

    print(f"Average Pixel Accuracy: {avg_pa:.4f}")
    print(f"Average Mean IoU: {avg_miou:.4f}")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train CA_SAUNet on PlantDoc or ISIC dataset')
    parser.add_argument('--dataset', type=str, default='plantdoc', choices=['plantdoc', 'isic'], help='Dataset to use (plantdoc or isic)')
    parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Igdir-Uni-Genel/Z_Works-2025', help='Path to dataset directory')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CA_SAUNet(num_classes=2, num_filters=32, pretrained=False, is_deconv=True, dataset=args.dataset).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_loader, val_loader, test_loader, train_samples, val_samples, test_samples = get_data_loaders(
        args.dataset, args.data_dir, args.batch_size
    )

    print(f"Train samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Test samples: {test_samples}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    for images, masks in test_loader:
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        break

    checkpoint_path = os.path.join(args.data_dir, f'checkpoint_{args.dataset}.pth')
    best_model_path = os.path.join(args.data_dir, f'best_model_{args.dataset}.pth')

    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device, checkpoint_path, best_model_path, args.dataset)
    evaluate_model(model, test_loader, device, args.dataset)

if __name__ == '__main__':
    main()