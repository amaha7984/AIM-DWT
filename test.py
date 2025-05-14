import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import precision_score, roc_auc_score
import numpy as np

from models import AIMClassificationModel
from dwt import HaarTransform
from data import val_transforms
from aim.v2.utils import load_pretrained

def load_model(checkpoint_path, device, base_model):
    model = AIMClassificationModel(base_model, num_classes=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model.to(device)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct_real, correct_fake, total_real, total_fake = 0, 0, 0, 0
    all_labels, all_outputs = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

            correct_real += ((preds == 0) & (labels == 0)).sum().item()
            correct_fake += ((preds == 1) & (labels == 1)).sum().item()
            total_real += (labels == 0).sum().item()
            total_fake += (labels == 1).sum().item()

    bin_preds = (np.array(all_outputs) > 0.5).astype(int)
    bin_labels = np.array(all_labels).astype(int)

    precision_real = precision_score(bin_labels, bin_preds, pos_label=0, zero_division=0)
    precision_fake = precision_score(bin_labels, bin_preds, pos_label=1, zero_division=0)

    accuracy_real = 100.0 * correct_real / total_real if total_real else 0.0
    accuracy_fake = 100.0 * correct_fake / total_fake if total_fake else 0.0

    auc = roc_auc_score(bin_labels, all_outputs)

    return accuracy_real, accuracy_fake, precision_real, precision_fake, auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = load_pretrained("aimv2-large-patch14-224", backend="torch")
    model = load_model(args.checkpoint_path, device, base_model)

    test_loader = DataLoader(
        datasets.ImageFolder(args.test_path, transform=val_transforms(img_size=224)),
        batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    acc_real, acc_fake, prec_real, prec_fake, auc = evaluate_model(model, test_loader, device)

    print(f"Accuracy (Real): {acc_real:.2f}%")
    print(f"Accuracy (Fake): {acc_fake:.2f}%")
    print(f"Precision (Real): {prec_real:.4f}")
    print(f"Precision (Fake): {prec_fake:.4f}")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate AIM-DWT model on test data")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    
    args = parser.parse_args()
    main(args)
