from args import get_args
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd

from dataset import bone
from model import UNetLext
from trainer import train_model

def main():
    args = get_args()

    csv_dir = args.csv_dir          
    train_path = os.path.join(csv_dir, "train.csv")
    val_path   = os.path.join(csv_dir, "val.csv")

    train_csv = pd.read_csv(train_path)
    val_csv   = pd.read_csv(val_path)

    train_dataset = bone(train_csv, size=224)
    val_dataset   = bone(val_csv,   size=224)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetLext(input_channels=3, output_channels=1, depth=4)
    model.to(device)

    best_model = train_model(model, train_loader, val_loader, device=device)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "best_model.pth")
    torch.save(best_model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
