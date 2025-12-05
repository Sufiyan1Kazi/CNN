from args import get_args
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import cv2
import numpy as np

from dataset import bone
from model import UNetLext

def evaluate_test():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(os.path.join(args.csv_dir, "test.csv"))
    test_dataset = bone(test_df, size=224)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = UNetLext(
        input_channels=3,
        output_channels=1,
        depth=4
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pth"), map_location=device))
    model.eval()

    os.makedirs("test_predictions", exist_ok=True)

    with torch.no_grad():
        for idx, (imgs, _) in enumerate(test_loader):
            imgs = imgs.to(device)
            logits = model(imgs)

            probs = torch.sigmoid(logits)[0,0].cpu().numpy()
            pred = (probs > 0.5).astype(np.uint8) * 255

            img_np = imgs[0].permute(1,2,0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"test_predictions/img_{idx}.png", img_np)
            cv2.imwrite(f"test_predictions/pred_{idx}.png", pred)

    print("Saved predictions!")

if __name__ == "__main__":
    evaluate_test()
