from args import get_args
import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, device):  
    args = get_args()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(int(args.epochs)):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)

            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}")

    return model
