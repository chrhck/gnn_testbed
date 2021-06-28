"""Utilities for training."""
import torch


def train_model(model, train_loader, test_loader, epochs=100, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 100
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, verbose=True
    )

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(
                data.x, data.edge_index, data.batch
            )  # Perform a single forward pass.
            # loss = criterion(out, data.y)  # Compute the loss.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            # correct += int((pred == data.y).sum())  # Check against ground-truth labels.
            correct += int(
                (pred == (data.y)).sum()
            )  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    for epoch in range(epochs):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
            )
        scheduler.step(test_acc)
