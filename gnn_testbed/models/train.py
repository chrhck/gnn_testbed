"""Utilities for training."""
import torch
from .evaluation import evaluate_model, predicted_class_hist, plot_confusion


def train_model(
    model,
    train_loader,
    test_loader,
    epochs=100,
    lr=0.005,
    swa=False,
    swa_lr=0.001,
    writer=None,
    scheduler="CosineAnnealingLR",
    n_classes=5,
):
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    if scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        def update_scheduler(_):
            scheduler.step()

    elif scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "max", factor=0.5, verbose=True, patience=5, min_lr=1e-5
        )

        def update_scheduler(acc):
            scheduler.step(acc)

    else:
        raise NotImplementedError(f"scheduler {scheduler} not understood")
    if swa:
        swa_start = int(0.7 * epochs)
        swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=swa_lr)

    def get_loss(model, data):
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        # loss = criterion(out, data.y)  # Compute the loss.
        loss = criterion(out, data.y)  # Compute the loss.
        pred = out.argmax(dim=1)

        return loss, pred

    def train(loader):
        model.train()

        total_loss = 0
        correct = 0

        for data in loader:  # Iterate in batches over the training dataset.

            loss, pred = get_loss(model, data)
            total_loss += loss.item() * len(torch.unique(data.batch))
            correct += int((pred == (data.y)).sum())

            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    def test(loader):
        model.eval()

        total_loss = 0
        correct = 0
        for data in loader:

            loss, pred = get_loss(model, data)
            total_loss += loss.item() * len(torch.unique(data.batch))
            correct += int((pred == (data.y)).sum())

        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    for epoch in range(epochs):
        train_loss, train_acc = train(train_loader)
        test_loss, test_acc = test(test_loader)

        # if epoch % 10 == 0:
        print(
            f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}"
        )

        if writer is not None:
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/test", test_acc, epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/test", test_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            preds, truths, scores = evaluate_model(model, test_loader)
            conf_matrix = predicted_class_hist(truths, preds, n_classes)
            fig = plot_confusion(conf_matrix)
            writer.add_figure("confusion_truth", fig, epoch)

        # scheduler.step(test_acc)
        if swa and (epoch > swa_start):
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            update_scheduler(test_acc)

    # torch.optim.swa_utils.update_bn(train_loader, swa_model)
    if swa:
        return swa_model
    else:
        return model
