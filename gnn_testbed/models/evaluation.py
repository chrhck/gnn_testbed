import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, loader):
    preds = []
    truths = []
    scores = []

    with torch.no_grad():

        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            preds.append(pred)
            truths.append(data.y)
            scores.append(out)

        preds = torch.cat(preds).cpu()
        truths = torch.cat(truths).cpu()
        scores = torch.cat(scores).cpu()

    return preds, truths, scores


def predicted_class_hist(truths, preds, n_classes):
    """Make a histogram of predicted class normalized on truth"""
    confusion_matrix = []

    for i in range(n_classes):
        true_sel = truths == i

        predictions = np.histogram(
            preds[true_sel], bins=np.arange(0, n_classes + 1, 1)
        )[0]
        predictions = predictions / predictions.sum()
        confusion_matrix.append(predictions)
    confusion_matrix = np.vstack(confusion_matrix)
    return confusion_matrix


def plot_confusion(confusion_matrix):
    label_map = {
        0: "Contained Cascade",
        1: "Throughgoing Track",
        2: "Starting Track",
        3: "Stopping Track",
        4: "Skimming",
    }
    fig = plt.figure()
    sns.heatmap(
        confusion_matrix,
        cmap=plt.cm.Blues,
        annot=True,
        xticklabels=list(label_map.values()),
        yticklabels=list(label_map.values()),
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    return fig
