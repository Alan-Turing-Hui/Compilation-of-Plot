import matplotlib.pyplot as plt

def plot_classification_metrics(train_losses, train_accuracies, val_losses, val_accuracies, 
                 val_precisions, val_recalls, val_f1s, save_path=None):
    plt.figure(figsize=(10, 15))

    # Loss
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(3, 1, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Precision, Recall, F1 Score
    plt.subplot(3, 1, 3)
    plt.plot(val_precisions, label='Val Precision', color='blue')
    plt.plot(val_recalls, label='Val Recall', color='orange')
    plt.plot(val_f1s, label='Val F1 Score', color='green')
    plt.title('Validation Metrics over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

    plt.show()