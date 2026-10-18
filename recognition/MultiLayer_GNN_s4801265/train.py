
import torch
from torch import nn

from dataset import load_dataset
from modules import MLP

import matplotlib.pyplot as plt
import csv

import time

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

# Training settings
SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 500
DROPOUT = 0.5
HIDDEN_DIM = 32


def train_epoch(model, data, optimizer, criterion):
    model.train()

    # Clear gradients from previous epoch
    optimizer.zero_grad()

    # Predict classes for all nodes
    output = model(data.x)

    # Calculate loss using training labels only
    loss = criterion(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    # Calculate gradients
    loss.backward()

    # Update model parameters
    optimizer.step()

    return loss.item()


def evaluate(model, data, mask, criterion):
    model.eval()

    with torch.no_grad():
        output = model(data.x)

        loss = criterion(
            output[mask],
            data.y[mask]
        )

        predictions = output[mask].argmax(dim=1)

        accuracy = (
            predictions == data.y[mask]
        ).float().mean().item()

    return loss.item(), accuracy


def evaluate_test_set(model, data):
    model.eval()

    with torch.no_grad():
        # Generate predictions for all nodes
        output = model(data.x)

        # Select only test nodes
        test_output = output[data.test_mask]
        test_labels = data.y[data.test_mask]

        # Convert logits into class probabilities
        probabilities = torch.softmax(test_output, dim=1)

        # Find the predicted class for each node
        predictions = probabilities.argmax(dim=1)

    # Move results to CPU for evaluation
    labels = test_labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    probabilities = probabilities.cpu().numpy()

    # Get original node indices
    node_indices = data.test_mask.nonzero(
        as_tuple=True
    )[0].cpu().numpy()

    # Calculate classification metrics
    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            predictions,
            labels=[0, 1, 2, 3],
            zero_division=0
        )
    )

    macro_f1 = f1.mean()

    # Calculate multiclass AUROC
    auroc = roc_auc_score(
        labels,
        probabilities,
        labels=[0, 1, 2, 3],
        multi_class="ovr",
        average="macro"
    )

    # Calculate confusion matrix
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1, 2, 3]
    )

    # Print results
    print("\n=== FINAL MLP TEST RESULTS ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Macro AUROC (OvR): {auroc:.4f}")

    print("\nPer-Class Results:")

    for class_id in range(4):
        print(
            f"Class {class_id}: "
            f"Precision={precision[class_id]:.4f}, "
            f"Recall={recall[class_id]:.4f}, "
            f"F1={f1[class_id]:.4f}, "
            f"Support={support[class_id]}"
        )

    print("\nConfusion Matrix:")
    print(matrix)

    # Save individual test predictions and probabilities
    with open("mlp_predictions.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "node_id",
            "true_label",
            "predicted_label",
            "confidence",
            "prob_class_0",
            "prob_class_1",
            "prob_class_2",
            "prob_class_3"
        ])

        for node_id, label, prediction, probs in zip(
            node_indices, labels, predictions, probabilities
        ):
            writer.writerow([
                int(node_id),
                int(label),
                int(prediction),
                float(probs.max()),
                *[float(p) for p in probs]
            ])

    print("\nMLP predictions saved to mlp_predictions.csv")


def plot_training_results(training_losses, validation_losses,
                          training_accuracies, validation_accuracies):
    """
    Plot and save training and validation results.

    Args:
        training_losses: Training loss for each epoch.
        validation_losses: Validation loss for each epoch.
        training_accuracies: Training accuracy for each epoch.
        validation_accuracies: Validation accuracy for each epoch.

    Returns:
        None
    """
    epochs = range(1, len(training_losses) + 1)

    # Plot training and validation losses
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("MLP Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("mlp_loss.png", dpi=200)
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MLP Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("mlp_accuracy.png", dpi=200)
    plt.close()

    print("\nTraining plots saved successfully.")



def calculate_ece(probabilities, labels, num_bins=10):
    """
    Calculate Expected Calibration Error using confidence bins.

    Args:
        probabilities: Class probabilities for each node.
        labels: Actual class labels.
        num_bins: Number of confidence bins.

    Returns:
        Expected Calibration Error.
    """
    confidences, predictions = probabilities.max(dim=1)
    correct = (predictions == labels).float()

    ece = 0.0

    for bin_id in range(num_bins):
        lower = bin_id / num_bins
        upper = (bin_id + 1) / num_bins

        # Include the upper bound in the final bin
        if bin_id == num_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)

        if in_bin.any().item():
            bin_accuracy = correct[in_bin].mean().item()
            bin_confidence = confidences[in_bin].mean().item()
            bin_fraction = in_bin.float().mean().item()

            ece += bin_fraction * abs(
                bin_accuracy - bin_confidence
            )

    return ece


def select_confidence_threshold(
    model, data, target_accuracy=0.90
):
    """
    Select a confidence threshold using validation data.

    Args:
        model: Trained MLP.
        data: Graph containing validation features and labels.
        target_accuracy: Minimum desired accuracy for accepted nodes.

    Returns:
        Selected threshold, or None if no threshold qualifies.
    """
    model.eval()

    with torch.no_grad():
        output = model(data.x)
        probabilities = torch.softmax(
            output[data.val_mask], dim=1
        )

        labels = data.y[data.val_mask]

        confidences, predictions = probabilities.max(dim=1)
        correct = predictions == labels

    # Candidate thresholds from 0.50 to 0.99
    thresholds = [i / 100 for i in range(50, 100)]

    for threshold in thresholds:
        accepted = confidences >= threshold

        # Avoid calculating accuracy on an empty set
        if accepted.sum().item() == 0:
            continue

        accepted_accuracy = (
            correct[accepted].float().mean().item()
        )

        if accepted_accuracy >= target_accuracy:
            print(
                f"\nSelected Confidence Threshold: {threshold:.2f}"
            )
            print(
                f"Validation Accepted Accuracy: "
                f"{accepted_accuracy:.4f}"
            )

            return threshold

    print("\nNo threshold met the target validation accuracy.")
    return None


def analyse_confidence(model, data, mask, threshold, split_name):
    """
    Analyse confidence reliability and manual-review workload
    for a selected data split.

    Args:
        model: Trained MLP.
        data: Graph containing features and labels.
        mask: Boolean mask for the selected split.
        threshold: Confidence threshold selected on validation data.
        split_name: Name of the evaluated split.

    Returns:
        None
    """
    model.eval()

    with torch.no_grad():
        output = model(data.x)

        probabilities = torch.softmax(
            output[mask], dim=1
        )

        labels = data.y[mask]

        confidences, predictions = probabilities.max(dim=1)
        correct = predictions == labels

    # Expected Calibration Error
    ece = calculate_ece(probabilities, labels)

    # Average confidence for correct/incorrect predictions
    correct_confidence = (
        confidences[correct].mean().item()
        if correct.any().item() else float("nan")
    )

    incorrect_confidence = (
        confidences[~correct].mean().item()
        if (~correct).any().item() else float("nan")
    )

    print(f"\n=== {split_name.upper()} CONFIDENCE ANALYSIS ===")
    print(f"ECE: {ece:.4f}")
    print(f"Mean Correct Confidence: {correct_confidence:.4f}")
    print(f"Mean Incorrect Confidence: {incorrect_confidence:.4f}")

    # Evaluate manual-review performance
    if threshold is None:
        print("No qualifying manual-review threshold.")
        return

    accepted = confidences >= threshold

    accepted_count = accepted.sum().item()
    total_count = len(labels)

    accepted_percentage = 100 * accepted_count / total_count
    review_percentage = 100 - accepted_percentage

    print(f"\nConfidence Threshold: {threshold:.2f}")
    print(f"Automatically Accepted: {accepted_percentage:.2f}%")
    print(f"Sent for Manual Review: {review_percentage:.2f}%")

    if accepted_count > 0:
        accepted_accuracy = (
            correct[accepted].float().mean().item()
        )

        print(
            f"Accepted Prediction Accuracy: "
            f"{accepted_accuracy:.4f}"
        )
    else:
        print("Accepted Prediction Accuracy: N/A")


def measure_resources(model, data, device):
    """
    Measure model parameters, inference time and peak GPU memory.

    Args:
        model: Trained MLP.
        data: Graph containing node features.
        device: CPU or GPU device.

    Returns:
        None
    """
    model.eval()

    # Count total model parameters
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print("\n=== MLP RESOURCE MEASUREMENTS ===")
    print(f"Model Parameters: {parameter_count:,}")

    # Warm up before timing
    with torch.inference_mode():
        for _ in range(5):
            model(data.x)

    # Synchronise GPU operations before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # Measure repeated forward passes
    inference_times = []

    with torch.inference_mode():
        for _ in range(20):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            output = model(data.x)
            test_output = output[data.test_mask]

            # Ensure GPU operations have completed
            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start_time
            inference_times.append(elapsed)

    average_time = sum(inference_times) / len(inference_times)

    print(f"Average Inference Time: {average_time:.6f} seconds")

    if device.type == "cuda":
        peak_memory = (
            torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        )

        print(f"Peak GPU Memory: {peak_memory:.2f} MiB")
    else:
        print("Peak GPU Memory: N/A (CPU execution)")


def analyse_failures(model, data, num_examples=5):
    """
    Identify confidently incorrect test-node predictions.

    Args:
        model: Trained MLP.
        data: Graph containing node features and test labels.
        num_examples: Number of failures to display.

    Returns:
        None
    """
    model.eval()

    with torch.no_grad():
        output = model(data.x)
        probabilities = torch.softmax(output, dim=1)

        confidences, predictions = probabilities.max(dim=1)

    # Find node IDs belonging to the test set
    test_indices = data.test_mask.nonzero(
        as_tuple=True
    )[0]

    # Find incorrectly classified test nodes
    incorrect = (
        predictions[test_indices] != data.y[test_indices]
    )

    incorrect_indices = test_indices[incorrect]

    # Sort incorrect nodes by descending confidence
    sorted_indices = incorrect_indices[
        torch.argsort(
            confidences[incorrect_indices],
            descending=True
        )
    ]

    print("\n=== MLP FAILURE ANALYSIS ===")

    for node_id in sorted_indices[:num_examples]:
        node = node_id.item()

        print(f"\nNode ID: {node}")
        print(f"Actual Class: {data.y[node].item()}")
        print(f"Predicted Class: {predictions[node].item()}")
        print(f"Confidence: {confidences[node].item():.4f}")

        print(
            "Class Probabilities:",
            probabilities[node].cpu().tolist()
        )


def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)

    # Choose GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Load the dataset and transfer it to the device
    data = load_dataset(seed=SEED).to(device)

    # Initialise the MLP baseline
    model = MLP(
        input_dim=128,
        hidden_dim=HIDDEN_DIM,
        output_dim=4,
        dropout=DROPOUT
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("MLP initialised successfully.")

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_path = "mlp_best.pt"

    training_losses = []
    validation_losses = []

    training_accuracies = []
    validation_accuracies = []

    # Train the MLP baseline
    for epoch in range(1, EPOCHS + 1):

        # Train the MLP for one epoch
        train_epoch(model, data, optimizer, criterion)

        # Evaluate training performance
        train_loss, train_accuracy = evaluate(
            model, data, data.train_mask, criterion
        )

        # Evaluate validation performance
        val_loss, val_accuracy = evaluate(
            model, data, data.val_mask, criterion
        )

        # Record losses and accuracy
        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)


        # Save model when validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(model.state_dict(), best_model_path)

            """
            print(
                f"Best model saved. "
                f"Validation Loss: {best_val_loss:.4f}"
            )
            """

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    # Load the best model after training
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True
        )
    )


    model.eval()

    print("\nBest MLP model loaded.")

    # Evaluate the best saved model on training nodes
    train_loss, train_accuracy = evaluate(
        model, data, data.train_mask, criterion
    )

    # Evaluate the best saved model on validation nodes
    val_loss, val_accuracy = evaluate(
        model, data, data.val_mask, criterion
    )

    print("\n=== BEST MLP MODEL RESULTS ===")
    print(f"Best Epoch: {best_epoch}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")


    # Save training and validation plots
    plot_training_results(
        training_losses,
        validation_losses,
        training_accuracies,
        validation_accuracies
    )

    # Select a confidence threshold using validation data
    threshold = select_confidence_threshold(
        model,
        data,
        target_accuracy=0.90
    )

    # Measure resource requirements
    measure_resources(model, data, device)

    # Final test-set classification evaluation
    evaluate_test_set(model, data)

    # Analyse test-set confidence and manual-review performance
    analyse_confidence(
        model,
        data,
        data.test_mask,
        threshold,
        "Test"
    )

    # Investigate three confidently incorrect predictions
    analyse_failures(model, data, num_examples=5)



if __name__ == "__main__":
    main()
