
import torch
from torch import nn

from dataset import load_dataset
from modules import MLP


# Training settings
SEED = 42
LEARNING_RATE = 0.01
EPOCHS = 1000
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

    # Train the MLP baseline
    for epoch in range(1, EPOCHS + 1):

        train_loss = train_epoch(
            model, data, optimizer, criterion
        )

        val_loss, val_accuracy = evaluate(
            model, data, data.val_mask, criterion
        )

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



if __name__ == "__main__":
    main()
