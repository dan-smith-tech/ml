import matplotlib
import torch
from matplotlib import pyplot as plt
from torch import nn

matplotlib.use("module://matplotlib-backend-kitty")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# create known parameters
WEIGHT = 0.7
BIAS = 0.3

# make a straight line with linear regression
X = torch.arange(0, 1, 0.02).unsqueeze(1)
y = WEIGHT * X + BIAS

# create train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


# visualize the data
def plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=None,
):
    """
    Plots training data, test data and compares predictions.
    """

    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot testing data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

    return plt


# Create linear regression model class
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1, requires_grad=True, dtype=torch.float)
        )
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float))

        # note: can also use nn.Linear(1, 1) instead of weights and bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

        # note: can also use self.linear_layer.forward(x)


torch.manual_seed(42)
model_0 = LinearRegression()
model_0.to(DEVICE)

# put data on the same device as the model
X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

# how well does the model predict y_test based on x_test
with torch.inference_mode():
    # inference mode disables gradient tracking, to save memory
    y_preds = model_0(X_test)

# plot the predictions
plot_original = plot_predictions(
    train_data=X_train.cpu(),
    train_labels=y_train.cpu(),
    test_data=X_test.cpu(),
    test_labels=y_test.cpu(),
    predictions=y_preds.cpu().detach().numpy(),
)

# setup loss function
loss_fn = nn.L1Loss()

# setup SGD optimiser
optimiser = torch.optim.SGD(model_0.parameters(), lr=0.01)

torch.manual_seed(42)
# an epoch is one complete pass through the data
epochs = 1000


epoch_count = []
loss_values = []
test_loss_values = []

# Training
for epoch in range(epochs):
    # set model to training mode
    # (sets all parameters in mode that require gradients, to tack gradients)
    model_0.train()

    # forward pass
    y_pred = model_0(X_train)

    # calculate loss
    loss = loss_fn(y_pred, y_train)

    # optimiser zero gradients
    # (this stops gradients accumulating across all epochs)
    optimiser.zero_grad()

    # backward pass
    loss.backward()

    # perform gradient descent
    optimiser.step()

    # turns off things not needed for evaluation
    model_0.eval()

    with torch.inference_mode():
        # forward pass
        test_pred = model_0(X_test)

        # calculate loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss.cpu())
        test_loss_values.append(test_loss.cpu())
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

plot_final = plot_predictions(predictions=test_pred.cpu().detach().numpy())

plot_original.show()
plot_final.show()

plot_train_loss_curves = plt.plot(
    epoch_count, torch.tensor(loss_values).cpu().numpy(), label="Train loss"
)
plot_test_loss_curves = plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# save the model
from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "01-linear-regression.pt"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# torch.save(model_0.state_dict(), MODEL_SAVE_PATH)

# load the model (parameters / state_dict)
model_0_loaded = LinearRegression()
model_0_loaded.to(DEVICE)
model_0_loaded.load_state_dict(torch.load(MODEL_SAVE_PATH))
model_0_loaded.eval()
with torch.inference_mode():
    y_preds_loaded = model_0_loaded(X_test)

plot_loaded = plot_predictions(
    train_data=X_train.cpu(),
    train_labels=y_train.cpu(),
    test_data=X_test.cpu(),
    test_labels=y_test.cpu(),
    predictions=y_preds_loaded.cpu().detach().numpy(),
)
plot_loaded.show()
