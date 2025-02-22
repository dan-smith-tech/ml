import matplotlib
import torch
from matplotlib import pyplot as plt
from torch import nn

matplotlib.use("module://matplotlib-backend-kitty")

# create known parameters
weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02

# make a straight line with linear regression
X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegression()

# how well does the model predict y_test based on x_test
with torch.inference_mode():
    # inference mode disables gradient tracking, to save memory
    y_preds = model_0(X_test)

# plot the predictions
plot_original = plot_predictions(
    train_data=X_train,
    train_labels=y_train,
    test_data=X_test,
    test_labels=y_test,
    predictions=y_preds.detach().numpy(),
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
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")

plot_final = plot_predictions(predictions=test_pred)

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
