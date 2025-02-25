import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

matplotlib.use("module://matplotlib-backend-kitty")

N_SAMPLES = 1000

# this is a toy dataset: small and simple, but enough to experiment with
X, y = make_circles(N_SAMPLES, noise=0.03, random_state=42)


circles = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
print(circles.head())


plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()


# input/output shapes

# turn X and y into PyTorch tensors (from numpy arrays)
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))


model_v0 = CircleModelV0().to(DEVICE)

model_v0 = nn.Sequential(
    nn.Linear(2, 5),
    nn.Linear(5, 1),
).to(DEVICE)


loss_fn = nn.BCEWithLogitsLoss()  # same as nn.BCELoss() but more stable
optimizer = torch.optim.SGD(model_v0.parameters(), lr=0.1)


# Calculate accuracy
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


# the raw outputs of the model that are not passed through an activation function are logits
# (we need to convert those into probabilities - use the sigmoid function)

with torch.inference_mode():
    y_logits = model_v0(X_test.to(DEVICE))
y_pred_probs = torch.sigmoid(y_logits)
y_preds = torch.round(y_pred_probs)
print(y_preds[:5])
