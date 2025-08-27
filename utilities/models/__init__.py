from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from .nn import NN, CNN
import torch
from utilities.visualization.plot_evaluation import plot_change_along_epoches
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_model(model_name, **kwargs):
    """
    Return the model object based on the model name
    :param model_name: name of the model
    :param kwargs: parameters of the model
    :return: model object
    """
    if model_name == 'logistic_regression':
        return LogisticRegression(**kwargs)
    elif model_name == 'random_forest':
        return RandomForestClassifier(**kwargs)
    elif model_name == 'svm':
        return SVC(**kwargs)
    elif model_name == 'knn':
        return KNeighborsClassifier(**kwargs)
    elif model_name == 'nn':
        return NN(**kwargs)
    elif model_name == 'cnn':
        return CNN(**kwargs).to(device)
    else:
        raise ValueError('Model name {} is not supported'.format(model_name))


class Model:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.args = kwargs
        self.model = get_model(model_name, **kwargs)

        self.loss = []
        self.accuracy = []

    def fit(self, X, y):
        self.model.fit(X, y)

    def train_loop(self, dataloader, optimizer, loss_fn):
        """
        Train the model on the data.
        :param data: dictionary of dataframes
        :param label: dictionary of labels
        :return: None
        """
        if self.model_name == 'nn' or self.model_name == 'cnn':
            size = len(dataloader.dataset)
            self.model.train()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                loss = loss_fn(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(self, dataloader, loss_fn):
        if self.model_name == 'nn' or self.model_name == 'cnn':
            size = len(dataloader.dataset)
            num_batches = len(dataloader)
            test_loss, correct = 0, 0
            self.model.eval()
            with torch.no_grad():
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = self.model(X)
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
            self.accuracy.append(correct)
            self.loss.append(test_loss)

    def training_process(self):
        fig = plot_change_along_epoches(self.accuracy, self.loss)
        fig.show()
        return fig

    def step_count(self, dataloader):
        pred_list = []
        target_list = []
        for X, y in dataloader:
            X, y = X.to("cuda"), y.to("cuda")
            # pred = self.model(X)
            pred = self.model(X)
            pred = pred.argmax(1)
            pred_list.append(pred)
            target_list.append(y.argmax(1))
        pred = torch.cat(pred_list).cpu().numpy()
        y = torch.cat(target_list).cpu().numpy()
        steps = np.sum(np.abs(pred[1:] - pred[:-1]))
        real_steps = np.sum(np.abs(y[1:] - y[:-1]))
        return steps, real_steps

    def save(self, path):
        if self.model_name == 'nn' or self.model_name == 'cnn':
            torch.save(self.model.state_dict(), path)


    def get_params(self):
        return self.model.get_params()
