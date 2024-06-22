import torch
from torchmetrics.metric import Metric


class BalancedAccuracy(Metric):

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.add_state('num_hits', torch.zeros(self.num_classes, dtype=torch.long))
        self.add_state('num_samples', torch.zeros(self.num_classes, dtype=torch.long))

    def update(self, y_hat, y):
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes)
        y_hat_winner = torch.argmax(y_hat, dim=1)
        y_hat_winner_onehot = torch.nn.functional.one_hot(y_hat_winner, self.num_classes)
        self.num_samples = self.num_samples + y_onehot.sum(dim=0)
        self.num_hits = self.num_hits + (y_onehot * y_hat_winner_onehot).sum(dim=0)
        return self.compute()

    def compute(self):
        return torch.sum(self.num_hits / self.num_samples.clamp(min=1)) / torch.sum(torch.sign(self.num_samples))
