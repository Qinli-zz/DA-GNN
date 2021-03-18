import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.last_val_loss = None

    def __call__(self, val_loss, model):

        # score = -val_loss
        if self.best_score is None:
            score = 0
            self.best_score = score
            self.last_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
        else:
            score = 0
            for n in range(25):
                if val_loss[n] < self.last_val_loss[n]:
                    score += 1

            if score < self.best_score + self.delta:
                self.counter += 1
                print('EarlyStopping counter: counter{0:d} out of patience{1:d}'.format(self.counter, self.patience))
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0


    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        val_loss_mean = np.mean(val_loss)
        if self.verbose:
            print('Validation loss decreased ({0:.6f} --> {1:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss_mean))
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss_mean
