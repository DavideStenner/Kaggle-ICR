import numpy as np

def calc_log_loss_weight(y_true):
    nc = np.bincount(y_true)
    w0, w1 = 1/(nc[0]/y_true.shape[0]), 1/(nc[1]/y_true.shape[0])
    return w0, w1

def lgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'balanced_log_loss', competition_log_loss(y_true, y_pred), False

def competition_log_loss(y_true, y_pred, return_all: bool = False):    
    #  _ _ _ The competition log loss - class weighted _ _ _
    # The Equation on the Evaluation page is the competition log loss
    # provided w_0=1 and w_1=1.
    # That is, the weights shown in the equation
    # are in addition to class balancing.
    # For this case:
    # _ A constant y_pred = 0.5 will give loss = 0.69 for any class ratio.
    # _ Predicting the observed training ratio, p_1 = 0.17504
    #   will give loss ~ 0.96 for any class ratio.
    #   This is confirmed by the score for this notebook:
    #   https://www.kaggle.com/code/avtandyl/simple-baseline-mean 
    # y_true: correct labels 0, 1
    # y_pred: predicted probabilities of class=1
    # Implements the Evaluation equation with w_0 = w_1 = 1.
    # Calculate the number of observations for each class
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    # Calculate the predicted probabilities for each class
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    # Calculate the average log loss for each class
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1
    # return the (not further weighted) average of the averages
    if return_all:
        return (log_loss_0 + log_loss_1)/2, log_loss_0, log_loss_1
    else:
        return (log_loss_0 + log_loss_1)/2