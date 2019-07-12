#-*- coding:utf-8 -*-

import numpy as np
import fire
from tqdm import tqdm, trange
from util import Vocab
from model import SoftmaxRegressor

def train(num_epochs, train_batch_size, save_path,
        lr=0.01, num_labels=5, early_stop_epoch=5,
        log_interval=200):
    vocab = Vocab()
    model = SoftmaxRegressor(num_labels=5, lr=0.01)
    min_loss = 10000
    tmp_epoch = 0
    for epoch in trange(num_epochs, desc="Epoch"):
        losses = []
        for lidx,(X, y) in enumerate(vocab.get_batch()):
            y_pred, loss = model(X, y)
            losses.append(loss)
            model.backward(X, y, y_pred)
            if lidx % log_interval == 0:
                print("iter : {} and the loss : {}".format(lidx, loss))
        mean_error = np.mean(np.array(losses))
        print('Epoch: {} and mean loss : {}'.format(epoch, mean_error))
        if mean_error >= min_loss:
            tmp_epoch += 1
        else:
            min_loss = mean_error
        if tmp_epoch >= early_stop_epoch:
            break
    model.save(save_path)


def predict(model, test_batch_size=256, save_path='./predict.txt'):
    vocab = Vocab('./data/test.tsv', 'test')
    all_predicts = []
    for X, _ in vocab.get_batch(test_batch_size):
        predict = model.predict(X)
        all_predicts += predict.tolist()
    with open(save_path, 'w') as f:
        for predict in all_predicts:
            f.write(str(predict) + '\n')
    

if __name__ == "__main__":
    train(num_epochs=1000, train_batch_size=256, save_path='./trained_Wb')

