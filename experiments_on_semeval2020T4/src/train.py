#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from data_utils import SemEval20Dataset


def main():
    train_path = '../dataset/train.csv'
    dev_path = '../dataset/dev.csv'
    train_dataset = SemEval20Dataset(train_path)
    dev_dataset = SemEval20Dataset(dev_path)

    X_train = train_dataset.get_embeddings_all()
    y_train = np.asarray(train_dataset.labels)
    X_dev = dev_dataset.get_embeddings_all()
    y_dev = np.asarray(dev_dataset.labels)

    # LR
    LR = LogisticRegression(random_state=0, max_iter=1000, C=0.1)
    LR.fit(X_train, y_train)
    # pred_labels = LR.predict(X_dev)
    # print(pred_labels)
    print("LR score: ")
    print(LR.score(X_dev, y_dev))

    # SVC
    SVC = LinearSVC(random_state=0, max_iter=1000, C=1)
    SVC.fit(X_train, y_train)
    # pred_labels = SVC.predict(X_dev)
    # print(pred_labels)
    print("SVC score: ")
    print(SVC.score(X_dev, y_dev))


if __name__ == '__main__':
    main()
