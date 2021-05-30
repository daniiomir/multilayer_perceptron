import argparse
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.model import Model
from src.layers import Dense
from src.activations import ReLU, SoftMax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD, Momentum, RMSProp, Adam
from src.nn_tools import split_to_train_val_test, seed_everything, init_weights, threshold_prediction, dataloader, \
    check_best_model, clip_gradients
from src.plots import make_learning_curves


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--learning_rate', default=0.00001)
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--esr', default=10)
    parser.add_argument('--batchsize', default=32)
    parser.add_argument('--test_size', default=0.33)
    parser.add_argument('--save_weights_path', default='../tmp/weights.pkl')
    parser.add_argument('--train_data_path', default='../tmp/train.pkl')
    parser.add_argument('--val_data_path', default='../tmp/val.pkl')
    parser.add_argument('--test_data_path', default='../tmp/test.pkl')
    args = parser.parse_args()
    return args.__dict__


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args['seed'])
    encoder = OneHotEncoder(sparse=False)
    x, y = load_iris(return_X_y=True)

    X_train, y_train, X_val, y_val, X_test, y_test = split_to_train_val_test(x, y, args['test_size'], args['seed'])

    y_train = encoder.fit_transform(y_train[:, np.newaxis])
    y_val = encoder.transform(y_val[:, np.newaxis])
    y_test = encoder.transform(y_test[:, np.newaxis])

    model = Model()
    model.add_layer(Dense(X_train.shape[1], 10))
    model.add_layer(ReLU())
    model.add_layer(Dense(10, 10))
    model.add_layer(ReLU())
    model.add_layer(Dense(10, 5))
    model.add_layer(ReLU())
    model.add_layer(Dense(5, 3))
    model.add_layer(SoftMax())

    init_weights(model.params, 'kaiming_normal', 'ones')

    criterion = CrossEntropyLoss()
    optimizer = Momentum(model.params, float(args['learning_rate']))

    print(f'Selected params: {args}')

    train_acc_by_epoch = []
    val_acc_by_epoch = []
    train_loss_by_epoch = []
    val_loss_by_epoch = []

    for epoch in range(int(args['epochs'])):
        train_loss_by_batch = []
        val_loss_by_batch = []
        train_acc_by_batch = []
        val_acc_by_batch = []

        model.train_mode()
        for x_batch_train, y_batch_train in dataloader(X_train, y_train,
                                                       batchsize=int(args['batchsize']),
                                                       shuffle=True):
            preds = model.forward(x_batch_train)
            loss = criterion(y_batch_train, preds)
            acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_batch_train, axis=1))
            train_loss_by_batch.append(loss)
            train_acc_by_batch.append(acc)
            criterion.backward(model, y_batch_train, preds)
            optimizer.step(epoch + 1)
            model.clear_cache()

        model.test_mode()
        for x_batch_val, y_batch_val in dataloader(X_val, y_val,
                                                   batchsize=int(args['batchsize']),
                                                   shuffle=True):
            preds = model.forward(x_batch_val)
            loss = criterion(y_batch_val, preds)
            acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_batch_val, axis=1))
            val_loss_by_batch.append(loss)
            val_acc_by_batch.append(acc)

        print(f'[{epoch + 1}/{args["epochs"]} epoch] '
              f'train loss - {round(np.mean(train_loss_by_batch), 5)}, '
              f'val loss - {round(np.mean(val_loss_by_batch), 5)}, '
              f'train acc - {round(np.mean(train_acc_by_batch), 5)}, '
              f'val acc - {round(np.mean(val_acc_by_batch), 5)}',
              end='')

        if check_best_model(val_loss_by_epoch, np.mean(val_loss_by_batch)):
            model.save_weights(args['save_weights_path'])
            print(' | Best model! Saving weights!', end='')

        print('')

        train_loss_by_epoch.append(np.mean(train_loss_by_batch))
        val_loss_by_epoch.append(np.mean(val_loss_by_batch))
        train_acc_by_epoch.append(np.mean(train_acc_by_batch))
        val_acc_by_epoch.append(np.mean(val_acc_by_batch))

    make_learning_curves(train_loss_by_epoch, val_loss_by_epoch, name='loss')
    make_learning_curves(train_acc_by_epoch, val_acc_by_epoch, name='accuracy')
    print(f'Best min val loss - {min(val_loss_by_epoch)} '
          f'on iteration {val_loss_by_epoch.index(min(val_loss_by_epoch)) + 1}')
    print('Training finished.')

    print('\n' + '=' * 20)
    print('Getting metrics on test data.')
    print('=' * 20 + '\n')

    model.test_mode()
    test_acc = np.mean(np.argmax(model.forward(X_test), axis=1) == np.argmax(y_test, axis=1))
    lr = LogisticRegression(random_state=args['seed']).fit(X_train, np.argmax(y_train, axis=1))
    mlp = MLPClassifier(random_state=args['seed'], max_iter=1000, hidden_layer_sizes=(20, 50)).fit(X_train, np.argmax(y_train, axis=1))

    print('My mlp model accuracy - ', round(test_acc, 4))
    print('Sklearn logistic regression accuracy - ', round(np.mean(lr.predict(X_test) == np.argmax(y_test, axis=1)), 4))
    print('Sklearn mlp accuracy - ', round(np.mean(mlp.predict(X_test) == np.argmax(y_test, axis=1)), 4))
