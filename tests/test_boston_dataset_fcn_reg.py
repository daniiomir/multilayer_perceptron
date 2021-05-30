import argparse
import numpy as np

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from src.model import Model
from src.layers import Dense
from src.activations import ReLU
from src.losses import MeanSquaredErrorLoss
from src.optimizers import SGD, Momentum, RMSProp, Adam
from src.nn_tools import split_to_train_val_test, seed_everything, init_weights, threshold_prediction, dataloader, \
    check_best_model
from src.plots import make_learning_curves


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--learning_rate', default=0.000001)
    parser.add_argument('--epochs', default=1000)
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
    scaler = StandardScaler()

    x, y = load_boston(return_X_y=True)

    x = scaler.fit_transform(x)

    X_train, y_train, X_val, y_val, X_test, y_test = split_to_train_val_test(x, y, args['test_size'], args['seed'])

    model = Model()
    model.add_layer(Dense(X_train.shape[1], 20))
    model.add_layer(ReLU())
    model.add_layer(Dense(20, 20))
    model.add_layer(ReLU())
    model.add_layer(Dense(20, 10))
    model.add_layer(ReLU())
    model.add_layer(Dense(10, 1))

    init_weights(model.params, 'kaiming_normal', 'ones')

    criterion = MeanSquaredErrorLoss()
    optimizer = SGD(model.params, float(args['learning_rate']))

    print(f'Selected params: {args}')

    train_loss_by_epoch = []
    val_loss_by_epoch = []

    for epoch in range(int(args['epochs'])):
        train_loss_by_batch = []
        val_loss_by_batch = []

        model.train_mode()
        for x_batch_train, y_batch_train in dataloader(X_train, y_train,
                                                       batchsize=int(args['batchsize']),
                                                       shuffle=True):
            preds = model.forward(x_batch_train)
            loss = criterion(y_batch_train, preds)
            train_loss_by_batch.append(loss)
            criterion.backward(model, y_batch_train, preds)
            optimizer.step(epoch + 1)
            model.clear_cache()

        model.test_mode()
        for x_batch_val, y_batch_val in dataloader(X_val, y_val,
                                                   batchsize=int(args['batchsize']),
                                                   shuffle=True):
            preds = model.forward(x_batch_val)
            loss = criterion(y_batch_val, preds)
            val_loss_by_batch.append(loss)

        print(f'[{epoch + 1}/{args["epochs"]} epoch] '
              f'train loss - {round(np.mean(train_loss_by_batch), 5)}, '
              f'val loss - {round(np.mean(val_loss_by_batch), 5)}, ',
              end='')

        if check_best_model(val_loss_by_epoch, np.mean(val_loss_by_batch)):
            model.save_weights(args['save_weights_path'])
            print(' | Best model! Saving weights!', end='')

        print('')

        train_loss_by_epoch.append(np.mean(train_loss_by_batch))
        val_loss_by_epoch.append(np.mean(val_loss_by_batch))

    make_learning_curves(train_loss_by_epoch, val_loss_by_epoch, name='loss')
    print(f'Best min val loss - {min(val_loss_by_epoch)} '
          f'on iteration {val_loss_by_epoch.index(min(val_loss_by_epoch)) + 1}')
    print('Training finished.')

    print('\n' + '=' * 20)
    print('Getting metrics on test data.')
    print('=' * 20 + '\n')

    model.test_mode()
    mse_error = mean_squared_error(y_test, model.forward(X_test))
    lr = LinearRegression().fit(X_train, y_train)
    mlp = MLPRegressor(random_state=args['seed'], max_iter=1000, hidden_layer_sizes=(20, 20)).fit(X_train, y_train)

    print('My mlp model mse - ', round(mse_error, 4))
    print('Sklearn linear regression mse - ', round(mean_squared_error(y_test, lr.predict(X_test)), 4))
    print('Sklearn mlp mse - ', round(mean_squared_error(y_test, mlp.predict(X_test)), 4))
