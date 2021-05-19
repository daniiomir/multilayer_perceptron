import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.plots import make_learning_curves
from src.argparser import parse_args_fcn
from src.tools import load, save, LabelEncoder, StandartScaler, most_correlated_features
from src.nn_tools import seed_everything, init_weights, dataloader, clip_gradients, check_best_model, EarlyStopping
from src.layers import Dense, Dropout
from src.losses import BinaryCrossEntropyLoss, CrossEntropyLoss
from src.activations import ReLU, Tanh, Sigmoid, SoftMax
from src.optimizers import SGD, Momentum, RMSProp, Adam
from src.model import Model

if __name__ == '__main__':
    args = parse_args_fcn()
    seed_everything(args['seed'])

    if args['preprocess'] == 'yes':
        dataset = pd.read_csv(args['dataset_path'])
        encoder = LabelEncoder()
        scaler = StandartScaler()

        dataset = pd.DataFrame(data=dataset.to_numpy(), columns=[str(i + 1) for i in range(dataset.shape[1])])

        labels = encoder.fit_transform(dataset['2'])
        dataset.drop(columns=['1', '2'], inplace=True, axis=1)
        dataset = dataset.astype(np.float64)
        dataset['labels'] = labels

        drop_features = most_correlated_features(dataset.drop(['labels'], axis=1), 0.9)

        print(dataset.describe().to_string())
        print('\nMost correlated features (over 0.9) - ', drop_features)
        print('\nBefore removing trash - ', dataset.shape)
        dataset.drop(columns=drop_features, axis=1, inplace=True)
        dataset = dataset.dropna()
        print('After removing trash - ', dataset.shape)

        y = dataset['labels']
        x = dataset.drop(columns=['labels'], axis=1)

        x = scaler.fit_transform(x)

        X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=float(args['test_size']),
                                                                      random_state=int(args['seed']))

        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=float(args['test_size']),
                                                          random_state=int(args['seed']))
        save((X_train, y_train), args['train_data_path'])
        save((X_val, y_val), args['val_data_path'])
        save((X_test, y_test), args['test_data_path'])
        save(encoder, '../tmp/encoder.pkl')

        print('\nDataset prepossessing finished.')

    else:
        X_train, y_train = load(args['train_data_path'])
        X_val, y_val = load(args['val_data_path'])
        X_test, y_test = load(args['test_data_path'])

    model = Model()
    model.add_layer(Dense(X_train.shape[1], 20))
    model.add_layer(Dropout())
    model.add_layer(ReLU())
    model.add_layer(Dense(20, 50))
    model.add_layer(Dropout())
    model.add_layer(ReLU())
    model.add_layer(Dense(50, 10))
    model.add_layer(ReLU())
    model.add_layer(Dropout())
    model.add_layer(Dense(10, 2))

    init_weights(model.params, 'kaiming_normal')

    es = EarlyStopping(esr=int(args['esr']))
    criterion = CrossEntropyLoss()
    optimizer = Momentum(model.params, float(args['learning_rate']))

    train_acc_by_epoch = []
    val_acc_by_epoch = []
    train_loss_by_epoch = []
    val_loss_by_epoch = []

    print(f'Selected params: {args}')

    for epoch in range(int(args['epochs'])):
        train_loss_by_batch = []
        val_loss_by_batch = []
        train_acc_by_batch = []
        val_acc_by_batch = []

        model.train_mode()
        for x_batch_train, y_batch_train in dataloader(X_train.to_numpy(), y_train.to_numpy(),
                                                       batchsize=int(args['batchsize']),
                                                       shuffle=True):
            preds = model.forward(x_batch_train)
            loss = criterion(y_batch_train, preds)
            acc = np.mean(np.argmax(preds, axis=1) == y_batch_train)
            train_loss_by_batch.append(loss)
            train_acc_by_batch.append(acc)
            criterion.backward(model, y_batch_train, preds)
            clip_gradients(model.params, float(args['clip_grad']))
            optimizer.step(epoch + 1)
            model.clear_cache()

        model.test_mode()
        for x_batch_val, y_batch_val in dataloader(X_val.to_numpy(), y_val.to_numpy(),
                                                   batchsize=int(args['batchsize']),
                                                   shuffle=True):
            preds = model.forward(x_batch_val)
            loss = criterion(y_batch_val, preds)
            acc = np.mean(np.argmax(preds, axis=1) == y_batch_val)
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

        es.add_loss(np.mean(val_acc_by_batch))

        if es.check_stop_training():
            break

    make_learning_curves(train_loss_by_epoch, val_loss_by_epoch, name='loss')
    make_learning_curves(train_acc_by_epoch, val_acc_by_epoch, name='accuracy')
    print(f'Best min val loss - {min(val_loss_by_epoch)} '
          f'on iteration {val_loss_by_epoch.index(min(val_loss_by_epoch)) + 1}')
    print('Training finished.')

    print('\n' + '=' * 20)
    print('Getting metrics on test data.')
    print('=' * 20 + '\n')
    model.test_mode()
    preds = model.forward(X_test.to_numpy())
    acc = np.mean(np.argmax(preds, axis=1) == y_test)
    print('Model accuracy on test data - ', round(acc, 4))
