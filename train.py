import numpy as np
from src.tools import parse_args_train, load, make_learning_curves
from src.nn_tools import seed_everything, init_weights, dataloader, clip_gradients, check_best_model, EarlyStopping
from src.layers import Dense, Dropout
from src.losses import BinaryCrossEntropyLoss, CrossEntropyLoss
from src.activations import ReLU, Tanh, Sigmoid, SoftMax
from src.optimizers import SGD, Momentum, RMSProp, Adam
from src.model import Model

if __name__ == '__main__':
    args = parse_args_train()
    seed_everything(args['seed'])

    X_train, y_train = load('tmp/train.pkl')
    X_val, y_val = load('tmp/val.pkl')

    model = Model()
    model.add_layer(Dense(X_train.shape[1], 100))
    model.add_layer(Dropout(p=0.5))
    model.add_layer(ReLU())
    model.add_layer(Dense(100, 200))
    model.add_layer(Dropout(p=0.5))
    model.add_layer(ReLU())
    model.add_layer(Dense(200, 50))
    model.add_layer(ReLU())
    model.add_layer(Dense(50, 2))

    init_weights(model.params, 'kaiming_normal')

    es = EarlyStopping(esr=30)
    criterion = CrossEntropyLoss()
    optimizer = Momentum(model.params, 1e-4)

    train_acc_by_epoch = []
    val_acc_by_epoch = []
    train_loss_by_epoch = []
    val_loss_by_epoch = []

    for epoch in range(args['epochs']):
        train_loss_by_batch = []
        val_loss_by_batch = []
        train_acc_by_batch = []
        val_acc_by_batch = []

        model.train_mode()
        for x_batch_train, y_batch_train in dataloader(X_train.to_numpy(), y_train.to_numpy(),
                                                       batchsize=args['batchsize'],
                                                       shuffle=True):
            preds = model.forward(x_batch_train)
            loss = criterion(y_batch_train, preds)
            acc = np.mean(np.argmax(preds, axis=1) == y_batch_train)
            train_loss_by_batch.append(loss)
            train_acc_by_batch.append(acc)
            criterion.backward(model, y_batch_train, preds)
            clip_gradients(model.params, 3)
            optimizer.step(epoch + 1)
            model.clear_cache()

        model.test_mode()
        for x_batch_val, y_batch_val in dataloader(X_val.to_numpy(), y_val.to_numpy(),
                                                   batchsize=args['batchsize'],
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

        if check_best_model(val_acc_by_epoch, np.mean(val_acc_by_batch)):
            model.save_weights()
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
    print(f'Best max val accuracy - {max(val_acc_by_epoch)} '
          f'on iteration {val_acc_by_epoch.index(max(val_acc_by_epoch)) + 1}')
    print('Training finished.')
