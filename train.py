import numpy as np
from src.tools import parse_args_train, load, make_learning_curves
from src.nn_tools import seed_everything, init_weights, dataloader, clip_gradients
from src.layers import Dense
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
    model.add_layer(ReLU())
    model.add_layer(Dense(100, 200))
    model.add_layer(ReLU())
    model.add_layer(Dense(200, 50))
    model.add_layer(ReLU())
    model.add_layer(Dense(50, 2))

    init_weights(model.params, 'kaiming_normal')

    criterion = CrossEntropyLoss()
    optimizer = Momentum(model.params, 1e-4)
    # optimizer = Adam(model.params, 1e-6)
    train_acc_by_epoch = []
    val_acc_by_epoch = []

    train_loss_by_epoch = []
    val_loss_by_epoch = []

    for epoch in range(args['epochs']):
        train_loss_by_batch = []
        val_loss_by_batch = []
        train_acc_by_batch = []
        val_acc_by_batch = []

        for x_batch, y_batch in dataloader(X_train.to_numpy(), y_train.to_numpy(), batchsize=args['batchsize'],
                                           shuffle=True):
            preds = model.forward(x_batch, mode='train')
            loss = criterion(y_batch, preds)
            acc = np.mean(np.argmax(preds, axis=1) == y_batch)
            train_loss_by_batch.append(loss)
            train_acc_by_batch.append(acc)
            criterion.backward(model, y_batch, preds)
            clip_gradients(model.params, 3)
            optimizer.step(epoch + 1)
            model.clear_cache()

        for x_batch_val, y_batch_val in dataloader(X_val.to_numpy(), y_val.to_numpy(), batchsize=args['batchsize'],
                                           shuffle=True):
            preds = model.forward(x_batch_val)
            loss = criterion(y_batch_val, preds)
            acc = np.mean(np.argmax(preds, axis=1) == y_batch_val)
            val_loss_by_batch.append(loss)
            val_acc_by_batch.append(acc)

        train_loss_by_epoch.append(np.mean(train_loss_by_batch))
        val_loss_by_epoch.append(np.mean(val_loss_by_batch))
        train_acc_by_epoch.append(np.mean(train_acc_by_batch))
        val_acc_by_epoch.append(np.mean(val_acc_by_batch))

        print(f'[{epoch + 1}/{args["epochs"]} epoch] '
              f'train loss - {round(np.mean(train_loss_by_batch), 5)}, '
              f'val loss - {round(np.mean(val_loss_by_batch), 5)}, '
              f'train acc - {round(np.mean(train_acc_by_batch), 5)}, '
              f'val acc - {round(np.mean(val_acc_by_batch), 5)}')

    make_learning_curves(train_loss_by_epoch, val_loss_by_epoch, name='loss')
    make_learning_curves(train_acc_by_epoch, val_acc_by_epoch, name='accuracy')
    print(f'Best max val accuracy - {max(val_acc_by_epoch)}')
    print('Training finished.')
