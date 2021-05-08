import numpy as np
from src.tools import parse_args_train, load, make_learning_curves
from src.nn_tools import seed_everything, init_weights, dataloader
from src.layers import Dense
from src.losses import BinaryCrossEntropyLoss, CrossEntropyLoss
from src.activations import ReLU, Tanh, Sigmoid, SoftMax
from src.optimizers import SGD, Momentum
from src.model import Model

if __name__ == '__main__':
    args = parse_args_train()
    seed_everything(args['seed'])

    X_train, y_train = load('tmp/train.pkl')
    X_val, y_val = load('tmp/val.pkl')

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    # m = LogisticRegression().fit(X_train, y_train)
    # print(accuracy_score(y_val, m.predict(X_val)))

    model = Model()
    model.add_layer(Dense(X_train.shape[1], 100))
    model.add_layer(ReLU())
    model.add_layer(Dense(100, 200))
    model.add_layer(ReLU())
    model.add_layer(Dense(200, 50))
    model.add_layer(ReLU())
    model.add_layer(Dense(50, 2))
    # model.add_layer(SoftMax())

    init_weights(model.params, 'kaiming_normal')

    criterion = CrossEntropyLoss()
    optimizer = Momentum(model.params, 1e-4, grad_clip=3)

    train_acc_by_epoch = []
    val_acc_by_epoch = []

    train_loss_by_epoch = []
    val_loss_by_epoch = []

    for epoch in range(args['epochs']):
        train_loss_by_batch = []
        val_loss_by_batch = []

        for x_batch, y_batch in dataloader(X_train.to_numpy(), y_train.to_numpy(), batchsize=args['batchsize'],
                                           shuffle=True):
            preds = model.forward(x_batch, mode='train')
            loss = criterion(y_batch, preds)
            train_loss_by_batch.append(loss)
            criterion.backward(model, y_batch, preds)
            optimizer.step()
            model.clear_cache()

        for x_batch_val, y_batch_val in dataloader(X_val.to_numpy(), y_val.to_numpy(), batchsize=args['batchsize'],
                                           shuffle=True):
            preds = model.forward(x_batch_val)
            loss = criterion(y_batch_val, preds)
            val_loss_by_batch.append(loss)

        train_loss_by_epoch.append(np.mean(train_loss_by_batch))
        val_loss_by_epoch.append(np.mean(val_loss_by_batch))

        train_acc = np.mean(np.argmax(model.forward(X_train), axis=1) == y_train)
        val_acc = np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)

        train_acc_by_epoch.append(train_acc)
        val_acc_by_epoch.append(val_acc)

        print(f'[{epoch + 1}/{args["epochs"]} epoch] '
              f'train loss - {round(np.mean(train_loss_by_batch), 5)}, '
              f'val loss - {round(np.mean(val_loss_by_batch), 5)}, '
              f'train acc - {round(train_acc, 5)}, '
              f'val acc - {round(val_acc, 5)}')

    make_learning_curves(train_loss_by_epoch, val_loss_by_epoch, name='loss')
    make_learning_curves(train_acc_by_epoch, val_acc_by_epoch, name='accuracy')

    print('Training finished.')
