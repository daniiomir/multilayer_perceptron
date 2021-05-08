import numpy as np
from src.tools import parse_args_train, load
from src.nn_tools import seed_everything, init_weights, dataloader
from src.layers import Dense
from src.losses import BinaryCrossEntropyLoss, CrossEntropyLoss
from src.activations import ReLU, Tanh, Sigmoid, SoftMax
from src.optimizers import SGD, Momentum
from src.model import Model

if __name__ == '__main__':
    args = parse_args_train()
    seed_everything(args['seed'])
    train_log = []
    val_log = []
    loss_by_epoch = []

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
    model.add_layer(SoftMax())

    init_weights(model.params, 'kaiming_normal')

    criterion = CrossEntropyLoss()
    optimizer = Momentum(model.params, 0.1)

    for epoch in range(args['epochs']):
        loss_by_batch = []
        for x_batch, y_batch in dataloader(X_train.to_numpy(), y_train.to_numpy(), batchsize=args['batchsize'],
                                           shuffle=True):
            preds = model.forward(x_batch, mode='train')
            loss = criterion(y_batch, preds)
            loss_by_batch.append(loss)
            criterion.backward(model, y_batch, preds)
            optimizer.step()
            model.clear_cache()

        train_acc = np.mean(np.argmax(model.forward(X_train), axis=1) == y_train)
        val_acc = np.mean(np.argmax(model.forward(X_val), axis=1) == y_val)

        print(f'[{epoch + 1} epoch] loss - {round(np.mean(loss_by_batch), 5)}, '
              f'train acc - {round(train_acc, 5)}, '
              f'val acc - {round(val_acc, 5)}')

        train_log.append(train_acc)
        val_log.append(val_acc)
        loss_by_epoch.append(np.mean(loss_by_batch))

    print('Training finished.')
