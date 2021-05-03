import pickle
import numpy as np
from src.tools import parse_args_train
from src.layers import Dense
from src.losses import BinaryCrossEntropy
from src.activations import ReLU, SoftMax
from src.optimizers import SGD
from src.model import Model

if __name__ == '__main__':
    args = parse_args_train()
    train_log = []
    val_log = []

    with open('tmp/train.pkl', 'r') as f:
        X_train, y_train = pickle.load(f)

    with open('tmp/val.pkl', 'r') as f:
        X_val, y_val = pickle.load(f)

    model = Model()
    model.add_layer(Dense(X_train.shape[1], 100))
    model.add_layer(ReLU())
    model.add_layer(Dense(100, 200))
    model.add_layer(ReLU())
    model.add_layer(Dense(100, 200))
    model.add_layer(ReLU())
    model.add_layer(Dense(200, 2))
    model.add_layer(SoftMax())

    criterion = BinaryCrossEntropy()
    optimizer = SGD()

    for epoch in range(args['epochs']):
        for x_batch, y_batch in model.loader(X_train, y_train, batchsize=args['batchsize'], shuffle=True):
            preds = model.forward(x_batch)
            layer_inputs = [x_batch] + model.forward_list

            loss = criterion(y_batch, preds)

            loss_grad = criterion.backward(y_batch, preds)
            model.backward(layer_inputs, loss_grad)
            optimizer.step()

        train_acc = np.mean(np.argmax(model.forward(X_train), axis=-1) == y_train)
        val_acc = np.mean(np.argmax(model.forward(X_val)) == y_val)

        train_log.append(train_acc)
        val_log.append(val_acc)
