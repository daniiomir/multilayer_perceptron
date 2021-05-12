import numpy as np
from src.tools import parse_args_test, load
from src.model import Model
from src.layers import Dense, Dropout
from src.activations import ReLU

if __name__ == '__main__':
    args = parse_args_test()

    X_test, y_test = load('tmp/test.pkl')

    model = Model()
    model.add_layer(Dense(X_test.shape[1], 100))
    model.add_layer(Dropout(p=0.5))
    model.add_layer(ReLU())
    model.add_layer(Dense(100, 200))
    model.add_layer(Dropout(p=0.5))
    model.add_layer(ReLU())
    model.add_layer(Dense(200, 50))
    model.add_layer(ReLU())
    model.add_layer(Dense(50, 2))

    model.load_weights('tmp/weights.pkl')

    model.test_mode()
    preds = model.forward(X_test.to_numpy())
    acc = np.mean(np.argmax(preds, axis=1) == y_test)
    print('Model accuracy on test data - ', round(acc, 4))
