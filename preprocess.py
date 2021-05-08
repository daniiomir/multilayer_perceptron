import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.tools import LabelEncoder, StandartScaler, parse_args_preprocess, most_correlated_features, make_corr_heatmap, \
    make_count_plot, save

if __name__ == '__main__':
    args = parse_args_preprocess()
    dataset = pd.read_csv(args['dataset_path'])
    encoder = LabelEncoder()
    scaler = StandartScaler()

    dataset = pd.DataFrame(data=dataset.to_numpy(), columns=[str(i + 1) for i in range(dataset.shape[1])])

    make_count_plot(dataset, dataset['2'])

    labels = encoder.fit_transform(dataset['2'])
    dataset.drop(columns=['1', '2'], inplace=True, axis=1)
    dataset = dataset.astype(np.float64)
    dataset['labels'] = labels

    drop_features = most_correlated_features(dataset.drop(['labels'], axis=1), 0.9)

    print(dataset.describe().to_string())
    make_corr_heatmap(dataset)
    print('\nMost correlated features (over 0.9) - ', drop_features)

    print('\nBefore removing trash - ', dataset.shape)
    dataset.drop(columns=drop_features, axis=1, inplace=True)
    dataset = dataset.dropna()
    print('After removing trash - ', dataset.shape)

    y = dataset['labels']
    x = dataset.drop(columns=['labels'], axis=1)

    x = scaler.fit_transform(x)

    X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=args['test_size'],
                                                                  random_state=args['seed'])

    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=args['test_size'],
                                                      random_state=args['seed'])

    save((X_train, y_train), 'tmp/train.pkl')
    save((X_val, y_val), 'tmp/val.pkl')
    save((X_test, y_test), 'tmp/test.pkl')
    save(encoder, 'tmp/encoder.pkl')

    print('\nDataset prepossessing finished.')
