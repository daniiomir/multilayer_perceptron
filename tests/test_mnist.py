import argparse


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42)
    parser.add_argument('--preprocess', default='no')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--esr', default=10)
    parser.add_argument('--clip_grad', default=3.)
    parser.add_argument('--batchsize', default=32)
    parser.add_argument('--test_size', default=0.33)
    parser.add_argument('--dataset_folder', default='../data/mnist/')
    parser.add_argument('--save_weights_path', default='../tmp/weights.pkl')
    parser.add_argument('--train_data_path', default='../tmp/train.pkl')
    parser.add_argument('--val_data_path', default='../tmp/val.pkl')
    parser.add_argument('--test_data_path', default='../tmp/test.pkl')
    args = parser.parse_args()
    return args.__dict__


if __name__ == '__main__':
    pass
