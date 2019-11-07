from models.lstm_attn import LSTM_Attn
import argparse
import torch
import torch.nn.functional as F
from utils import load_dataset
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='pcnn_att', help='name of the model')
parser.add_argument('--checkpoint', type=str, default='', help='load existing checkpoint')
parser.add_argument('--dataset', type=str, default='sst', help='the dataset used for text classification')
parser.add_argument('--epoch', type=int, default='30', help='The number of epochs')
parser.add_argument('--optimizer', type=str, default='sgd', help='Type of otpimizer')
parser.add_argument('--lr', type=float, default=0.05)
args = parser.parse_args()

def train(model, optimizer, train_batches, total_epoch):
    step = 0
    for epoch in range(total_epoch):
        for batch in train_batches:
            optimizer.zero_grad()
            features, target = batch.text.cuda(), batch.label.cuda()
            pred = model(features)
            loss = F.cross_entroy(pred, target)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.4f}'.format(step, loss.item()))

if __name__ == "__main___":
    if args.model == "lstm":
        model = LSTM_Attn()
        model.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    train_batches, dev_batches, test_batches = load_dataset()

    train(model, optimizer, train_batches, args.epoch)

