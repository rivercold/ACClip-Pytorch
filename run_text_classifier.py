from models.lstm_attn import LSTM_Attn
import argparse
import torch
import torch.nn.functional as F
from utils import load_dataset
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', help='name of the model')
parser.add_argument('--checkpoint', type=str, default='', help='load existing checkpoint')
parser.add_argument('--dataset', type=str, default='sst', help='the dataset used for text classification')
parser.add_argument('--epoch', type=int, default='30', help='The number of epochs')
parser.add_argument('--optimizer', type=str, default='sgd', help='Type of otpimizer')
parser.add_argument('--lr', type=float, default=0.001)
args = parser.parse_args()

def train(model, optimizer, train_batches, dev_batches, total_epoch):
    step = 0
    model.train()
    for epoch in range(total_epoch):
        for batch in train_batches:
            optimizer.zero_grad()
            inputs, target = batch.text.cuda(), batch.label.cuda()
            pred = model(inputs)
            loss = F.cross_entropy(pred, target)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.4f}'.format(step, loss.item()))
                eval_loss, eval_acc = eval(model, dev_batches)
                print ("validation loss {:.4f} and acc {:.4f}".format(eval_loss, eval_acc))

def eval(model, eval_batches):
    model.eval()
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(eval_batches):
            inputs = batch.text.cuda()
            target = batch.label.cuda()
            target = torch.autograd.Variable(target).long()
            prediction = model(inputs)
            loss = F.cross_entropy(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
    model.train()
    return total_epoch_loss/len(eval_batches), total_epoch_acc/len(eval_batches)

batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

if __name__ == "__main__":
    train_batches, dev_batches, test_batches, vocab_size, word_embeddings = load_dataset()

    if args.model == "lstm":
        model = LSTM_Attn(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
        model.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(model, optimizer, train_batches, dev_batches, args.epoch)

