from models.lstm_attn import LSTM_Attn
import argparse
import torch
import torch.nn.functional as F
from utils import load_dataset
import sys, os, pickle
from optimizers.ACClip import ACClip

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lstm', help='name of the model')
parser.add_argument('--checkpoint', type=str, default='', help='load existing checkpoint')
parser.add_argument('--dataset', type=str, default='imdb', help='the dataset used for text classification')
parser.add_argument('--epoch', type=int, default='20', help='The number of epochs')
parser.add_argument('--optimizer', type=str, default='sgd', help='Type of otpimizer')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--emb_dim', type=int, default=300)
args = parser.parse_args()

def train(model, optimizer, train_batches, eval_batches, total_epoch, model_name):
    step = 0
    model.train()
    train_loss, test_loss, test_acc = [], [], []
    for epoch in range(total_epoch):
        total_loss = 0.0
        for batch in train_batches:
            optimizer.zero_grad()
            inputs, target = batch.text, batch.label
            if torch.cuda.is_available():
                inputs, target = inputs.cuda(), target.cuda()
            pred = model(inputs)
            loss = F.cross_entropy(pred, target)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            step += 1
            if step % 100 == 0:
                sys.stdout.write('\rBatch[{}] - loss: {:.4f}\n'.format(step, loss.item()))
        train_loss.append(total_loss/len(train_batches))
        eval_loss, eval_acc = eval(model, eval_batches)
        test_loss.append(eval_loss)
        test_acc.append(eval_acc)
        print ("validation loss {:.4f} and acc {:.4f}".format(eval_loss, eval_acc))

        if not os.path.exists("./curves"):
            os.mkdir("./curves")
        with open(os.path.join('./curves', model_name), "wb") as f:
            pickle.dump({'train_loss': train_loss, 'test_loss': test_loss, 'test_acc': test_acc}, f)

def eval(model, eval_batches):
    model.eval()
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(eval_batches):
            inputs = batch.text
            target = batch.label
            if torch.cuda.is_available():
                inputs, target = inputs.cuda(), target.cuda()
            target = torch.autograd.Variable(target).long()
            prediction = model(inputs)
            loss = F.cross_entropy(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()
    model.train()
    return total_epoch_loss/len(eval_batches), total_epoch_acc/len(eval_batches)

if __name__ == "__main__":
    print ("Start loading and processing dataset")
    train_batches, dev_batches, test_batches, class_num, vocab_size, word_embeds = load_dataset(word_dim=args.emb_dim)

    if args.model == "lstm":
        model = LSTM_Attn(args.batch_size, class_num, args.hidden_size, vocab_size, args.emb_dim, word_embeds)
        if torch.cuda.is_available():
            model.cuda()

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "acclip":
        optimizer = ACClip(model.parameters(), lr=args.lr)

    print ("Start training models!")
    model_name = "{}-{}-{}".format(args.optimizer, args.dataset, args.model)
    train(model, optimizer, train_batches, dev_batches, args.epoch, model_name)

