
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from torch_geometric.datasets import TUDataset
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import random
import time

import warnings

from GCN import GCN
from SGForrmer.ours_1 import accuracy, val, SGFormer
# from GCN import GCN
from split_dataset import split_dataset
from sub_process import get_dataset
import os

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

MODEL_PATH = '/home_b/zhaoke1/GT-K-FOLD-1.0/SGFormer'


def train(args, model, train_loader, val_loader, test_loader):
    saved_epoch_id = 0
    test_acc_result = 0

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=False)

    min_loss = 1e10
    max_acc = 0
    early_stop_patience = 30  # 设置早停的耐心值
    early_stop_counter = 0
    best_epoch = 0
    for epoch in tqdm(range(args.epochs)):
        losses_list = []
        train_acc_list = []
        model.train()
        for data in train_loader:
            pred_graph = []
            # loss_F1_graph = []
            data = data.to(args.device)
            optimizer.zero_grad()
            # pred = model(data)
            for i in range(len(data.sub_batch)):

                # pred = model(data)
                pred = model(data[i])
                pred_graph.append(pred)
            # print(len(pred_graph))
            # loss_F1_graph.append(loss_F1)

            pred_graph_batch = torch.cat(pred_graph, dim=0)
            loss = loss_func(pred_graph_batch, data.y)
            loss.backward()
            optimizer.step()
            losses_list.append(loss.detach().item())
            train_acc_list.append(accuracy(pred_graph_batch, data.y))
            torch.cuda.empty_cache()

        loss = np.average(losses_list)
        train_acc = np.average(train_acc_list)
        val_acc, val_loss = val(model, val_loader, args)
        test_acc, test_loss = val(model, test_loader, args)
        # if val_loss < min_loss and epoch > args.epochs * 0.7:
        if val_acc > max_acc:
            max_acc = val_acc
            test_acc_result = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_PATH + 'Model-NCI' + '.kpl')
            print("\n EPOCH:{}\t "
                  "Train Loss:{:.4f}\tTrain ACC: {:.4f}\t"
                  "Val Loss:{:.4f}\t Val ACC: {:.4f}\t"
                  "Test Loss:{:.4f}\t Test ACC: {:.4f}\t"
                  "Model saved.".format(epoch, loss, train_acc, val_loss, val_acc, test_loss, test_acc))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print("\n EPOCH:{}\t "
                  "Train Loss:{:.4f}\tTrain ACC: {:.4f}\t"
                  "Val Loss:{:.4f}\t Val ACC: {:.4f}\t"
                  "Test Loss:{:.4f}\t Test ACC: {:.4f}\t"
                  "Early stopping counter: {}/{}".format(epoch, loss, train_acc, val_loss, val_acc, test_loss, test_acc,
                                                         early_stop_counter, early_stop_patience))

        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    print("SUCCESS: Model Training Finished.")
    return best_epoch, test_acc_result

# @torch.no_grad()
# def test(test_loader, args, model):
#     model = Net(args).to(args.device)
#     model.load_state_dict(torch.load(MODEL_PATH + 'Model-NCI' + '.kpl'))
#     test_acc, test_loss = val(model, test_loader, args)
#     return test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="NCI109", help='name of dataset')
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--seed', type=int, default=809, help='random seed')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('--wd', type=float, default=5e-3, help='weight decay value')
    parser.add_argument('--ours_layers', type=int, default=2)
    parser.add_argument('--edge_dim', type=int, default=8)
    parser.add_argument('--task_num', type=int, default=2)
    parser.add_argument('--ours_dropout', type=float, help='gnn dropout.', default=0.5)
    parser.add_argument('--dropout', type=float, help='gnn dropout.', default=0.5)
    parser.add_argument('--num_kfold', type=int, default=10, help='number of kfold')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_graph', action='store_true', default=True
                        , help='use pos emb')
    parser.add_argument('--ours_use_weight', action='store_true', help='use weight for trans convolution')
    parser.add_argument('--ours_use_residual', action='store_true', help='use residual link for each trans layer')
    parser.add_argument('--ours_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for residual link')
    parser.add_argument('--ratio_save_model', type=float, default=0.7, help='weight for residual link')
    parser.add_argument('--ratio_entropy', type=float, default=0.7, help='weight for residual link')
    parser.add_argument('--graph_weight', type=float, default=0.5, help='graph weight.')
    parser.add_argument('--device', type=str, help='The type of the running device')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for deep methods')
    parser.add_argument('--num_random', type=int, default=1, help='number of layers for deep methods')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')
    parser.add_argument('--num_heads', type=int, default=1, help='number of heads for attention')
    args = parser.parse_args()
    print('hello world')

    if torch.cuda.is_available():
        args.device = 'cuda:1'


    dataset = get_dataset(args.dataset)

    # dataset = TUDataset(root='/home_b/zhaoke1/GT-K-FOLD-1.0/TUDataset', name='MUTAG', use_node_attr=True)
    d = dataset.sub_batch[0].x.size(1)

    # d = dataset.x.size(1)

    random_seed = list(range(args.num_random))
    # random_seed = list([2])
    test_acc_list = []

    time_train_list = []
    for seed in random_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        kf = KFold(n_splits=args.num_kfold, random_state=seed, shuffle=True)
        for iter, (train_index, test_index) in enumerate(kf.split(dataset)):
            gnn = GCN(hidden_channels=64, num_node_features=d,
                      num_classes=args.task_num).to(args.device)
            # gnn = GCN(in_channels=d,
            #             hidden_channels=args.dim,
            #             out_channels=args.task_num,
            #             num_layers=args.num_layers,
            #             dropout=args.dropout,
            #             use_bn=args.use_bn).to(args.device)
            model = SGFormer(d, args.dim, args.task_num, num_layers=args.ours_layers, alpha=args.alpha,
                             dropout=args.ours_dropout, num_heads=args.num_heads,
                             use_bn=args.use_bn, use_residual=args.ours_use_residual, use_graph=args.use_graph,
                             use_weight=args.ours_use_weight, use_act=args.ours_use_act, graph_weight=args.graph_weight,
                             gnn=gnn, aggregate=args.aggregate).to(args.device)
            print("Fold:{}".format(iter))
            # if iter < 3:
            #     continue
            train_loader, val_loader, test_loader = split_dataset(args, train_index, test_index, dataset, iter)

            time_train_start = time.time()
            saved_epoch_id, test_acc = train(args, model, train_loader, val_loader, test_loader)
            time_train_end = time.time()
            time_train_delta = time_train_end - time_train_start
            # test_acc = test(test_loader, args)
            test_acc_list.append(test_acc)
            time_train_list.append(time_train_delta)
            print("\nFold:{} \t Test ACC:{:.4f}".format(iter, test_acc))
        acc_avg = np.average(test_acc_list)
        acc_std = np.std(test_acc_list)
        time_train_avg = np.average(time_train_list)
        time_train_std = np.std(time_train_list)
        print("\n TOTAL: Test ACC:{:.4f}, Test ACC STD:{:.4f} Using the model at epoch {}".format(acc_avg, acc_std,
                                                                                                  saved_epoch_id))
        print("Test ACC list:")
        print(test_acc_list)
        print("Running Time:")
        print("Train Time AVG: {:.4f}".format(time_train_avg))
        print("Train Time STD: {:.4f}".format(time_train_std))


if __name__ == "__main__":
    main()