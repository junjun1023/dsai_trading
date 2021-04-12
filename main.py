from torch.utils.data import DataLoader
from code import model
from code.data import Dataset
from code import epoch
from code import evaluation

import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

from tqdm import tqdm
import json

from sklearn.preprocessing import normalize

import pandas as pd

root = os.getcwd()
batch = 32
# forecast = 30
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
# reference = 300
date_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
<<<<<<< HEAD
# os.makedirs(os.path.join(root, "results", date_time))
# date_time = "2021-04-12_07-59"
=======
os.makedirs(os.path.join(root, "results", date_time))
# date_time = "2021-04-12_07-59"
load_checkpoint = False
>>>>>>> 4a67b81c86ddc8f4efb4a7c4afbf7706da8c9910

# You can write code above the if-main block.


def load_checkpoint(filepath, device, forecast):

    encoder = model.Extractor(in_channels=1, out_channels=1, maxpool=False)
    decoder = model.Decoder(classes=forecast, hidden_size=32,
                            num_layers=5)  # forecast 30 days

    predictor = model.Model(encoder=encoder, decoder=decoder).to(device)

    if os.path.exists(filepath) and load_checkpoint:
        checkpoint = torch.load(filepath)
        predictor.load_state_dict(checkpoint['model_stat'])
        optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-4)
        optimizer.load_state_dict(checkpoint['optimizer_stat'])
        print("Find pretrain.")

    else:
        optimizer = torch.optim.Adam(predictor.parameters(), lr=3e-4)
        print("New model and optimizer")

    return predictor, optimizer


if __name__ == '__main__':
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
<<<<<<< HEAD
        parser.add_argument('--testing',
                                default='testing_data.csv',
                                help='input testing data file name')
        parser.add_argument('--output',
                                default='output.csv',
                                help='output file name')
        parser.add_argument('--epoch',
                                default='30',
                                help='epoch to train')
        parser.add_argument('--sample',
                                default='10',
                                help='sample times')
        parser.add_argument('--reference',
                                default='300',
                                help='days to reference')
        parser.add_argument('--forecast',
                                default='30',
                                help='days to forecast')
        args = parser.parse_args()

        ###### Model
        predictor, optimizer = load_checkpoint(filepath=os.path.join(root, "results", date_time, "{}.pth".format(date_time)), device=device, forecast=int(args.forecast))

        ###### End Model

        ###### Data: Train
        trainset = Dataset(args.training, reference=int(args.reference), forecast=int(args.forecast))
        trainloader = DataLoader(trainset, batch_size=batch, shuffle=False, num_workers=2)

        ###### End Data

        ###### Train
        train_info = {
                "kendal": []
        }

        kendal_min = 1
        for e in range(int(args.epoch)):
        
                train_loss = epoch.train_epoch(predictor, optimizer, trainloader, device, args.sample)
                pr, gt = epoch.test_epoch(predictor, trainset, device)
                
                kendal = evaluation.normalised_kendall_tau_distance(gt, pr)
                train_info["kendal"].append(kendal)
                
                print("Epoch: {}, loss = {:.5f}, kendal = {:.5f}".format(e+1, train_loss, kendal))
                
                if kendal < kendal_min:
                        checkpoint = {
                                'model_stat': predictor.state_dict(),
                                'optimizer_stat': optimizer.state_dict(),
                        }

                        torch.save(checkpoint, os.path.join(root, "results", date_time, "{}.pth".format(date_time)))
                        kendal_min = kendal
                with open(os.path.join(root, "results", date_time, "{}.json".format(date_time)), 'w') as f:
                        json.dump(train_info, f)
        ###### End Train

        ###### Test
        train_df = pd.read_csv(args.training, names=["open", "high", "low", "close"])
        train_df = train_df.tail(int(args.reference))
        test_df = pd.read_csv(args.testing, names=["open", "high", "low", "close"])
        dataframe = pd.concat([train_df, test_df], ignore_index=True)
        dataset = Dataset(path=dataframe, reference=int(args.reference), forecast=int(args.forecast))

        predictor, optimizer = load_checkpoint(filepath=os.path.join(root, "results", date_time, "{}.pth".format(date_time)), device=device, forecast=int(args.forecast))
        actions = epoch.online_trading(predictor, dataset, device)

        actions = np.array(actions)

        np.savetxt(args.output, actions, fmt="%d")

        # trader = Trader()
        # trader.train(training_data)

        # testing_data = load_data(args.testing)
        # with open(args.output, 'w') as output_file:
        #         for row in testing_data:
        #                 # We will perform your action as the open price in the next day.
        #                 action = trader.predict_action(row)
        #                 output_file.write(action)

        #                 # this is your option, you can leave it empty.
        #                 trader.re_training(i)
=======
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    parser.add_argument('--epoch',
                        default='30',
                        help='epoch to train')
    parser.add_argument('--sample',
                        default='10',
                        help='sample times')
    parser.add_argument('--reference',
                        default='300',
                        help='days to reference')
    parser.add_argument('--forecast',
                        default='30',
                        help='days to forecast')
    args = parser.parse_args()

    # Model
    predictor, optimizer = load_checkpoint(filepath=os.path.join(
        root, "results", date_time, "{}.pth".format(date_time)), device=device, forecast=int(args.forecast))

    # End Model

    ###### Data: Train
    trainset = Dataset(args.training, reference=int(
        args.reference), forecast=int(args.forecast))
    trainloader = DataLoader(trainset, batch_size=batch,
                             shuffle=False, num_workers=2)

    # End Data

    # Train
    train_info = {
        "kendal": []
    }
    print(trainset[0][0].shape)

    kendal_min = 1
    for e in range(int(args.epoch)):

        train_loss = epoch.train_epoch(
            predictor, optimizer, trainloader, device, args.sample)
        pr, gt = epoch.test_epoch(predictor, trainset, device)

        kendal = evaluation.normalised_kendall_tau_distance(gt, pr)
        train_info["kendal"].append(kendal)

        print("Epoch: {}, loss = {:.5f}, kendal = {:.5f}".format(
            e+1, train_loss, kendal))

        if kendal < kendal_min:
            checkpoint = {
                'model_stat': predictor.state_dict(),
                'optimizer_stat': optimizer.state_dict(),
            }

            torch.save(checkpoint, os.path.join(root, "results",
                                                date_time, "{}.pth".format(date_time)))
            kendal_min = kendal
        with open(os.path.join(root, "results", date_time, "{}.json".format(date_time)), 'w') as f:
            json.dump(train_info, f)
    # End Train

    # Test
    train_df = pd.read_csv(args.training, names=[
                           "open", "high", "low", "close"])
    train_df = train_df.tail(int(args.reference))
    test_df = pd.read_csv(args.testing, names=["open", "high", "low", "close"])
    dataframe = pd.concat([train_df, test_df], ignore_index=True)
    dataset = Dataset(path=dataframe, reference=int(
        args.reference), forecast=int(args.forecast))

    predictor, optimizer = load_checkpoint(filepath=os.path.join(
        root, "results", date_time, "{}.pth".format(date_time)), device=device, forecast=int(args.forecast))
    actions = epoch.online_trading(predictor, dataset, device)

    actions = np.array(actions)

    np.savetxt(args.output, actions, fmt="%d")

    # trader = Trader()
    # trader.train(training_data)

    # testing_data = load_data(args.testing)
    # with open(args.output, 'w') as output_file:
    #         for row in testing_data:
    #                 # We will perform your action as the open price in the next day.
    #                 action = trader.predict_action(row)
    #                 output_file.write(action)

    #                 # this is your option, you can leave it empty.
    #                 trader.re_training(i)
>>>>>>> 4a67b81c86ddc8f4efb4a7c4afbf7706da8c9910
