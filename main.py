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
batch = 8
forecast = 30
samples = 5
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
reference = 300
date_time = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M")
os.makedirs(os.path.join(root, "results", date_time))


# You can write code above the if-main block.
def load_checkpoint(filepath, device):

        encoder = model.Extractor(in_channels=1, out_channels=1, maxpool=True)
        decoder = model.Decoder(classes=forecast, hidden_size=32, num_layers=5) # forcast 30 days

        predictor = model.Model(encoder=encoder, decoder=decoder).to(device)
        
                
        if os.path.exists(filepath):
                checkpoint = torch.load(filepath)
                predictor.load_state_dict(checkpoint['model_stat'])
                optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
                optimizer.load_state_dict(checkpoint['optimizer_stat'])
                
                print("Find pretrain.")

        else:
                optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)
                
                print("New model and optimizer")
                
        return predictor, optimizer



if __name__ == '__main__':
        # You should not modify this part.
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--training',
                        default='training_data.csv',
                        help='input training data file name')
        parser.add_argument('--testing',
                                default='testing_data.csv',
                                help='input testing data file name')
        parser.add_argument('--output',
                                default='output.csv',
                                help='output file name')
        args = parser.parse_args()

        ###### Model
        predictor, optimizer = load_checkpoint(
                                filepath=os.path.join(root, "results", date_time, "{}.pth".format(date_time))),
                                device=device)

        ###### End Model

        ###### Data: Train
        training_data = load_data(args.training)

        trainset = Dataset(args.training, reference=reference, forecast=forecast)
        trainloader = DataLoader(trainset, batch_size=batch, shuffle=False, num_workers=2)

        ###### End Data

        ###### Train

        train_info = {
                "kendal": []
        }

        kendal_min = 1
        for e in range(5000):
        
                train_loss = epoch.train_epoch(predictor, optimizer, trainloader, device)
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
        train_df = train_df.tail(reference)
        test_df = pd.read_csv(args.testing, names=["open", "high", "low", "close"])
        dataframe = pd.concat([train_df, test_df], ignore_index=True)
        dataset = Dataset(path=dataframe)

        predictor, optimizer = load_checkpoint(
                        filepath=os.path.join(root, "results", date_time, "{}.pth".format(date_time))),
                        device=device)
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