# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob
import torch
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from model import MyAwesomeModel


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('figures_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path(exists=True))
def main(input_filepath,figures_filepath ,model_filepath):
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.0001)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[4:])
    print(args)
    

    model = MyAwesomeModel()
    train_images = torch.load(input_filepath + '/train_images.pt')
    train_labels = torch.load(input_filepath + '/train_labels.pt')
    train_set = TensorDataset(train_images,train_labels)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    epochs = 5
    AccToSave = []
    LossToSave = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

        else:
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            AccToSave.append(accuracy)
            LossToSave.append(loss.detach().numpy())
            print(f'Epoch: {e}----------- Loss: {loss} -------------- Accuracy: {accuracy.item()*100}%')
            # print(f'Accuracy: {accuracy.item()*100}%')
            # print(f'Loss: {loss}')
    
    plt.plot(AccToSave)
    plt.plot(LossToSave)
    plt.savefig(figures_filepath + '/EvaluationMetrics.png')
    torch.save(model.state_dict(), model_filepath + '/TrainedModel.pth')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()