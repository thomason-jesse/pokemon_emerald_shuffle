import argparse
import json
import math
import numpy as np
import pandas as pd
import random
import torch


# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship', 'safariZoneFleeRate']
stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.linear1(x).clamp(min=0)
        return h_relu


class Decoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Decoder, self).__init__()
        self.linear2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = self.linear2(x)
        return y_pred


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.encoder.forward(x)
        y_pred = self.decoder.forward(h_relu)
        return y_pred


def process_mon_df(df, type_list):

    # Construct input row by row.
    d = []
    for idx in df.index:
        # Numeric stats (z score normalized inputs).
        numeric_d = [df[n_d][idx] for n_d in int_data]

        # Types (one-hot).
        type_d = [0] * (len(type_list) * 2)
        type_d[type_list.index(df["type1"][idx])] = 1
        type_d[type_list.index(df["type2"][idx]) + len(type_list)] = 1

        d.append(numeric_d + type_d)

    return torch.tensor(d)


def print_mon_vec(y, z_mean, z_std, type_list):
    # TODO: see nearest neighbor by calculating penalty against original mon.

    # Numeric stats.
    for idx in range(len(int_data)):
        print("%s\t%.2f" % (int_data[idx], int(y[idx] * z_std[idx] + z_mean[idx]))

    # Types.
    type1_idx = np.argmax(y[len(int_data):len(int_data)+len(type_list)])
    type2_idx = np.argmax(y[len(int_data) + len(type_list):
                            len(int_data) + 2*len(type_list)])
    print("type1 %s\ntype2 %s" % (type_list[type1_idx], type_list[type2_idx]))


def main(args):

    print("Reading in mon metadata and evolutions; then z-scoring and reversing structures.")
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_metadata = d["mon_metadata"]
        type_list = d["type_list"]
        mon_evolution = d["mon_evolution"]
    # Perform z score normalization on numeric metadata.
    mon_list = list(mon_metadata.keys())
    z_mean = [None] * len(int_data)
    z_std = [None] * len(int_data)
    for idx in range(len(int_data)):
        data_name = int_data[idx]
        data_mean = np.mean([mon_metadata[a][data_name] for a in mon_list])
        data_std = np.std([mon_metadata[a][data_name] for a in mon_list])
        for a in mon_list:
            mon_metadata[a][data_name] = (mon_metadata[a][data_name] - data_mean) / data_std
        z_mean[idx] = data_mean
        z_std[idx] = data_std
    # Create reversed evolutions list.
    mon_evolution_r = {}
    for a in mon_evolution:
        for b in mon_evolution[a]:
            if b in mon_evolution_r:
                print("WARNING: multiple base species for %d" % b)
            mon_evolution_r[b] = a
    print("... done")

    # Create little encoder decoder layers for type and stats.
    df = pd.DataFrame.from_dict(mon_metadata, orient='index')

    # Convert mon data into usable inputs.
    x = process_mon_df(df, type_list)
    input_dim = len(x[0])

    # Construct our model by instantiating the class defined above
    h = 8
    model = Autoencoder(input_dim, h, input_dim)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = torch.nn.MSELoss()  # TODO: define custom loss (MSE for base; CE for type)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(100000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, x)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Sample some mon
    with torch.no_grad():
        while True:
            z = torch.randn(h)
            y = model.decoder.forward(z)
            print_mon_vec(y, z_mean, z_std, type_list)
            _ = input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    args = parser.parse_args()


    main(args)
