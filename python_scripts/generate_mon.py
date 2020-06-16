import argparse
import json
import numpy as np
import pandas as pd
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


class MonReconstructionLoss:
    def __init__(self, type_list, abilities_list, move_list, n_levelup_moves):
        self.int_data_loss = torch.nn.MSELoss()
        self.type_loss = torch.nn.BCEWithLogitsLoss()
        self.n_evolution_loss = torch.nn.CrossEntropyLoss()
        self.ability_loss = torch.nn.BCEWithLogitsLoss()
        self.levelup_level_loss = torch.nn.MSELoss()
        self.levelup_move_loss = [torch.nn.CrossEntropyLoss() for _ in range(n_levelup_moves)]
        self.tmhm_move_loss = torch.nn.BCEWithLogitsLoss()

        self.type_list = type_list
        self.abilities_list = abilities_list
        self.move_list = move_list
        self.n_levelup_moves = n_levelup_moves

    def forward(self, input, target):
        target_long = torch.tensor(target, dtype=torch.long)
        idx = 0

        # Numeric data loss.
        int_data_l = self.int_data_loss(input[:, idx:idx + len(int_data)],
                                        target[:, idx:idx + len(int_data)])
        idx += len(int_data)

        # Type loss.
        type_l = self.type_loss(input[:, idx:idx + len(self.type_list)],
                                target[:, idx:idx + len(self.type_list)])
        idx += len(self.type_list)

        # N evolutions loss.
        n_evolutions_l = self.n_evolution_loss(input[:, idx:idx+3],
                                               target_long[:, idx:idx+3].nonzero()[:, 1])
        idx += 3

        # Abilities loss.
        abilities_l = self.ability_loss(input[:, idx:idx + len(self.abilities_list)],
                                        target[:, idx:idx + len(self.abilities_list)])
        idx += len(self.abilities_list)

        # Levelup moveset loss.
        levelup_level_l = self.levelup_level_loss(input[:, idx:idx + self.n_levelup_moves],
                                                  target[:, idx:idx + self.n_levelup_moves])
        idx += self.n_levelup_moves
        levelup_move_l = 0
        for jdx in range(self.n_levelup_moves):
            levelup_move_l += self.levelup_move_loss[jdx](
                input[:, idx:idx + len(self.move_list)],
                target_long[:, idx:idx + len(self.move_list)].nonzero()[:, 1])
            idx += len(self.move_list)

        # TMHM moveset loss.
        tmhm_move_l = self.tmhm_move_loss(input[:, idx:idx + len(self.move_list)],
                                          target[:, idx:idx + len(self.move_list)])
        idx += len(self.move_list)

        # TODO: add coefficients to reweight this loss function.
        # print("%.5f\t%.5f\t%.5f" % (int_data_l, type_l, n_evolutions_l))
        return (int_data_l +
                type_l +
                n_evolutions_l +
                abilities_l +
                levelup_level_l + levelup_move_l +
                tmhm_move_l)


def process_mon_df(df,
                   type_list, abilities_list, move_list, n_levelup_moves):
    n_abilities = len(abilities_list)
    n_moves = len(move_list)

    # Construct input row by row.
    d = []
    for idx in df.index:
        # Numeric stats (z score normalized inputs).
        numeric_d = [df[n_d][idx] for n_d in int_data]

        # Types (multi-hot).
        type_d = [0] * len(type_list)
        type_d[type_list.index(df["type1"][idx])] = 1
        type_d[type_list.index(df["type2"][idx])] = 1

        # n evolutions.
        n_evolutions_d = [0] * 3
        n_evolutions_d[df["n_evolutions"][idx]] = 1

        # Abilities (multi-hot).
        abilities_d = [0] * n_abilities
        for jdx in range(2):
            abilities_d[abilities_list.index(df["abilities"][idx][jdx])] = 1

        # Levelup moveset (series of one-hots).
        levelup_d = [0] * (n_levelup_moves + (n_levelup_moves * n_moves))
        for jdx in range(n_levelup_moves):
            if jdx < len(df["levelup"][idx]):
                levelup_d[jdx] = df["levelup"][idx][jdx][0]
                levelup_d[n_levelup_moves + (jdx * n_moves) + move_list.index(df["levelup"][idx][jdx][1])] = 1
            else:  # This mon doens't learn as many moves as we'd like to infer.
                levelup_d[jdx] = 1
                levelup_d[n_levelup_moves + (jdx * n_moves) + move_list.index("MOVE_NONE")] = 1

        # TMHM moveset (multi-hot).
        tmhm_d = [0] * n_moves
        for jdx in range(len(df["tmhm"][idx])):
            tmhm_d[move_list.index(df["tmhm"][idx][jdx])] = 1

        d.append(numeric_d + type_d + n_evolutions_d + abilities_d + levelup_d + tmhm_d)

    return torch.tensor(d)


def print_mon_vec(y, z_mean, z_std,
                  type_list, abilities_list, move_list, n_levelup_moves):

    # Numeric stats.
    for idx in range(len(int_data)):
        print("%s\t%.2f" % (int_data[idx], int(y[idx] * z_std[idx] + z_mean[idx])))

    # Types.
    idx = len(int_data)
    type_v = list(y[idx:idx + len(type_list)].detach())
    type1_idx = int(np.argmax(type_v))
    type_v[type1_idx] = -float('inf')
    type2_logit = np.max(type_v)
    if type2_logit > 0.5:
        type2_idx = np.argmax(type_v)
    else:
        type2_idx = type1_idx
    idx += len(type_list)
    print("type1 %s\ntype2 %s" % (type_list[type1_idx],
                                  type_list[type2_idx]))

    # N evolutions.
    n_evolutions = np.argmax(y[idx:idx+3])
    idx += 3
    print("n evolutions\t%d" % n_evolutions)

    # Abilities.
    ability_v = y[idx:idx + len(abilities_list)]
    ability1_idx = np.argmax(ability_v)
    ability_v[ability1_idx] = -float('inf')
    ability2_idx = np.argmax(ability_v)
    idx += len(abilities_list)
    print("ability1 %s\nability2 %s" % (abilities_list[ability1_idx],
                                        abilities_list[ability2_idx]))

    # Levelup moveset.
    levels = [l * 100 for l in y[idx:idx+n_levelup_moves]]
    idx += n_levelup_moves
    moves = []
    for jdx in range(n_levelup_moves):
        moves.append(move_list[np.argmax(y[idx:idx + len(move_list)])])
        idx += len(move_list)
    print("levelup moveset:")
    for jdx in range(n_levelup_moves):
        print("\t%.2f\t%s" % (levels[jdx], moves[jdx]))

    # TMHM moveset.
    print("TMHM moveset:")
    for jdx in range(len(move_list)):
        if y[idx+jdx] > 0.5:
            print("\t%s" % move_list[jdx])


def main(args):

    print("Reading in mon metadata and evolutions; then z-scoring.")
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_metadata = d["mon_metadata"]
        type_list = d["type_list"]
        mon_evolution = d["mon_evolution"]
        mon_levelup_moveset = d["mon_levelup_moveset"]
        mon_tmhm_moveset = d["mon_tmhm_moveset"]

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
    print("... done")

    # Augment metadata with evolutions.
    print("Augmenting metadata with evolutions...")
    for a in mon_metadata:
        mon_metadata[a]["n_evolutions"] = 0
        if a in mon_evolution:
            mon_metadata[a]["n_evolutions"] += 1
            if np.any([b in mon_evolution for b in mon_evolution[a]]):
                mon_metadata[a]["n_evolutions"] += 1
    print("... done")

    # Augment metadata with levelup and tmhm movesets.
    # Note abilities on the way.
    print("Augmenting metadata with levelup and tmhm movesets...")
    abilities_set = set()
    moves_set = set()
    for a in mon_metadata:
        for ab in mon_metadata[a]["abilities"]:
            abilities_set.add(ab)
        mon_metadata[a]["levelup"] = []
        for level, move in mon_levelup_moveset[a]:
            mon_metadata[a]["levelup"].append([level / 100., move])
        for _, move in mon_levelup_moveset[a]:
            moves_set.add(move)
        mon_metadata[a]["tmhm"] = mon_tmhm_moveset[a][:]
        for move in mon_tmhm_moveset[a]:
            moves_set.add(move)
    abilities_list = list(abilities_set)
    moves_list = list(moves_set)
    moves_list.append("MOVE_NONE")
    print("... done")

    # Create little encoder decoder layers for type and stats.
    df = pd.DataFrame.from_dict(mon_metadata, orient='index')

    # Convert mon data into usable inputs.
    avg_num_moves = int(np.round(sum([len(df["levelup"][idx]) for idx in df.index]) / float(len(df.index))))
    x = process_mon_df(df, type_list, abilities_list, moves_list, avg_num_moves)
    input_dim = len(x[0])

    # Construct our model by instantiating the class defined above
    h = 8
    model = Autoencoder(input_dim, h, input_dim)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = MonReconstructionLoss(type_list, abilities_list, moves_list, avg_num_moves)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for t in range(10000):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion.forward(y_pred, x)
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
            print_mon_vec(y, z_mean, z_std,
                          type_list, abilities_list, moves_list, avg_num_moves)
            _ = input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    args = parser.parse_args()


    main(args)
