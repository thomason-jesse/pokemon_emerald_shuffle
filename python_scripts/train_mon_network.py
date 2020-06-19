import argparse
import json
import numpy as np
import pandas as pd
import torch

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
        self.linear1 = torch.nn.Linear(input_dim, 2 * hidden_dim)
        self.nonlinear1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.nonlinear2 = torch.nn.Tanh()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h = self.linear1(x)
        h = self.nonlinear1(h)
        h = self.linear2(h)
        h = self.nonlinear2(h)

        return h


class Decoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim,
                 n_int_data, n_types, n_abilities,
                 n_moves, n_tmhm_moves):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.nonlinear1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.nonlinear2 = torch.nn.Tanh()

        self.int_data_linear = torch.nn.Linear(hidden_dim, n_int_data)
        self.type_linear = torch.nn.Linear(hidden_dim, n_types)
        self.n_evolutions_linear = torch.nn.Linear(hidden_dim, 3)
        self.ability_linear = torch.nn.Linear(hidden_dim, n_abilities)
        self.levelup_linear = torch.nn.Linear(hidden_dim, n_moves)
        self.levelup_lvl_linear = torch.nn.Linear(hidden_dim, n_moves)
        self.tmhm_linear = torch.nn.Linear(hidden_dim, n_tmhm_moves)

        self.n_int_data = n_int_data
        self.n_types = n_types
        self.n_abilities = n_abilities
        self.n_moves = n_moves
        self.n_tmhm_moves = n_tmhm_moves

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        # Embedding values transformed by an appropriate layer.
        y_int_data = self.int_data_linear(x)  # project down (use directly for loss)
        y_types = self.type_linear(x)  # project down (goes to softmax+BCE)
        # y_n_evolutions = self.n_evolutions_linear(x)
        y_abilities = self.ability_linear(x)  # project down (goes to softmax+BCE)
        y_moves = self.levelup_linear(x)  # project down (goes to softmax+BCE)
        y_moves_lvl = self.levelup_lvl_linear(x)  # project down (use directly for loss)
        y_tmhm = self.tmhm_linear(x)  # project down (goes to softmax+BCE)

        return torch.cat((y_int_data,
                          y_types,
                          # y_n_evolutions,
                          y_abilities,
                          y_moves,
                          y_moves_lvl,
                          y_tmhm),
                         1)


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_int_data, n_types, n_abilities,
                 n_moves, n_tmhm_moves):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim,
                               n_int_data, n_types, n_abilities,
                               n_moves, n_tmhm_moves)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu = self.encoder(x)
        y_pred = self.decoder(h_relu)
        return y_pred


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class MonReconstructionLoss:
    def __init__(self, type_list, abilities_list,
                 move_list, tmhm_move_list):
        self.int_data_loss = torch.nn.MSELoss()
        self.type_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        # self.n_evolution_loss = torch.nn.CrossEntropyLoss()
        self.ability_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.levelup_move_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.levelup_move_lvl_loss = MaskedMSELoss()  # Will be masked.
        self.tmhm_move_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.

        self.type_list = type_list
        self.abilities_list = abilities_list
        self.move_list = move_list
        self.tmhm_move_list = tmhm_move_list

    def forward(self, input, target, debug=False):
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
        # n_evolutions_l = self.n_evolution_loss(input[:, idx:idx+3],
        #                                        target[:, idx:idx+3].nonzero()[:, 1])
        # idx += 3

        # Abilities loss.
        abilities_l = self.ability_loss(input[:, idx:idx + len(self.abilities_list)],
                                        target[:, idx:idx + len(self.abilities_list)])
        idx += len(self.abilities_list)

        # Levelup moveset loss.
        levelup_move_l = self.levelup_move_loss(input[:, idx:idx + len(self.move_list)],
                                                target[:, idx:idx + len(self.move_list)])
        idx += len(self.move_list)
        lvl_target_output = target[:, idx:idx + len(self.move_list)]
        t = torch.Tensor([0.0])
        levelup_move_lvl_l = self.levelup_move_lvl_loss(input[:, idx:idx + len(self.move_list)],
                                                        lvl_target_output,
                                                        (lvl_target_output > t).float() * 1)

        # TMHM moveset loss.
        tmhm_move_l = self.tmhm_move_loss(input[:, idx:idx + len(self.tmhm_move_list)],
                                          target[:, idx:idx + len(self.tmhm_move_list)])
        idx += len(self.tmhm_move_list)

        # TODO: add coefficients to reweight this loss function.
        if debug:
            print("int_data loss %.5f(%.5f)" % (int_data_l.item(), int_data_l.item() / len(int_data)))
            print("type loss %.5f(%.5f)" % (type_l.item(), type_l.item() / len(self.type_list)))
            # print("evolution loss %.5f(%.5f)" % (n_evolutions_l.item(), n_evolutions_l.item() / 3.))
            print("abilities loss %.5f(%.5f)" % (abilities_l.item(), abilities_l.item() / len(self.abilities_list)))
            print("levelup loss %.5f(%.5f)" % (levelup_move_l.item(), levelup_move_l.item() / len(self.move_list)))
            print("levelup lvl loss %.5f(%.5f)" % (levelup_move_lvl_l.item(), levelup_move_lvl_l.item() / len(self.move_list)))
            print("tmhm loss %.5f(%.5f)" % (tmhm_move_l.item(), tmhm_move_l.item() / len(self.tmhm_move_list)))
        return (int_data_l +
                type_l +
                # n_evolutions_l +
                abilities_l +
                levelup_move_l + levelup_move_lvl_l +
                tmhm_move_l)


def process_mon_df(df,
                   type_list, abilities_list,
                   move_list, tmhm_move_list):
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
        # n_evolutions_d = [0] * 3
        # n_evolutions_d[df["n_evolutions"][idx]] = 1

        # Abilities (multi-hot).
        abilities_d = [0] * n_abilities
        for jdx in range(2):
            abilities_d[abilities_list.index(df["abilities"][idx][jdx])] = 1

        # Levelup moveset (2 vectors)
        # First vec is a multi-hot
        # Second vec is cont w 0 for never learned or level/100. for level learned.
        levelup_d = [0] * n_moves
        levelup_lvl_d = [0] * n_moves
        for level, move in df["levelup"][idx]:
            levelup_d[move_list.index(move)] = 1
            levelup_lvl_d[move_list.index(move)] = level / 100.

        # TMHM moveset (multi-hot).
        tmhm_d = [0] * len(tmhm_move_list)
        for jdx in range(len(df["tmhm"][idx])):
            tmhm_d[tmhm_move_list.index(df["tmhm"][idx][jdx])] = 1

        d.append(numeric_d +
                 type_d +
                 # n_evolutions_d +
                 abilities_d +
                 levelup_d + levelup_lvl_d +
                 tmhm_d)

    return torch.tensor(d)


def print_mon_vec(y, z_mean, z_std,
                  type_list, abilities_list,
                  move_list, tmhm_move_list):

    # Numeric stats.
    for idx in range(len(int_data)):
        print("%s\t%.0f" % (int_data[idx], int(y[idx] * z_std[idx] + z_mean[idx])))

    # Types.
    idx = len(int_data)
    type_v = list(y[idx:idx + len(type_list)].detach())
    type1_idx = int(np.argmax(type_v))
    type_v[type1_idx] = -float('inf')
    type2_idx = int(np.argmax(type_v))
    idx += len(type_list)
    print("type1 %s\ntype2 %s" % (type_list[type1_idx],
                                  type_list[type2_idx]))

    # N evolutions.
    # n_evolutions = np.argmax(y[idx:idx+3])
    # idx += 3
    # print("n evolutions\t%d" % n_evolutions)

    # Abilities.
    ability_v = y[idx:idx + len(abilities_list)]
    ability1_idx = np.argmax(ability_v)
    ability_v[ability1_idx] = -float('inf')
    ability2_idx = np.argmax(ability_v)
    idx += len(abilities_list)
    print("ability1 %s\nability2 %s" % (abilities_list[ability1_idx],
                                        abilities_list[ability2_idx]))

    # Levelup moveset.
    levels = []
    moves = []
    for jdx in range(len(move_list)):
        if y[idx + jdx] > 0.5:
            levels.append(y[idx + jdx + len(move_list)] * 100)
            moves.append(move_list[jdx])
    idx += len(move_list) * 2
    print("levelup moveset:")
    for jdx in np.argsort(levels):
        print("\t%.0f\t%s" % (levels[jdx], moves[jdx]))

    # TMHM moveset.
    print("TMHM moveset:")
    probs = [y[idx+jdx] / sum(y[idx:idx+len(tmhm_move_list)])
             for jdx in range(len(tmhm_move_list))]
    for jdx in np.argsort(probs)[::-1]:
        if probs[jdx] > 0.5:
            print("\t%.2f\t%s" % (probs[jdx], tmhm_move_list[jdx]))


def main(args):

    print("Reading in mon metadata and evolutions; then z-scoring.")
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_metadata = d["mon_metadata"]
        for a in mon_metadata:  # replace secondary type match with TYPE_NONE
            if mon_metadata[a]["type1"] == mon_metadata[a]["type2"]:
                mon_metadata[a]["type2"] = "TYPE_NONE"
        type_list = d["type_list"]
        type_list.append("TYPE_NONE")  # add TYPE_NONE for prediction purposes.
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
    tmhm_moves_set = set()
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
            tmhm_moves_set.add(move)
    abilities_list = list(abilities_set)
    moves_list = list(moves_set)
    tmhm_moves_list = list(tmhm_moves_set)
    print("... done")

    # Create little encoder decoder layers for type and stats.
    df = pd.DataFrame.from_dict(mon_metadata, orient='index')

    # Convert mon data into usable inputs.
    x = process_mon_df(df, type_list, abilities_list, moves_list, tmhm_moves_list)
    input_dim = len(x[0])

    # Construct our model by instantiating the class defined above
    h = 64
    print("Input size: %d; embedding dimension: %d" % (input_dim, h))
    model = Autoencoder(input_dim, h, input_dim,
                        len(int_data), len(type_list), len(abilities_list),
                        len(moves_list), len(tmhm_moves_list))

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    criterion = MonReconstructionLoss(type_list, abilities_list, moves_list, tmhm_moves_list)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    epochs = 1000000
    print_every = int(epochs / 100.)
    mon_every = int(epochs / 10.)
    n_mon_every = 1
    tsne = TSNE(random_state=1, n_iter=1000, metric="euclidean")  # For visualizing embeddings.
    for t in range(epochs + 1):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion.forward(y_pred, x)
        if t % print_every == 0:
            print("epoch %d\tloss %.5f" % (t, loss.item()))

            # Sample some mon
            if t % mon_every == 0:
                with torch.no_grad():
                    # Show sample forward pass.
                    print("Forward pass categorical losses:")
                    criterion.forward(y_pred, x, debug=True)

                    # Visualize mon embeddings at this stage.
                    embs = model.encoder.forward(x)
                    if h > 2:
                        embs_2d = tsne.fit_transform(embs)
                    elif h == 2:
                        embs_2d = embs.detach()
                    else:  # h == 1
                        embs_2d = np.zeros(shape=(len(mon_list), 2))
                        embs_2d[:, 0] = embs.detach()[:, 0]
                        embs_2d[:, 1] = embs.detach()[:, 0]
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.scatter(embs_2d[:, 0], embs_2d[:, 1], alpha=.1)
                    paths = [df["icon"][midx] for midx in range(len(mon_list))]
                    for x0, y0, path in zip(embs_2d[:, 0], embs_2d[:, 1], paths):
                        img = plt.imread(path)
                        ab = AnnotationBbox(OffsetImage(img[:32, :32, :]),
                                            (x0, y0), frameon=False)
                        ax.add_artist(ab)
                    plt.savefig("%s.%d.mon_embeddings.pdf" % (args.output_fn, t),
                                bbox_inches='tight')

                    # Show reconstruction mon.
                    print("Sample reconstruction mon:")
                    for _ in range(n_mon_every):
                        midx = np.random.choice(list(range(len(mon_metadata))))
                        z = model.encoder.forward(x[midx].unsqueeze(0))
                        print(df["species"][midx], z)
                        y = model.decoder.forward(z)
                        print_mon_vec(y[0], z_mean, z_std,
                                      type_list, abilities_list, moves_list, tmhm_moves_list)

                    # Show sample mon.
                    print("Sample generated mon:")
                    for _ in range(n_mon_every):
                        z = torch.randn(h).clamp(min=-1, max=1)
                        print(z)
                        y = model.decoder.forward(z.unsqueeze(0))
                        print_mon_vec(y[0], z_mean, z_std,
                                      type_list, abilities_list, moves_list, tmhm_moves_list)

                    # Write trained model to file.
                    torch.save(model.state_dict(), "%s.%d.model" %
                               (args.output_fn, t))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the trained generator network')
    args = parser.parse_args()


    main(args)
