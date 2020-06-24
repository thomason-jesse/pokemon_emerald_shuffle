import argparse
import json
import numpy as np
import pandas as pd
import torch

from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship']  # don't actually care about predicting 'safariZoneFleeRate'


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, 2 * hidden_dim)
        self.nonlinear1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(2 * hidden_dim, hidden_dim)
        self.nonlinear2 = torch.nn.Tanh()
        self.vae_linear = [torch.nn.Linear(hidden_dim, hidden_dim),
                           torch.nn.Linear(hidden_dim, hidden_dim)]

    def forward(self, x):
        h = self.linear1(x)
        h = self.nonlinear1(h)
        h = self.linear2(h)
        h = self.nonlinear2(h)
        h_mu = self.vae_linear[0](h)
        h_std = self.vae_linear[1](h)

        return h_mu, h_std


class Decoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim,
                 n_int_data, n_types, n_abilities,
                 n_moves, n_tmhm_moves,
                 name_len, name_chars,
                 n_egg_groups, n_growth_rates, n_gender_ratios):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, 2 * hidden_dim)
        self.nonlinear1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(2 * hidden_dim, output_dim)
        self.nonlinear2 = torch.nn.Tanh()

        self.name_linear = torch.nn.Linear(hidden_dim, name_len * len(name_chars))
        self.int_data_linear = torch.nn.Linear(hidden_dim, n_int_data)
        self.type_linear = torch.nn.Linear(hidden_dim, n_types)
        self.n_evolutions_linear = torch.nn.Linear(hidden_dim, 3)
        self.ability_linear = torch.nn.Linear(hidden_dim, n_abilities)
        self.levelup_linear = torch.nn.Linear(hidden_dim, n_moves)
        self.tmhm_linear = torch.nn.Linear(hidden_dim, n_tmhm_moves)
        self.egg_linear = torch.nn.Linear(hidden_dim, n_egg_groups)
        self.growth_linear = torch.nn.Linear(hidden_dim, n_growth_rates)
        self.gender_linear = torch.nn.Linear(hidden_dim, n_gender_ratios)

        self.n_int_data = n_int_data
        self.n_types = n_types
        self.n_abilities = n_abilities
        self.n_moves = n_moves
        self.n_tmhm_moves = n_tmhm_moves
        self.n_egg_groups = n_egg_groups
        self.n_growth_rates = n_growth_rates
        self.n_gender_ratios = n_gender_ratios

    def forward(self, x):
        # Embedding values transformed by an appropriate layer.
        # y_name = self.name_linear(x)  # project down (goes to softmaxes)
        y_int_data = self.int_data_linear(x)  # project down (use directly for MSE loss)
        y_types = self.type_linear(x)  # project down (goes to softmax+BCE)
        y_n_evolutions = self.n_evolutions_linear(x)  # project down (goes to softmax+CE)
        y_abilities = self.ability_linear(x)  # project down (goes to softmax+BCE)
        y_moves = self.levelup_linear(x)  # project down (goes to softmax+BCE)
        y_tmhm = self.tmhm_linear(x)  # project down (goes to softmax+BCE)
        y_egg = self.egg_linear(x)  # project down (goes to softmax+BCE)
        y_growth = self.growth_linear(x)  # project down (goes to softmax+CE)
        y_gender = self.gender_linear(x)  # project down (goes to softmax+CE)

        return torch.cat((# y_name,
                          y_int_data,
                          y_types,
                          y_n_evolutions,
                          y_abilities,
                          y_moves,
                          y_tmhm,
                          y_egg,
                          y_growth,
                          y_gender),
                         1)


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_int_data, n_types, n_abilities,
                 n_moves, n_tmhm_moves,
                 name_len, name_chars,
                 n_egg_groups, n_growth_rates, n_gender_ratios):

        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim,
                               n_int_data, n_types, n_abilities,
                               n_moves, n_tmhm_moves,
                               name_len, name_chars,
                               n_egg_groups, n_growth_rates, n_gender_ratios)
        self.hidden_dim = hidden_dim

    def forward(self, x):

        h_mu, h_std = self.encoder(x)
        y_pred = self.decoder(h_mu + h_std * torch.randn((h_std.shape[0], self.hidden_dim)))
        return y_pred


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class MonReconstructionLoss:
    def __init__(self, n_types, n_abilities,
                 n_moves, n_tmhm_moves,
                 moves_weights, tmhm_weights,
                 name_len, name_chars,
                 n_egg_groups, n_growth_rates, n_gender_ratios):
        # self.name_loss = [torch.nn.CrossEntropyLoss() for _ in range(name_len)]
        self.int_data_loss = torch.nn.MSELoss()
        self.type_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.n_evolution_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.
        self.ability_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.levelup_move_loss = torch.nn.BCEWithLogitsLoss(pos_weight=moves_weights)  # Has sigmoid.
        self.tmhm_move_loss = torch.nn.BCEWithLogitsLoss(pos_weight=tmhm_weights)  # Has sigmoid.
        self.egg_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.growth_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.
        self.gender_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.

        self.n_types = n_types
        self.n_abilities = n_abilities
        self.n_moves = n_moves
        self.n_tmhm_moves = n_tmhm_moves
        self.name_len = name_len
        self.name_chars = name_chars
        self.n_egg_groups = n_egg_groups
        self.n_growth_rates = n_growth_rates
        self.n_gender_ratios = n_gender_ratios

    def forward(self, input, target, debug=False):
        idx = 0

        # Name loss.
        # name_l = sum([self.name_loss[jdx](input[:, idx + jdx * len(self.name_chars):idx + (jdx + 1) * len(self.name_chars)],
        #                                   target[:, idx + jdx * len(self.name_chars):idx + (jdx + 1) * len(self.name_chars)].nonzero()[:, 1])
        #               for jdx in range(self.name_len)])
        # idx += self.name_len * len(self.name_chars)

        # Numeric data loss.
        int_data_l = self.int_data_loss(input[:, idx:idx + len(int_data)],
                                        target[:, idx:idx + len(int_data)])
        idx += len(int_data)

        # Type loss.
        type_l = self.type_loss(input[:, idx:idx + self.n_types],
                                target[:, idx:idx + self.n_types])
        idx += self.n_types

        # N evolutions loss.
        n_evolutions_l = self.n_evolution_loss(input[:, idx:idx+3],
                                               target[:, idx:idx+3].nonzero()[:, 1])
        idx += 3

        # Abilities loss.
        abilities_l = self.ability_loss(input[:, idx:idx + self.n_abilities],
                                        target[:, idx:idx + self.n_abilities])
        idx += self.n_abilities

        # Levelup moveset loss.
        levelup_move_l = self.levelup_move_loss(input[:, idx:idx + self.n_moves],
                                                target[:, idx:idx + self.n_moves])
        idx += self.n_moves

        # TMHM moveset loss.
        tmhm_move_l = self.tmhm_move_loss(input[:, idx:idx + self.n_tmhm_moves],
                                          target[:, idx:idx + self.n_tmhm_moves])
        idx += self.n_tmhm_moves

        # Egg groups loss.
        egg_l = self.egg_loss(input[:, idx:idx + self.n_egg_groups],
                              target[:, idx:idx + self.n_egg_groups])
        idx += self.n_egg_groups

        # Growth rate loss.
        growth_l = self.growth_loss(input[:, idx:idx + self.n_growth_rates],
                                    target[:, idx:idx + self.n_growth_rates].nonzero()[:, 1])
        idx += self.n_growth_rates

        # Gender ratios loss.
        gender_l = self.gender_loss(input[:, idx:idx + self.n_gender_ratios],
                                    target[:, idx:idx + self.n_gender_ratios].nonzero()[:, 1])
        idx += self.n_gender_ratios

        if debug:
            # print("name loss %.5f(%.5f)" % (name_l.item(), name_l.item() / (self.name_len * len(self.name_chars))))
            print("int_data loss %.5f(%.5f)" % (int_data_l.item(), int_data_l.item() / len(int_data)))
            print("type loss %.5f(%.5f)" % (type_l.item(), type_l.item() / self.n_types))
            print("evolution loss %.5f(%.5f)" % (n_evolutions_l.item(), n_evolutions_l.item() / 3.))
            print("abilities loss %.5f(%.5f)" % (abilities_l.item(), abilities_l.item() / self.n_abilities))
            print("levelup loss %.5f(%.5f)" % (levelup_move_l.item(), levelup_move_l.item() / self.n_moves))
            print("tmhm loss %.5f(%.5f)" % (tmhm_move_l.item(), tmhm_move_l.item() / self.n_tmhm_moves))
            print("egg groups loss %.5f(%.5f)" % (egg_l.item(), egg_l.item() / self.n_egg_groups))
            print("growth rate loss %.5f(%.5f)" % (growth_l.item(), growth_l.item() / self.n_growth_rates))
            print("gender ratio loss %.5f(%.5f)" % (gender_l.item(), gender_l.item() / self.n_gender_ratios))
        return (# name_l +
                int_data_l +
                type_l +
                n_evolutions_l +
                abilities_l +
                levelup_move_l +
                tmhm_move_l)


def process_mon_df(df,
                   type_list, abilities_list,
                   move_list, tmhm_move_list,
                   name_len, name_chars,
                   egg_groups_list, growth_rates_list, gender_ratios_list):
    n_abilities = len(abilities_list)
    n_moves = len(move_list)

    # Construct input row by row.
    d = []
    for idx in df.index:
        # Name data.
        # name_d = [0] * len(name_chars) * name_len
        # for jdx in range(len(df["name_chars"][idx])):
        #     name_d[jdx * len(name_chars) + name_chars.index(df["name_chars"][idx][jdx])] = 1
        # for jdx in range(len(df["name_chars"][idx]), name_len):
        #     name_d[jdx * len(name_chars) + name_chars.index('_')] = 1

        # Numeric stats (z score normalized inputs).
        numeric_d = [df[n_d][idx] for n_d in int_data]

        # Types (multi-hot).
        type_d = [0] * len(type_list)
        for jdx in range(2):
            type_d[type_list.index(df["type%d" % (jdx + 1)][idx])] = 1

        # n evolutions (one-hot).
        n_evolutions_d = [0] * 3
        n_evolutions_d[df["n_evolutions"][idx]] = 1

        # Abilities (multi-hot).
        abilities_d = [0] * n_abilities
        for jdx in range(2):
            abilities_d[abilities_list.index(df["abilities"][idx][jdx])] = 1

        # Levelup moveset (multi-hot).
        levelup_d = [0] * n_moves
        for level, move in df["levelup"][idx]:
            levelup_d[move_list.index(move)] = 1

        # TMHM moveset (multi-hot).
        tmhm_d = [0] * len(tmhm_move_list)
        for jdx in range(len(df["tmhm"][idx])):
            tmhm_d[tmhm_move_list.index(df["tmhm"][idx][jdx])] = 1

        # Egg groups (multi-hot).
        egg_d = [0] * len(egg_groups_list)
        for jdx in range(2):
            egg_d[egg_groups_list.index(df["eggGroups"][idx][jdx])] = 1

        # Growth rate (one-hot).
        growth_d = [0] * len(growth_rates_list)
        growth_d[growth_rates_list.index(df["growthRate"][idx])] = 1

        # Gender ratio (one-hot).
        gender_d = [0] * len(gender_ratios_list)
        gender_d[gender_ratios_list.index(df["genderRatio"][idx])] = 1

        d.append(# name_d +
                 numeric_d +
                 type_d +
                 n_evolutions_d +
                 abilities_d +
                 levelup_d +
                 tmhm_d +
                 egg_d +
                 growth_d +
                 gender_d)

    # Turn list of lists to tensor.
    d = torch.tensor(d)

    # Create pos_weight tensors for moves.
    idx_start = len(int_data) + \
                len(type_list) + \
                3 + \
                n_abilities
                # + (name_len * len(name_chars))
    moves_pos_weight = torch.tensor([sum(1 - d[:, idx_start+idx]) / sum(d[:, idx_start+idx])
                                     for idx in range(n_moves)])
    idx_start += n_moves
    tmhm_pos_weight = torch.tensor([sum(1 - d[:, idx_start + idx]) / sum(d[:, idx_start + idx])
                                    for idx in range(len(tmhm_move_list))])

    return d, moves_pos_weight, tmhm_pos_weight


def print_mon_vec(y, z_mean, z_std,
                  type_list, abilities_list,
                  move_list, tmhm_move_list,
                  name_len, name_chars,
                  egg_groups_list, growth_rates_list, gender_ratios_list):
    idx = 0

    # Name.
    # print("name\t" + ''.join([name_chars[np.argmax(y[jdx * len(name_chars): (jdx+1) * len(name_chars)])]
    #                           for jdx in range(idx, idx+name_len)]))
    # idx += name_len * len(name_chars)

    # Numeric stats.
    for jdx in range(idx, idx+len(int_data)):
        print("%s:\t%.0f" % (int_data[jdx - idx], int(y[jdx] * z_std[jdx - idx] + z_mean[jdx - idx])))
    idx = len(int_data)

    # Types.
    type_v = list(y[idx:idx + len(type_list)].detach())
    type1_idx = int(np.argmax(type_v))
    type_v[type1_idx] = -float('inf')
    type2_idx = int(np.argmax(type_v))
    idx += len(type_list)
    print("type1:\t%s\ntype2:\t%s" % (type_list[type1_idx],
                                      type_list[type2_idx]))

    # N evolutions.
    n_evolutions = np.argmax(y[idx:idx+3])
    idx += 3
    print("n evolutions:\t%d" % n_evolutions)

    # Abilities.
    ability_v = list(y[idx:idx + len(abilities_list)].detach())
    ability1_idx = int(np.argmax(ability_v))
    ability_v[ability1_idx] = -float('inf')
    ability2_idx = int(np.argmax(ability_v))
    idx += len(abilities_list)
    print("ability1\t%s\nability2:\t%s" % (abilities_list[ability1_idx],
                                           abilities_list[ability2_idx]))

    # Levelup moveset.
    levels = []
    moves = []
    probs = torch.sigmoid(y[idx:idx + len(move_list)]).numpy()
    idx += len(move_list)
    for jdx in range(len(probs)):
        if probs[jdx] >= torch.sigmoid(torch.tensor([1], dtype=torch.float64)).item():
            moves.append(move_list[jdx])
            levels.append(probs[jdx])  # Just report confidence
    print("levelup moveset:")
    for jdx in np.argsort(levels)[::-1]:  # show in order of confidence
        print("\t%.2f\t%s" % (levels[jdx], moves[jdx]))

    # TMHM moveset.
    print("TMHM moveset:")
    probs = torch.sigmoid(y[idx:idx+len(tmhm_move_list)]).numpy()
    idx += len(tmhm_move_list)
    for jdx in np.argsort(probs)[::-1]:
        if probs[jdx] >= torch.sigmoid(torch.tensor([1], dtype=torch.float64)).item():
            print("\t%.2f\t%s" % (probs[jdx].item(), tmhm_move_list[jdx]))

    # Egg groups.
    egg_v = list(y[idx:idx + len(egg_groups_list)].detach())
    egg1_idx = int(np.argmax(egg_v))
    egg_v[egg1_idx] = -float('inf')
    egg2_idx = int(np.argmax(egg_v))
    idx += len(egg_groups_list)
    print("eggGroup1:\t%s\neggGroup2:\t%s" % (egg_groups_list[egg1_idx],
                                              egg_groups_list[egg2_idx]))

    # Growth rate.
    gr_idx = np.argmax(y[idx:idx + len(growth_rates_list)])
    idx += len(growth_rates_list)
    print("growth rate: \t%s" % growth_rates_list[gr_idx])

    # Gender ratio.
    gd_idx = np.argmax(y[idx:idx + len(gender_ratios_list)])
    idx += len(gender_ratios_list)
    print("gender ratio: \t%s" % gender_ratios_list[gd_idx])


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

    # Convert names to character lists.
    print("Reading mon names and converting to character lists")
    name_len = 0
    name_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ_')
    for a in mon_metadata:
        n = mon_metadata[a]["species"][len('SPECIES_'):]
        mon_metadata[a]['name_chars'] = [c if c in name_chars else '_' for c in n]
        name_len = max(name_len, len(mon_metadata[a]['name_chars']))
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
    # Note abilities, growth rate, egg group, and gender ratio on the pass.
    print("Augmenting metadata with levelup and tmhm movesets...")
    ab_mon = {}  # map from abilities to mon who have them
    mv_mon = {}  # map from moves to mon who have them
    eg_mon = {}  # map from egg groups to mon who have them
    growth_rates_list = []
    gender_ratios_list = []
    tmhm_moves_set = set()
    for a in mon_metadata:
        for ab in mon_metadata[a]["abilities"]:
            if ab not in ab_mon:
                ab_mon[ab] = []
            if a not in ab_mon[ab]:
                ab_mon[ab].append(a)
        gr = mon_metadata[a]["growthRate"]
        if gr not in growth_rates_list:
            growth_rates_list.append(gr)
        gn = mon_metadata[a]["genderRatio"]
        if gn not in gender_ratios_list:
            gender_ratios_list.append(gn)
        for eg in mon_metadata[a]["eggGroups"]:
            if eg not in eg_mon:
                eg_mon[eg] = []
            if a not in eg_mon[eg]:
                eg_mon[eg].append(a)
        mon_metadata[a]["levelup"] = []
        for level, move in mon_levelup_moveset[a]:
            mon_metadata[a]["levelup"].append([level, move])
        for _, move in mon_levelup_moveset[a]:
            if move not in mv_mon:
                mv_mon[move] = []
            if a not in mv_mon[move]:
                mv_mon[move].append(a)
        mon_metadata[a]["tmhm"] = mon_tmhm_moveset[a][:]
        for move in mon_tmhm_moveset[a]:
            tmhm_moves_set.add(move)

    # Collapse single-appearance abilities and moves.
    abilities_list = [ab for ab in ab_mon if len(ab_mon[ab]) > 1] + ["INFREQUENT"]
    infrequent_abilities = {ab: ab_mon[ab] for ab in ab_mon if len(ab_mon[ab]) == 1}
    moves_list = [mv for mv in mv_mon if len(mv_mon[mv]) > 1] + ["INFREQUENT"]
    infrequent_moves = {mv: mv_mon[mv] for mv in mv_mon if len(mv_mon[mv]) == 1}
    egg_group_list = [eg for eg in eg_mon if len(eg_mon[eg]) > 1] + ["INFREQUENT"]
    infrequent_egg_groups = {eg: eg_mon[eg] for eg in eg_mon if len(eg_mon[eg]) == 1}
    print("infrequent abilities", infrequent_abilities)
    print("infrequent learned moves", infrequent_moves)
    print("infrequent egg groups", infrequent_egg_groups)
    with open("%s.infrequent.json" % args.output_fn, 'w') as f:
        json.dump({"abilities": infrequent_abilities,
                   "moves": infrequent_moves,
                   "egg_groups": infrequent_egg_groups}, f)
    for a in mon_metadata:
        for idx in range(len(mon_metadata[a]["levelup"])):
            level, move = mon_metadata[a]["levelup"][idx]
            if move in infrequent_moves:
                mon_metadata[a]["levelup"][idx] = [level, "INFREQUENT"]
        for idx in range(len(mon_metadata[a]["abilities"])):
            if mon_metadata[a]["abilities"][idx] in infrequent_abilities:
                mon_metadata[a]["abilities"][idx] = "INFREQUENT"
        for idx in range(len(mon_metadata[a]["eggGroups"])):
            if mon_metadata[a]["eggGroups"][idx] in infrequent_egg_groups:
                mon_metadata[a]["eggGroups"][idx] = "INFREQUENT"

    tmhm_moves_list = list(tmhm_moves_set)
    print("... done")

    # Create little encoder decoder layers for type and stats.
    df = pd.DataFrame.from_dict(mon_metadata, orient='index')

    # Convert mon data into usable inputs.
    x, moves_pos_weight, tmhm_pos_weight = process_mon_df(df, type_list, abilities_list,
                                                          moves_list, tmhm_moves_list,
                                                          name_len, name_chars,
                                                          egg_group_list, growth_rates_list, gender_ratios_list)
    input_dim = len(x[0])

    # Construct our model.
    h = 128
    print("Input size: %d; embedding dimension: %d" % (input_dim, h))
    model = Autoencoder(input_dim, h, input_dim,
                        len(int_data), len(type_list), len(abilities_list),
                        len(moves_list), len(tmhm_moves_list),
                        name_len, name_chars,
                        len(egg_group_list), len(growth_rates_list), len(gender_ratios_list))

    # Construct our loss function and an Optimizer.
    criterion = MonReconstructionLoss(len(type_list), len(abilities_list),
                                      len(moves_list), len(tmhm_moves_list),
                                      moves_pos_weight, tmhm_pos_weight,
                                      name_len, name_chars,
                                      len(egg_group_list), len(growth_rates_list), len(gender_ratios_list))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train.
    epochs = 10000
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
                    embs_mu, embs_std = model.encoder.forward(x)
                    if h > 2:
                        embs_2d = tsne.fit_transform(embs_mu)
                    elif h == 2:
                        embs_2d = embs_mu.detach()
                    else:  # h == 1
                        embs_2d = np.zeros(shape=(len(mon_list), 2))
                        embs_2d[:, 0] = embs_mu.detach()[:, 0]
                        embs_2d[:, 1] = embs_mu.detach()[:, 0]
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
                        print("----------")
                        print(df["species"][midx], "orig")
                        print_mon_vec(x[midx], z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      name_len, name_chars,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------")
                        print(df["species"][midx], "embedded", embs_mu[midx])
                        y = model.decoder.forward(embs_mu[midx].unsqueeze(0))
                        print_mon_vec(y[0], z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      name_len, name_chars,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------")

                    # Show sample mon.
                    print("Sample generated mon:")
                    for _ in range(n_mon_every):
                        z = torch.randn(h)
                        print("----------")
                        print("random embedding", z)
                        y = model.decoder.forward(z.unsqueeze(0))
                        print_mon_vec(y[0], z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      name_len, name_chars,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------")

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
