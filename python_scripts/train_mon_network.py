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

torch.manual_seed(0)

# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship']  # don't actually care about predicting 'safariZoneFleeRate'

# Training loss weights.
# Learn these first
int_data_w = np.power(2, 3)  # learn these first
type_w = np.power(2, 3)
n_evolutions_w = np.power(2, 3)
# Learn these second
abilities_w = np.power(2, 2)
growth_w = np.power(2, 2)
# Learn these third.
levelup_move_w = np.power(2, 1)
tmhm_move_w = np.power(2, 1)
# Learn these last.
egg_w = 1
gender_w = 1

# Increase KL weight to get more Guassian-looking clusters if overfitting; decrease to more tightly fit training data.
kl_w = 1


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 n_int_data, n_types, n_abilities,
                 n_moves, n_tmhm_moves,
                 n_egg_groups, n_growth_rates, n_gender_ratios,
                 device):
        super(Autoencoder, self).__init__()

        # Encoder layers.
        self.linear1 = torch.nn.Linear(input_dim, 2 * hidden_dim).to(device)
        self.nonlinear1 = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(2 * hidden_dim, hidden_dim).to(device)
        self.nonlinear2 = torch.nn.Tanh()
        self.vae_linear = [torch.nn.Linear(hidden_dim, hidden_dim).to(device),
                           torch.nn.Linear(hidden_dim, hidden_dim).to(device)]

        # Decoder layers.
        self.int_data_linear = torch.nn.Linear(hidden_dim, n_int_data).to(device)
        self.type_linear = torch.nn.Linear(hidden_dim, n_types).to(device)
        self.n_evolutions_linear = torch.nn.Linear(hidden_dim, 3).to(device)
        self.ability_linear = torch.nn.Linear(hidden_dim, n_abilities).to(device)
        self.levelup_linear = torch.nn.Linear(hidden_dim, n_moves).to(device)
        self.tmhm_linear = torch.nn.Linear(hidden_dim, n_tmhm_moves).to(device)
        self.egg_linear = torch.nn.Linear(hidden_dim, n_egg_groups).to(device)
        self.growth_linear = torch.nn.Linear(hidden_dim, n_growth_rates).to(device)
        self.gender_linear = torch.nn.Linear(hidden_dim, n_gender_ratios).to(device)

        # Params to remember.
        self.hidden_dim = hidden_dim
        self.n_int_data = n_int_data
        self.n_types = n_types
        self.n_abilities = n_abilities
        self.n_moves = n_moves
        self.n_tmhm_moves = n_tmhm_moves
        self.n_egg_groups = n_egg_groups
        self.n_growth_rates = n_growth_rates
        self.n_gender_ratios = n_gender_ratios

    def encode(self, x):
        h = self.linear1(x)
        h = self.nonlinear1(h)
        h = self.linear2(h)
        h = self.nonlinear2(h)
        h_mu = self.vae_linear[0](h)
        h_std = self.vae_linear[1](h)

        return h_mu, h_std

    def decode(self, x):
        # Embedding values transformed by an appropriate layer.
        y_int_data = self.int_data_linear(x)  # project down (use directly for MSE loss)
        y_types = self.type_linear(x)  # project down (goes to softmax+BCE)
        y_n_evolutions = self.n_evolutions_linear(x)  # project down (goes to softmax+CE)
        y_abilities = self.ability_linear(x)  # project down (goes to softmax+BCE)
        y_moves = self.levelup_linear(x)  # project down (goes to softmax+BCE)
        y_tmhm = self.tmhm_linear(x)  # project down (goes to softmax+BCE)
        y_egg = self.egg_linear(x)  # project down (goes to softmax+BCE)
        y_growth = self.growth_linear(x)  # project down (goes to softmax+CE)
        y_gender = self.gender_linear(x)  # project down (goes to softmax+CE)

        return torch.cat((y_int_data,
                          y_types,
                          y_n_evolutions,
                          y_abilities,
                          y_moves,
                          y_tmhm,
                          y_egg,
                          y_growth,
                          y_gender),
                         1)

    def forward(self, x):
        h_mu, h_std = self.encode(x)
        if self.training:
            std_coeff = torch.rand_like(h_std)
        else:
            std_coeff = torch.ones_like(h_std)
        y_pred = self.decode(h_mu + torch.exp(h_std) * std_coeff)
        return y_pred, h_std


class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
        result = torch.sum(diff2) / torch.sum(mask)
        return result


class MonReconstructionLoss:
    def __init__(self, batch_size, hidden_dim,
                 n_types, n_abilities,
                 n_moves, n_tmhm_moves,
                 moves_weights, tmhm_weights,
                 n_egg_groups, n_growth_rates, n_gender_ratios,
                 device):

        # Reconstruction losses.
        self.int_data_loss = torch.nn.MSELoss()
        self.type_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.n_evolution_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.
        self.ability_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.levelup_move_loss = torch.nn.BCEWithLogitsLoss(pos_weight=moves_weights)  # Has sigmoid.
        self.tmhm_move_loss = torch.nn.BCEWithLogitsLoss(pos_weight=tmhm_weights)  # Has sigmoid.
        self.egg_loss = torch.nn.BCEWithLogitsLoss()  # Has sigmoid.
        self.growth_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.
        self.gender_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.

        # Expects log probabilities as input, probabilities as targets.
        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.kl_target = torch.ones(batch_size, hidden_dim).to(device)

        self.n_types = n_types
        self.n_abilities = n_abilities
        self.n_moves = n_moves
        self.n_tmhm_moves = n_tmhm_moves
        self.n_egg_groups = n_egg_groups
        self.n_growth_rates = n_growth_rates
        self.n_gender_ratios = n_gender_ratios

    def forward(self, input_emb, input_std, target, debug=False):
        idx = 0

        # Numeric data loss.
        int_data_l = self.int_data_loss(input_emb[:, idx:idx + len(int_data)],
                                        target[:, idx:idx + len(int_data)])
        idx += len(int_data)

        # Type loss.
        type_l = self.type_loss(input_emb[:, idx:idx + self.n_types],
                                target[:, idx:idx + self.n_types])
        idx += self.n_types

        # N evolutions loss.
        n_evolutions_l = self.n_evolution_loss(input_emb[:, idx:idx+3],
                                               target[:, idx:idx+3].nonzero()[:, 1])
        idx += 3

        # Abilities loss.
        abilities_l = self.ability_loss(input_emb[:, idx:idx + self.n_abilities],
                                        target[:, idx:idx + self.n_abilities])
        idx += self.n_abilities

        # Levelup moveset loss.
        levelup_move_l = self.levelup_move_loss(input_emb[:, idx:idx + self.n_moves],
                                                target[:, idx:idx + self.n_moves])
        idx += self.n_moves

        # TMHM moveset loss.
        tmhm_move_l = self.tmhm_move_loss(input_emb[:, idx:idx + self.n_tmhm_moves],
                                          target[:, idx:idx + self.n_tmhm_moves])
        idx += self.n_tmhm_moves

        # Egg groups loss.
        egg_l = self.egg_loss(input_emb[:, idx:idx + self.n_egg_groups],
                              target[:, idx:idx + self.n_egg_groups])
        idx += self.n_egg_groups

        # Growth rate loss.
        growth_l = self.growth_loss(input_emb[:, idx:idx + self.n_growth_rates],
                                    target[:, idx:idx + self.n_growth_rates].nonzero()[:, 1])
        idx += self.n_growth_rates

        # Gender ratios loss.
        gender_l = self.gender_loss(input_emb[:, idx:idx + self.n_gender_ratios],
                                    target[:, idx:idx + self.n_gender_ratios].nonzero()[:, 1])
        idx += self.n_gender_ratios

        # KL divergence of input std dev with Guassian.
        kl_l = torch.abs(self.kl_loss(input_std, self.kl_target))

        if debug:
            print("int_data loss\t%.5f\t(%.5f)" % (int_data_l.item(), int_data_w * int_data_l.item()))
            print("type loss\t%.5f\t(%.5f)" % (type_l.item(), type_w * type_l.item()))
            print("evolution loss\t%.5f\t(%.5f)" % (n_evolutions_l.item(), n_evolutions_w * n_evolutions_l.item()))
            print("abilities loss\t%.5f\t(%.5f)" % (abilities_l.item(), abilities_w * abilities_l.item()))
            print("levelup loss\t%.5f\t(%.5f)" % (levelup_move_l.item(), levelup_move_w * levelup_move_l.item()))
            print("tmhm loss\t%.5f\t(%.5f)" % (tmhm_move_l.item(), tmhm_move_w * tmhm_move_l.item()))
            print("egg groups loss\t%.5f\t(%.5f)" % (egg_l.item(), egg_w * egg_l.item()))
            print("growth rate loss\t%.5f\t(%.5f)" % (growth_l.item(), growth_w * growth_l.item()))
            print("gender ratio loss\t%.5f\t(%.5f)" % (gender_l.item(), gender_w * gender_l.item()))
            print("kl loss on std\t%.5f\t(%.5f)" % (kl_l.item(), kl_w * kl_l.item()))
        return (int_data_w * int_data_l +
                type_w * type_l +
                n_evolutions_w * n_evolutions_l +
                abilities_w * abilities_l +
                levelup_move_w * levelup_move_l +
                tmhm_move_w * tmhm_move_l +
                egg_w * egg_l +
                growth_w * growth_l +
                gender_w * gender_l +
                kl_w * kl_l)


def print_mon_vec(y, z_mean, z_std,
                  type_list, abilities_list,
                  move_list, tmhm_move_list,
                  egg_groups_list, growth_rates_list, gender_ratios_list):
    idx = 0

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
    print(' ; '.join(["%s (%.2f)" % (moves[jdx], levels[jdx])
                      for jdx in np.argsort(levels)[::-1]]))

    # TMHM moveset.
    print("TMHM moveset:")
    probs = torch.sigmoid(y[idx:idx+len(tmhm_move_list)]).numpy()
    idx += len(tmhm_move_list)
    print(' ; '.join(["%s (%.2f)" % (tmhm_move_list[jdx], probs[jdx].item())
                      for jdx in np.argsort(probs)[::-1]
                      if probs[jdx] >= torch.sigmoid(torch.tensor([1], dtype=torch.float64)).item()]))

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Reading mon model metadata from '%s'..." % args.input_fn)
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_list = d["mon_list"]
        mon_icons = [d['mon_metadata'][mon_list[midx]]["icon"] for midx in range(len(mon_list))]
        z_mean = d["z_mean"]
        z_std = d["z_std"]
        input_dim = d["input_dim"]
        h = d["hidden_dim"]
        int_data = d["int_data"]
        type_list = d["type_list"]
        abilities_list = d["abilities_list"]
        moves_list = d["moves_list"]
        tmhm_moves_list = d["tmhm_moves_list"]
        egg_group_list = d["egg_group_list"]
        growth_rates_list = d["growth_rates_list"]
        gender_ratios_list = d["gender_ratios_list"]
        moves_pos_weight = torch.tensor(np.asarray(d["moves_pos_weight"]), dtype=torch.float32).to(device)
        tmhm_pos_weight = torch.tensor(np.asarray(d["tmhm_pos_weight"]), dtype=torch.float32).to(device)
        x = torch.tensor(np.asarray(d["x_input"]), dtype=torch.float32).to(device)
    print("... done")

    # Construct our model.
    print("Input size: %d; embedding dimension: %d" % (input_dim, h))
    model = Autoencoder(input_dim, h,
                        len(int_data), len(type_list), len(abilities_list),
                        len(moves_list), len(tmhm_moves_list),
                        len(egg_group_list), len(growth_rates_list), len(gender_ratios_list),
                        device)

    # Construct our loss function and an Optimizer.
    criterion = MonReconstructionLoss(x.shape[0], h,
                                      len(type_list), len(abilities_list),
                                      len(moves_list), len(tmhm_moves_list),
                                      moves_pos_weight, tmhm_pos_weight,
                                      len(egg_group_list), len(growth_rates_list), len(gender_ratios_list),
                                      device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    if args.input_model_pref:
        print("Loading initial model and optimizer weights from prefix '%s'..." % args.input_model_pref)
        model.load_state_dict(torch.load("%s.model" % args.input_model_pref))
        optimizer.load_state_dict(torch.load("%s.opt" % args.input_model_pref))
        print("... done")

    # Train.
    epochs = 10000
    print_every = int(epochs / 100.)
    mon_every = int(epochs / 10.)
    n_mon_every = 1
    tsne = TSNE(random_state=1, n_iter=1000, metric="euclidean")  # For visualizing embeddings.
    least_loss = None
    for t in range(epochs + 1):
        model.train()
        # Forward pass: Compute predicted y by passing x to the model
        y_pred, h_std_pred = model(x)

        # Compute and print loss
        loss = criterion.forward(y_pred, h_std_pred, x)
        if t % print_every == 0:
            print("epoch %d\tloss %.5f" % (t, loss.item()))

            # Sample some mon
            if t % mon_every == 0:
                with torch.no_grad():
                    model.eval()
                    # Show sample forward pass.
                    y_pred, h_std_pred = model(x)
                    print("Forward pass categorical losses:")
                    criterion.forward(y_pred, h_std_pred, x, debug=True)

                    # Visualize mon embeddings at this stage.
                    embs_mu, embs_std = model.encode(x)
                    if h > 2:
                        embs_2d = tsne.fit_transform(embs_mu.cpu())
                    elif h == 2:
                        embs_2d = embs_mu.detach()
                    else:  # h == 1
                        embs_2d = np.zeros(shape=(len(mon_list), 2))
                        embs_2d[:, 0] = embs_mu.detach()[:, 0]
                        embs_2d[:, 1] = embs_mu.detach()[:, 0]
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.scatter(embs_2d[:, 0], embs_2d[:, 1], alpha=.1)
                    paths = mon_icons
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
                        midx = np.random.choice(list(range(len(mon_list))))
                        print("----------")
                        print(mon_list[midx], "orig")
                        print_mon_vec(x[midx].cpu(), z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------")
                        print(mon_list[midx], "embedded")
                        y = model.decode(embs_mu[midx].unsqueeze(0))
                        print_mon_vec(y[0].cpu(), z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------")

                    # Show sample mon.
                    print("Sample generated mon:")
                    for _ in range(n_mon_every):
                        z = torch.mean(embs_mu.cpu(), dim=0) + torch.std(embs_mu.cpu(), dim=0) * torch.randn(h)
                        print("----------")
                        print("random embedding")
                        y = model.decode(z.unsqueeze(0).to(device))
                        print_mon_vec(y[0].cpu(), z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------")

        # If this is the least loss achieved, write file.
        if least_loss is None or loss < least_loss:
            least_loss = loss.detach().item()
            if args.verbose:
                print("... wrote least loss model with l=%.5f at epoch %d" % (least_loss, t))  # verbose
            torch.save(model.state_dict(), "%s.best.model" % args.output_fn)
            torch.save(optimizer.state_dict(), "%s.best.opt" % args.output_fn)

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
    parser.add_argument('--input_model_pref', type=str,
                        help='input model/optimizer weights prefix to continue training')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='whether to print a bunch')
    args = parser.parse_args()

    main(args)
