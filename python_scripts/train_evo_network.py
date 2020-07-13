import argparse
import json
import numpy as np
import pandas as pd
import torch

from train_mon_network import Autoencoder, print_mon_vec


class EvoNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_evo_types, n_evo_items):
        super(EvoNet, self).__init__()

        # Shrink with non-linear layer.
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.nonlinear1 = torch.nn.Tanh()

        # Prediction layers.
        self.evo_emb_linear = torch.nn.Linear(hidden_dim, output_dim)
        self.evo_type_linear = torch.nn.Linear(hidden_dim, n_evo_types)
        self.evo_item_linear = torch.nn.Linear(hidden_dim, n_evo_items)
        self.evo_level_linear = torch.nn.Linear(hidden_dim, 1)

        self.hidden_dim = hidden_dim
        self.n_evo_types = n_evo_types
        self.n_evo_items = n_evo_items

    def forward(self, x):
        h = self.linear1(x)
        h = self.nonlinear1(h)

        # Embedding values transformed by an appropriate layer.
        y_emb = self.evo_emb_linear(h)  # project down (use directly for MSE loss)
        y_type = self.evo_type_linear(h)  # project down (goes to softmax+CE)
        y_item = self.evo_item_linear(h)  # project down (goes to softmax+CE)
        y_level = self.evo_level_linear(h)  # project down (use directly for MSE loss)

        return y_emb, y_type, y_item, y_level


class EvoReconstructionLoss:
    def __init__(self, evo_item_ignore_idx):
        self.emb_loss = torch.nn.MSELoss()
        self.type_loss = torch.nn.CrossEntropyLoss()  # Has sigmoid.
        self.item_loss = torch.nn.CrossEntropyLoss(ignore_index=evo_item_ignore_idx)  # Has sigmoid.
        self.level_loss = torch.nn.MSELoss()

    def forward(self, input_emb, input_type, input_item, input_level,
                target_emb, target_type, target_item, target_level,
                debug=False):

        emb_loss = self.emb_loss(input_emb, target_emb)
        type_loss = self.type_loss(input_type, target_type)
        item_loss = self.item_loss(input_item, target_item)
        level_loss = self.level_loss(input_level, target_level)

        if debug:
            print("emb loss %.5f" % emb_loss.item())
            print("type loss %.5f" % type_loss.item())
            print("item loss %.5f" % item_loss.item())
            print("level loss %.5f" % level_loss.item())
        return emb_loss + type_loss + item_loss + level_loss


def main(args):

    print("Reading mon model metadata from '%s'..." % args.input_meta_fn)
    with open(args.input_meta_fn, 'r') as f:
        d = json.load(f)
        mon_list = d["mon_list"]
        mon_metadata = d["mon_metadata"]
        z_mean = d["z_mean"]
        z_std = d["z_std"]
        mon_evolution = d["mon_evolution"]
        ae_input_dim = d["input_dim"]
        ae_hidden_dim = d["hidden_dim"]
        int_data = d["int_data"]
        type_list = d["type_list"]
        abilities_list = d["abilities_list"]
        moves_list = d["moves_list"]
        tmhm_moves_list = d["tmhm_moves_list"]
        name_len = d["name_len"]
        name_chars = d["name_chars"]
        egg_group_list = d["egg_group_list"]
        growth_rates_list = d["growth_rates_list"]
        gender_ratios_list = d["gender_ratios_list"]
        ae_x = torch.tensor(np.asarray(d["x_input"]), dtype=torch.float32)
    df = pd.DataFrame.from_dict(mon_metadata, orient='index')
    print("... done")

    print("Initializing trained mon Autoencoder model from '%s'..." % args.input_model_fn)
    autoencoder_model = Autoencoder(ae_input_dim, ae_hidden_dim, ae_input_dim,
                                    len(int_data), len(type_list), len(abilities_list),
                                    len(moves_list), len(tmhm_moves_list),
                                    name_len, name_chars,
                                    len(egg_group_list), len(growth_rates_list), len(gender_ratios_list))
    autoencoder_model.load_state_dict(torch.load(args.input_model_fn))
    print("... done")

    # Create mon embeddings by running them through the autoencoder encoder layer.
    print("Creating input representations for mon embeddings, evo types, evo items...")
    embs_mu, embs_std = autoencoder_model.encoder.forward(ae_x)
    embs_mu = embs_mu.detach()
    embs_std = embs_std.detach()

    evo_types = []
    evo_items = ['NONE']
    for a in mon_evolution:
        for b, t, v in mon_evolution[a]:
            if t not in evo_types:
                evo_types.append(t)
            if t == 'EVO_ITEM' and v not in evo_items:
                evo_items.append(v)
    n_evo_types = len(evo_types)
    n_evo_items = len(evo_items)
    print("... done; noted %d evo types and %d evo items" % (n_evo_types, n_evo_items))

    # Construct our model.
    h = ae_hidden_dim // 2
    print("Input size: %d; one hidden layer of dimension %d" % (ae_hidden_dim, h))
    model = EvoNet(ae_hidden_dim, h,
                   ae_hidden_dim, n_evo_types, n_evo_items)

    # Given mon embeddings, note input/output pairs based on evolution data.
    idx_in = []
    idx_out = []
    evo_type_out = []
    evo_item_out = []
    evo_level_out = []
    for idx in range(len(mon_list)):
        a = mon_list[idx]
        if a in mon_evolution:
            for b, t, v in mon_evolution[a]:
                idx_in.append(idx)
                idx_out.append(mon_list.index(b))
                evo_type_out.append(evo_types.index(t))
                evo_item_out.append(evo_items.index(v) if t == 'EVO_ITEM' else evo_items.index('NONE'))
                evo_level_out.append(float(v) / 100. if t == 'EVO_LEVEL' else 0.)
    x_emb = torch.zeros((len(idx_in), ae_hidden_dim))
    y_emb = torch.zeros((len(idx_in), ae_hidden_dim))
    y_type = torch.zeros(len(idx_in), dtype=torch.long)
    y_item = torch.zeros(len(idx_in), dtype=torch.long)
    y_level = torch.zeros(len(idx_in))

    # Construct our loss function and an Optimizer.
    criterion = EvoReconstructionLoss(evo_items.index('NONE'))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train.
    epochs = 10000
    print_every = int(epochs / 100.)
    mon_every = int(epochs / 10.)
    n_mon_every = 1
    for t in range(epochs + 1):

        # For each epoch, introduce new randomly perturbed inputs/outputs based on VAE encoding.
        for jdx in range(len(idx_in)):
            r = torch.randn(ae_hidden_dim)
            x_emb[jdx, :] = embs_mu[idx_in[jdx]] + r * embs_std[idx_in[jdx]]
            y_emb[jdx, :] = embs_mu[idx_out[jdx]] + r * embs_std[idx_out[jdx]]
            y_type[jdx] = evo_type_out[jdx]
            y_item[jdx] = evo_item_out[jdx]
            y_level[jdx] = evo_level_out[jdx]

        # Forward pass: Compute predicted y by passing x to the model
        emb_pred, type_pred, item_pred, level_pred = model(x_emb)

        # Compute and print loss
        loss = criterion.forward(emb_pred, type_pred, item_pred, level_pred,
                                 y_emb, y_type, y_item, y_level)
        if t % print_every == 0:
            print("epoch %d\tloss %.5f" % (t, loss.item()))

            # Sample some mon
            if t % mon_every == 0:
                with torch.no_grad():
                    # Show sample forward pass.
                    print("Forward pass categorical losses:")
                    criterion.forward(emb_pred, type_pred, item_pred, level_pred,
                                      y_emb, y_type, y_item, y_level,
                                      debug=True)

                    # Show sample evolution mon.
                    print("Sample evolution:")
                    for _ in range(n_mon_every):
                        midx = np.random.choice(list(range(len(mon_metadata))))
                        print("----------Base mon")
                        print(df["species"][midx], "orig")
                        y_base = autoencoder_model.decoder.forward(embs_mu[midx].unsqueeze(0))
                        print_mon_vec(y_base[0], z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      name_len, name_chars,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------Evolved mon")
                        y_evo_emb, y_type_pred, y_item_pred, y_level_pred = model.forward(embs_mu[midx].unsqueeze(0))
                        y_evo_base = autoencoder_model.decoder.forward(y_evo_emb)
                        print("Evo type: %s\nEvo item: %s\nEvo level: %.0f" %
                              (evo_types[int(np.argmax(y_type_pred.detach()))],
                               evo_items[int(np.argmax(y_item_pred.detach()))],
                               y_level_pred[0].item() * 100))
                        print_mon_vec(y_evo_base[0], z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      name_len, name_chars,
                                      egg_group_list, growth_rates_list, gender_ratios_list)

            # Write trained model to file.
            torch.save(model.state_dict(), "%s.%d.model" %
                       (args.output_fn, t))

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_meta_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--input_model_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the trained generator network')
    args = parser.parse_args()

    main(args)
