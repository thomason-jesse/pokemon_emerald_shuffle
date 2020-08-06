import argparse
import json
import numpy as np
import torch

torch.manual_seed(0)

from train_mon_network import Autoencoder, print_mon_vec

emb_w = np.power(2, 3)
lvl_w = np.power(2, 2)
item_w = np.power(2, 1)
type_w = np.power(2, 0)


class EvoNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 n_evo_types, n_evo_items,
                 device):
        super(EvoNet, self).__init__()

        # Just do one linear transformation.
        self.emb_linear = torch.nn.Linear(input_dim, output_dim).to(device)

        # Prediction layers will take in B - A and produce output.
        self.evo_type_linear = torch.nn.Linear(output_dim, n_evo_types).to(device)
        self.evo_item_linear = torch.nn.Linear(output_dim, n_evo_items).to(device)
        self.evo_level_linear = torch.nn.Linear(output_dim, 1).to(device)

        self.n_evo_types = n_evo_types
        self.n_evo_items = n_evo_items

    def forward(self, x):
        y_emb = self.emb_linear(x)
        y_type = self.evo_type_linear(y_emb - x)  # project down (goes to softmax+CE)
        y_item = self.evo_item_linear(y_emb - x)  # project down (goes to softmax+CE)
        y_level = self.evo_level_linear(y_emb - x)  # project down (use directly for MSE loss)

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
            print("emb loss %.5f(%.5f)" % (emb_loss.item(), emb_w * emb_loss.item()))
            print("type loss %.5f(%.5f)" % (type_loss.item(), type_w * type_loss.item()))
            print("item loss %.5f(%.5f)" % (item_loss.item(), item_w * item_loss.item()))
            print("level loss %.5f(%.5f)" % (level_loss.item(), lvl_w * level_loss.item()))
        return emb_w * emb_loss + type_w * type_loss + item_w * item_loss + lvl_w * level_loss


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Reading mon model metadata from '%s'..." % args.input_meta_fn)
    with open(args.input_meta_fn, 'r') as f:
        d = json.load(f)
        mon_list = d["mon_list"]
        z_mean = d["z_mean"]
        z_std = d["z_std"]
        mon_evolution = d["mon_evolution"]
        ev_types_list = d["ev_types_list"]
        ev_items_list = d["ev_items_list"]
        ae_input_dim = d["input_dim"]
        ae_hidden_dim = d["hidden_dim"]
        int_data = d["int_data"]
        type_list = d["type_list"]
        abilities_list = d["abilities_list"]
        moves_list = d["moves_list"]
        tmhm_moves_list = d["tmhm_moves_list"]
        egg_group_list = d["egg_group_list"]
        growth_rates_list = d["growth_rates_list"]
        gender_ratios_list = d["gender_ratios_list"]
        ae_x = torch.tensor(np.asarray(d["x_input"]), dtype=torch.float32).to(device)
    print("... done")

    print("Initializing trained mon Autoencoder model from '%s'..." % args.input_model_fn)
    autoencoder_model = Autoencoder(ae_input_dim, ae_hidden_dim,
                                    len(int_data), len(type_list), len(abilities_list),
                                    len(moves_list), len(tmhm_moves_list),
                                    len(egg_group_list), len(growth_rates_list), len(gender_ratios_list),
                                    device).to(device)
    autoencoder_model.load_state_dict(torch.load(args.input_model_fn))
    print("... done")

    # Create mon embeddings by running them through the autoencoder encoder layer.
    print("Creating input representations for mon embeddings, evo types, evo items...")
    with torch.no_grad():
        autoencoder_model.eval()
        embs_mu, embs_std = autoencoder_model.encode(ae_x)

    ev_items_list.append('NONE')
    n_evo_types = len(ev_types_list)
    n_evo_items = len(ev_items_list)
    print("... done; noted %d evo types and %d evo items" % (n_evo_types, n_evo_items))

    # Construct our model.
    h = ae_hidden_dim // 2
    model = EvoNet(ae_hidden_dim,
                   ae_hidden_dim, n_evo_types, n_evo_items,
                   device).to(device)

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
                evo_type_out.append(ev_types_list.index(t) if t in ev_types_list
                                    else ev_types_list.index('INFREQUENT'))
                if t == 'EVO_ITEM':
                    evo_item_out.append(ev_items_list.index(v) if v in ev_items_list
                                        else ev_items_list.index('INFREQUENT'))
                else:
                    evo_item_out.append(ev_items_list.index('NONE'))
                evo_level_out.append(float(v) / 100. if t == 'EVO_LEVEL' else 0.)
    x_emb = torch.zeros((len(idx_in), ae_hidden_dim)).to(device)
    x_std = torch.zeros((len(idx_in), ae_hidden_dim)).to(device)
    y_emb = torch.zeros((len(idx_in), ae_hidden_dim)).to(device)
    y_type = torch.zeros(len(idx_in), dtype=torch.long).to(device)
    y_item = torch.zeros(len(idx_in), dtype=torch.long).to(device)
    y_level = torch.zeros(len(idx_in)).to(device)

    # Construct our loss function and an Optimizer.
    criterion = EvoReconstructionLoss(ev_items_list.index('NONE'))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    if args.input_model_pref:
        print("Loading initial model and optimizer weights from prefix '%s'..." % args.input_model_pref)
        model.load_state_dict(torch.load("%s.model" % args.input_model_pref))
        optimizer.load_state_dict(torch.load("%s.opt" % args.input_model_pref))
        print("... done")

    # Prep inputs.
    for jdx in range(len(idx_in)):
        x_emb[jdx, :] = embs_mu[idx_in[jdx]]
        x_std[jdx, :] = embs_std[idx_in[jdx]]
        y_emb[jdx, :] = embs_mu[idx_out[jdx]]
        y_type[jdx] = evo_type_out[jdx]
        y_item[jdx] = evo_item_out[jdx]
        y_level[jdx] = evo_level_out[jdx]

    # Train.
    epochs = 10000
    print_every = int(epochs / 100.)
    mon_every = int(epochs / 10.)
    n_mon_every = 1
    least_loss = None
    for t in range(epochs + 1):
        model.train()

        # Forward pass: Compute predicted y by passing x to the model
        emb_pred, type_pred, item_pred, level_pred = model(x_emb + torch.rand_like(x_std) * x_std)

        # Compute and print loss
        loss = criterion.forward(emb_pred, type_pred, item_pred, level_pred,
                                 y_emb, y_type, y_item, y_level)
        if t % print_every == 0:
            print("epoch %d\tloss %.5f" % (t, loss.item()))

            # Sample some mon
            if t % mon_every == 0:
                with torch.no_grad():
                    model.eval()
                    # Show sample forward pass.
                    print("Forward pass categorical losses:")
                    criterion.forward(emb_pred, type_pred, item_pred, level_pred,
                                      y_emb, y_type, y_item, y_level,
                                      debug=True)

                    # Show sample evolution mon.
                    print("Sample evolution:")
                    for _ in range(n_mon_every):
                        midx = np.random.choice(list(range(len(mon_list))))
                        print("----------Base mon")
                        print(mon_list[midx], "orig")
                        y_base = autoencoder_model.decode(embs_mu[midx].unsqueeze(0))
                        print_mon_vec(y_base[0].cpu(), z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      egg_group_list, growth_rates_list, gender_ratios_list)
                        print("----------Evolved mon")
                        y_evo_emb, y_type_pred, y_item_pred, y_level_pred = model.forward(embs_mu[midx].unsqueeze(0))
                        y_evo_base = autoencoder_model.decode(y_evo_emb)
                        print("Evo type: %s\nEvo item: %s\nEvo level: %.0f" %
                              (ev_types_list[int(np.argmax(y_type_pred.cpu()))],
                               ev_items_list[int(np.argmax(y_item_pred.cpu()))],
                               y_level_pred[0].item() * 100))
                        print_mon_vec(y_evo_base[0].cpu(), z_mean, z_std,
                                      type_list, abilities_list,
                                      moves_list, tmhm_moves_list,
                                      egg_group_list, growth_rates_list, gender_ratios_list)

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
    parser.add_argument('--input_meta_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--input_model_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the trained generator network')
    parser.add_argument('--input_model_pref', type=str,
                        help='input model/optimizer weights prefix to continue training')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='whether to print a bunch')
    args = parser.parse_args()

    main(args)
