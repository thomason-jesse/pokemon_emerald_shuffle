import argparse
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

torch.manual_seed(0)

from train_mon_network import Autoencoder, print_mon_vec


class SpriteNet(torch.nn.Module):
    def __init__(self, input_dim, device):
        super(SpriteNet, self).__init__()
        self.input_dim = input_dim

        self.down = torch.nn.ModuleList([torch.nn.Conv2d(in_channels=3, out_channels=input_dim//8, kernel_size=35).to(device),
                                         torch.nn.Conv2d(in_channels=input_dim // 8, out_channels=input_dim // 4, kernel_size=16).to(device),
                                         torch.nn.Conv2d(in_channels=input_dim // 4, out_channels=input_dim // 2, kernel_size=8).to(device),
                                         torch.nn.Conv2d(in_channels=input_dim // 2, out_channels=128, kernel_size=8).to(device)])

        self.up = torch.nn.ModuleList([torch.nn.ConvTranspose2d(in_channels=input_dim, out_channels=input_dim // 2, kernel_size=8).to(device),
                                       torch.nn.ConvTranspose2d(in_channels=input_dim // 2, out_channels=input_dim // 4, kernel_size=8).to(device),
                                       torch.nn.ConvTranspose2d(in_channels=input_dim // 4, out_channels=input_dim // 8, kernel_size=16).to(device),
                                       torch.nn.ConvTranspose2d(in_channels=input_dim // 8, out_channels=3, kernel_size=35).to(device)])

        # Tanh
        self.tanh = torch.nn.Tanh()

        # Params.
        # self.params = [item for sublist in [l.parameters() for l in self.down + self.up] for item in sublist]

    def forward(self, x_emb, x_im):

        # Down
        h = x_im
        for layer in self.down:
            h = layer(h)
            h = self.tanh(h)
            # print('h down', h.shape)  # DEBUG

        # Dot.
        emb = x_emb.view(x_emb.size(0), self.input_dim, 1, 1)
        # print('view emb', emb.shape)  # DEBUG
        h = h * emb

        # Up
        for layer in self.up:
            h = layer(h)
            h = self.tanh(h)
            # print('h up', h.shape)  # DEBUG

        # To pixel space.
        h = h / 2 + 0.5

        return h


class SpriteReconstructionLoss:
    def __init__(self):
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, pred, target, debug=False):
        l = self.l2_loss(pred, target)

        if debug:
            print("loss %.5f" % l.item())
        return l


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
    autoencoder_model.load_state_dict(torch.load(args.input_model_fn, map_location=torch.device(device)))
    print("... done")

    # Create mon embeddings by running them through the autoencoder encoder layer.
    print("Creating input representations for mon embeddings...")
    with torch.no_grad():
        autoencoder_model.eval()
        embs_mu, embs_std = autoencoder_model.encode(ae_x)
    print("... done")

    # Create sprite embeddings as output layers.
    print("Loading sprite representations...")
    true_front = np.zeros((len(mon_list), 3, 64, 64))
    true_back = np.zeros((len(mon_list), 3, 64, 64))
    true_icon = np.zeros((len(mon_list), 3, 64, 64))
    for idx in range(len(mon_list)):
        m = mon_list[idx][len('SPECIES_'):].lower()
        front = transforms.ToTensor()(Image.open('orig/graphics/%s.front.png' % m).convert("RGB").crop((0, 0, 64, 64)))
        true_front[idx, :, :, :] = front
        back = transforms.ToTensor()(Image.open('orig/graphics/%s.back.png' % m).convert("RGB").crop((0, 0, 64, 64)))
        true_back[idx, :, :, :] = back
        icon = Image.open('orig/graphics/%s.icon.png' % m).convert("RGB").crop((0, 0, 32, 32))
        # resize icon to match front/back sizes so they can all use the same backbone
        icon = transforms.ToTensor()(icon.resize((64, 64)))
        true_icon[idx, :, :, :] = icon
    true_front = torch.tensor(true_front).float().to(device)
    true_back = torch.tensor(true_back).float().to(device)
    true_icon = torch.tensor(true_icon).float().to(device)
    print("... done; tensorized sprites from %d mon" % len(mon_list))

    # Construct our model.
    model = SpriteNet(ae_hidden_dim, device).to(device)

    # Construct our loss function and an Optimizer.
    criterion = SpriteReconstructionLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if args.input_model_pref:
        print("Loading initial model and optimizer weights from prefix '%s'..." % args.input_model_pref)
        model.load_state_dict(torch.load("%s.model" % args.input_model_pref, map_location=torch.device(device)))
        optimizer.load_state_dict(torch.load("%s.opt" % args.input_model_pref, map_location=torch.device(device)))
        print("... done")

    # Train.
    epochs = 10000
    save_every = int(epochs / 1000.)
    print_every = int(epochs / 100.)
    mon_every = int(epochs / 10.)
    least_loss = None
    for t in range(epochs + 1):
        model.train()

        loss = 0
        for target in [true_front, true_back, true_icon]:
            pred = model(embs_mu + torch.rand_like(embs_std) * embs_std, target)
            loss += criterion.forward(pred, target)

        if t % print_every == 0:
            print("epoch %d\tloss %.5f" % (t, loss.item()))

            # Sample some mon
            if t % mon_every == 0:
                with torch.no_grad():
                    model.eval()
                    print("Forward pass losses:")
                    for desc, target in [['front', true_front], ['back', true_back], ['icon', true_icon]]:
                        pred = model(embs_mu, target)
                        print('loss from %s' % desc)
                        criterion.forward(pred, target, debug=True)

                    # Show sample evolution mon.
                    print("Writing sample sprites")
                    midx = np.random.choice(list(range(len(mon_list))))
                    print(mon_list[midx], "sprites output")
                    for s, ts in [['front', true_front], ['back', true_back], ['icon', true_icon]]:
                        pr = model(embs_mu[midx].unsqueeze(0), ts[midx, :].unsqueeze(0))
                        im = transforms.ToPILImage()(pr.squeeze(0).detach().cpu())
                        im.save('%s.sample.%s.png' % (args.output_fn, s), mode='RGB')
                        im = transforms.ToPILImage()(ts[midx].detach().cpu())
                        im.save('%s.true.%s.png' % (args.output_fn, s), mode='RGB')
                    print("... done")
                    # Show sample drawn mon.
                    print("Writing random draw sprites")
                    z = torch.mean(embs_mu.cpu(), dim=0) + torch.std(embs_mu.cpu(), dim=0) * torch.randn(ae_hidden_dim)
                    dists = [np.linalg.norm(z - embs_mu[jdx, :].cpu()) for jdx in range(len(mon_list))]
                    nn_emb_idx = np.argsort(dists)[0]
                    for s, tn in [['front', true_front], ['back', true_back], ['icon', true_icon]]:
                        pr = model(z.unsqueeze(0).to(device), tn[nn_emb_idx, :].unsqueeze(0))
                        im = transforms.ToPILImage()(pr.squeeze(0).detach().cpu())
                        im.save('%s.random.%s.png' % (args.output_fn, s), mode='RGB')
                    print("... done")

        # If this is the least loss achieved, write file.
        if t % save_every == 0 and (least_loss is None or loss < least_loss):
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
                        help='the input file for the trained autoencoder network')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the trained generator network')
    parser.add_argument('--input_model_pref', type=str,
                        help='input model/optimizer weights prefix to continue training')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='whether to print a bunch')
    args = parser.parse_args()

    main(args)
