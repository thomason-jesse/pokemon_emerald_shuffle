import argparse
import json
import numpy as np
import pandas as pd
import torch

torch.manual_seed(0)

from train_mon_network import Autoencoder, int_data
from train_evo_network import EvoNet

stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']
global_vars = {'pop_idx': 0}


def create_valid_mon(y, mon_metadata,
                     z_mean, z_std, int_data_ranges,
                     type_list, abilities_list,
                     move_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                     tmhm_move_list, max_tmhm_moves,
                     egg_groups_list, growth_rates_list, gender_ratios_list,
                     min_move_level=1):
    idx = 0
    d = {'species': global_vars['pop_idx']}

    # Numeric stats.
    for jdx in range(idx, idx+len(int_data)):
        d[int_data[jdx - idx]] = int(y[jdx] * z_std[jdx - idx] + z_mean[jdx - idx])
        d[int_data[jdx - idx]] = max(d[int_data[jdx - idx]], int_data_ranges[jdx - idx][0])
        d[int_data[jdx - idx]] = min(d[int_data[jdx - idx]], int_data_ranges[jdx - idx][1])
    idx = len(int_data)

    # Nearest stat neighbor.
    nn = None
    min_sp = 0
    for a in mon_metadata:
        sp = sum([abs(mon_metadata[a][stat_data[sidx]] - ((d[stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]))
                      for sidx in range(len(stat_data))])
        if nn is None or sp < min_sp:
            nn = a
            min_sp = sp
    d['nn'] = nn

    # Types.
    type_v = list(y[idx:idx + len(type_list)].detach())
    type1_idx = int(np.argmax(type_v))
    type_v[type1_idx] = -float('inf')
    type2_idx = int(np.argmax(type_v))
    idx += len(type_list)
    if type_list[type1_idx] == 'TYPE_NONE':
        d['type1'] = d['type2'] = type_list[type2_idx]
    elif type_list[type2_idx] == 'TYPE_NONE':
        d['type1'] = d['type2'] = type_list[type1_idx]
    else:
        d['type1'] = type_list[type1_idx]
        d['type2'] = type_list[type2_idx]

    # N evolutions.
    n_evolutions = np.argmax(y[idx:idx+3])
    idx += 3
    d['evo_stages'] = int(n_evolutions)

    # Abilities.
    ability_v = list(y[idx:idx + len(abilities_list)].detach())
    ability1_idx = int(np.argmax(ability_v))
    ability_v[ability1_idx] = -float('inf')
    ability2_idx = int(np.argmax(ability_v))
    idx += len(abilities_list)
    if abilities_list[ability1_idx] == 'ABILITY_NONE':
        d['abilities'] = [abilities_list[ability2_idx], abilities_list[ability1_idx]]
    else:
        d['abilities'] = [abilities_list[ability1_idx], abilities_list[ability2_idx]]

    # Levelup moveset.
    levels = []
    moves = []
    probs = torch.sigmoid(y[idx:idx + len(move_list)]).numpy()
    idx += len(move_list)
    for jdx in range(len(probs)):
        moves.append(move_list[jdx])
        levels.append(probs[jdx])  # Just report confidence
    d['levelup_moveset'] = []
    for jdx in np.argsort(levels)[::-1]:
        if (len(d['levelup_moveset']) == 0 or
                levels[jdx] >= torch.sigmoid(torch.tensor([0.5], dtype=torch.float64)).item()):
            lvl = int(np.round(np.random.random() * lvl_move_std[jdx] + lvl_move_avg[jdx]))
            lvl = max(min_move_level, min(100, lvl))
            mv = move_list[jdx]
            if mv == 'INFREQUENT':
                # Decode unique move as the one whose owner has the closest stats.
                nn = None
                min_sp = 0
                for a in mon_to_infrequent_moves:
                    sp = sum(
                        [abs(mon_metadata[a][stat_data[sidx]] - ((d[stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]))
                         for sidx in range(len(stat_data))])
                    if nn is None or sp < min_sp:
                        nn = a
                        min_sp = sp
                mv = mon_to_infrequent_moves[a]
            d['levelup_moveset'].append([lvl, mv])
        if len(d['levelup_moveset']) == max_learned_moves:
            break
    # Earliest learned move should be at birth, and levels sorted in order.
    d['levelup_moveset'].sort(key=lambda x: x[0])
    d['levelup_moveset'][0][0] = min_move_level

    # TMHM moveset.
    probs = torch.sigmoid(y[idx:idx+len(tmhm_move_list)]).numpy()
    idx += len(tmhm_move_list)
    d['mon_tmhm_moveset'] = []
    for jdx in np.argsort(probs)[::-1]:
        if probs[jdx] >= torch.sigmoid(torch.tensor([0.5], dtype=torch.float64)).item():
            d['mon_tmhm_moveset'].append(tmhm_move_list[jdx])
        if len(d['mon_tmhm_moveset']) == max_tmhm_moves:
            break

    # Egg groups.
    # TODO: could make it so mon can be in a single egg group. Who cares.
    egg_v = list(y[idx:idx + len(egg_groups_list)].detach())
    egg1_idx = int(np.argmax(egg_v))
    egg_v[egg1_idx] = -float('inf')
    egg2_idx = int(np.argmax(egg_v))
    idx += len(egg_groups_list)
    d['eggGroups'] = [egg_groups_list[egg1_idx], egg_groups_list[egg2_idx]]

    # Growth rate.
    gr_idx = np.argmax(y[idx:idx + len(growth_rates_list)])
    idx += len(growth_rates_list)
    d['growthRate'] = growth_rates_list[gr_idx]

    # Gender ratio.
    gd_idx = np.argmax(y[idx:idx + len(gender_ratios_list)])
    idx += len(gender_ratios_list)
    d['genderRatio'] = gender_ratios_list[gd_idx]

    return d


def evolve(x, evo_model, autoencoder_model, base_mon,
           ev_types_list, ev_items_list, min_evo_lvl,
           mon_metadata,
           z_mean, z_std, int_data_ranges,
           type_list, abilities_list,
           moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
           tmhm_moves_list, max_tmhm_moves,
           egg_group_list, growth_rates_list, gender_ratios_list):

    if base_mon['evo_stages'] == 0:
        return [base_mon]

    base_mon['evolution'] = []

    # Evolve into exactly one thing.
    # TODO: get frequency for weirder evo splits (2, 6, etc.) and sample from that here.
    y_evo_emb, y_type_pred, y_item_pred, y_level_pred = evo_model.forward(x)
    ev_type = ev_types_list[int(np.argmax(y_type_pred.cpu().detach()))]
    if ev_type == 'EVO_LEVEL':
        ev_val = int(np.round(y_level_pred[0].item() * 100))
        ev_val = max(min_evo_lvl, min(ev_val, 100))
    elif ev_type == 'EVO_ITEM':
        ev_val = ev_items_list[int(np.argmax(y_item_pred.cpu().detach()))]
    else:
        ev_val = None
        print("Unhandled ev type '%s'" % ev_type)
    y_evo_base = autoencoder_model.decode(y_evo_emb)
    # TODO: set int_data_ranges to prevent devolving stats
    y_mon = create_valid_mon(y_evo_base[0].cpu().detach(), mon_metadata,
                             z_mean, z_std, int_data_ranges,
                             type_list, abilities_list,
                             moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                             tmhm_moves_list, max_tmhm_moves,
                             egg_group_list, growth_rates_list, gender_ratios_list,
                             min_move_level=ev_val if ev_type == 'EVO_LEVEL' else 1)
    y_mon['evo_stages'] = base_mon['evo_stages'] - 1
    base_mon['evolution'].append([global_vars['pop_idx'], ev_type, ev_val])
    global_vars['pop_idx'] += 1

    return [base_mon] + evolve(y_evo_emb, evo_model, autoencoder_model, y_mon,
                               ev_types_list, ev_items_list, 2*ev_val if ev_type == 'EVO_LEVEL' else 5,
                               mon_metadata,
                               z_mean, z_std, int_data_ranges,
                               type_list, abilities_list,
                               moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                               tmhm_moves_list, max_tmhm_moves,
                               egg_group_list, growth_rates_list, gender_ratios_list)


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("Reading all mon model metadata from '%s'..." % args.input_meta_mon_fn)
    with open(args.input_meta_mon_fn, 'r') as f:
        d = json.load(f)
        mon_levelup_moveset = d["mon_levelup_moveset"]
        mon_tmhm_moveset = d["mon_tmhm_moveset"]
    print("... done")

    print("Reading processed mon model metadata from '%s'..." % args.input_meta_network_fn)
    with open(args.input_meta_network_fn, 'r') as f:
        d = json.load(f)
        mon_list = d["mon_list"]
        mon_metadata = d["mon_metadata"]
        z_mean = d["z_mean"]
        z_std = d["z_std"]
        mon_evolution = d["mon_evolution"]
        ev_types_list = d["ev_types_list"]
        infrequent_ev_types = d["infrequent_ev_types"]
        ev_items_list = d["ev_items_list"]
        infrequent_ev_items = d["infrequent_ev_items"]
        infrequent_moves_map = d["infrequent_moves_map"]
        ae_input_dim = d["input_dim"]
        ae_hidden_dim = d["hidden_dim"]
        type_list = d["type_list"]
        abilities_list = d["abilities_list"]
        moves_list = d["moves_list"]
        tmhm_moves_list = d["tmhm_moves_list"]
        egg_group_list = d["egg_group_list"]
        growth_rates_list = d["growth_rates_list"]
        gender_ratios_list = d["gender_ratios_list"]
        ae_x = torch.tensor(np.asarray(d["x_input"]), dtype=torch.float32).to(device)
    print("... done")

    print("Noting ranges of various data values and getting needed averages for sampling...")
    int_data_ranges = [(int(min([mon_metadata[m][int_data[didx]] * z_std[didx] + z_mean[didx] for m in mon_list])),
                        int(max([mon_metadata[m][int_data[didx]] * z_std[didx] + z_mean[didx] for m in mon_list])))
                       for didx in range(len(int_data))]
    max_learned_moves = max([len(mon_levelup_moveset[m]) for m in mon_list])
    max_tmhm_moves = max([len(mon_tmhm_moveset[m]) for m in mon_list])
    lvl_move_entries = {}
    for a in mon_levelup_moveset:
        for level, move in mon_levelup_moveset[a]:
            if move in infrequent_moves_map:
                move = 'INFREQUENT'
            if move not in lvl_move_entries:
                lvl_move_entries[move] = []
            lvl_move_entries[move].append(level)
    lvl_move_avg = [np.average(lvl_move_entries[move]) for move in moves_list]
    lvl_move_std = [np.std(lvl_move_entries[move]) for move in moves_list]
    mon_to_infrequent_moves = {infrequent_moves_map[m][0]: m for m in infrequent_moves_map}

    print("... done")

    print("Initializing trained mon Autoencoder model from '%s'..." % args.input_mon_model_fn)
    autoencoder_model = Autoencoder(ae_input_dim, ae_hidden_dim,
                                    len(int_data), len(type_list), len(abilities_list),
                                    len(moves_list), len(tmhm_moves_list),
                                    len(egg_group_list), len(growth_rates_list), len(gender_ratios_list),
                                    device).to(device)
    autoencoder_model.load_state_dict(torch.load(args.input_mon_model_fn, map_location=torch.device(device)))
    with torch.no_grad():
        autoencoder_model.eval()
        embs_mu, embs_std = autoencoder_model.encode(ae_x)
    print("... done")

    print("Initializing trained mon EvoNet model from '%s'..." % args.input_evo_model_fn)
    ev_items_list.append('NONE')
    n_evo_types = len(ev_types_list)
    n_evo_items = len(ev_items_list)
    h = ae_hidden_dim // 2
    evo_model = EvoNet(ae_hidden_dim, h,
                       ae_hidden_dim, n_evo_types, n_evo_items,
                       device).to(device)
    evo_model.load_state_dict(torch.load(args.input_evo_model_fn, map_location=torch.device(device)))
    print("... done")

    print("Sampling...")
    new_mon = []
    while len(new_mon) < len(mon_list):
        print(len(new_mon), len(mon_list))  # DEBUG

        # Sample from a distribution based on the center of the real 'mon embeddings.
        z = torch.mean(embs_mu.cpu(), dim=0) + torch.std(embs_mu.cpu(), dim=0) * torch.randn(ae_hidden_dim)
        y = autoencoder_model.decode(z.unsqueeze(0).to(device))
        m = create_valid_mon(y[0].cpu().detach(), mon_metadata,
                             z_mean, z_std, int_data_ranges,
                             type_list, abilities_list,
                             moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                             tmhm_moves_list, max_tmhm_moves,
                             egg_group_list, growth_rates_list, gender_ratios_list)
        global_vars['pop_idx'] += 1

        # Evolve.
        evs = evolve(z.unsqueeze(0).to(device), evo_model, autoencoder_model, m,
                     ev_types_list, ev_items_list, 5,
                     mon_metadata,
                     z_mean, z_std, int_data_ranges,
                     type_list, abilities_list,
                     moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                     tmhm_moves_list, max_tmhm_moves,
                     egg_group_list, growth_rates_list, gender_ratios_list)

        # Add to population.
        if len(new_mon) + len(evs) <= len(mon_list):
            new_mon.extend(evs)
            print('\n'.join([str(m) for m in evs]))
            _ = input()
        else:
            global_vars['pop_idx'] -= len(evs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_meta_mon_fn', type=str, required=True,
                        help='the input file for the network metadata')
    parser.add_argument('--input_meta_network_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--input_mon_model_fn', type=str, required=True,
                        help='the trained mon autoencoder weights')
    parser.add_argument('--input_evo_model_fn', type=str, required=True,
                        help='the trained mon evolver weights')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output json for the sampled mon')
    args = parser.parse_args()

    main(args)
