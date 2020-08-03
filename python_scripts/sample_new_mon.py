import argparse
import copy
import json
import random
import numpy as np
import pandas as pd
import torch

from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

torch.manual_seed(0)

from train_mon_network import Autoencoder, int_data
from train_evo_network import EvoNet

stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']
ev_data = ['evYield_HP', 'evYield_Attack', 'evYield_Defense',
           'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense']
other_int_data_base_lowerbound = ['expYield']
other_int_data_base_upperbound = ['catchRate']
global_vars = {'pop_idx': 0}


def get_mon_by_species_id(mon_list, sid):
    for m in mon_list:
        if m['species'] == sid:
            return m
    return None


def create_valid_mon(y, mon_metadata,
                     z_mean, z_std, int_data_ranges,
                     type_list, preserve_primary_type,
                     abilities_list, mon_to_infrequent_abilities,
                     move_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                     tmhm_move_list,
                     egg_groups_list, growth_rates_list, gender_ratios_list):
    idx = 0
    d = {'species': global_vars['pop_idx']}

    # Numeric stats.
    max_ev = None
    max_ev_logit = None
    for jdx in range(idx, idx+len(int_data)):
        if int_data[jdx - idx] in ev_data:
            if max_ev is None or max_ev_logit < y[jdx]:
                max_ev_logit = y[jdx]
                max_ev = int_data[jdx - idx]
        d[int_data[jdx - idx]] = int(y[jdx] * z_std[jdx - idx] + z_mean[jdx - idx])
        d[int_data[jdx - idx]] = max(d[int_data[jdx - idx]], int_data_ranges[jdx - idx][0])
        d[int_data[jdx - idx]] = min(d[int_data[jdx - idx]], int_data_ranges[jdx - idx][1])
    # Ensure at least one EV is earned.
    if sum([d[ev] for ev in ev_data]) == 0:
        d[max_ev] = 1
    idx += len(int_data)

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
    if preserve_primary_type is None:
        type1_idx = int(np.argmax(type_v))
    else:
        type1_idx = type_list.index(preserve_primary_type)
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

    # Infrequent abilities.
    for jdx in range(2):
        ab = d['abilities'][jdx]
        if ab == 'INFREQUENT':
            # Decode unique move as the one whose owner has the closest stats.
            nn = None
            min_sp = 0
            for a in mon_to_infrequent_abilities:
                sp = sum(
                    [abs(mon_metadata[a][stat_data[sidx]] - ((d[stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]))
                     for sidx in range(len(stat_data))])
                if nn is None or sp < min_sp:
                    nn = a
                    min_sp = sp
            ab = mon_to_infrequent_abilities[nn]
        d['abilities'][jdx] = ab

    # Levelup moveset.
    confs = []
    moves = []
    probs = torch.sigmoid(y[idx:idx + len(move_list)]).numpy()
    idx += len(move_list)
    for jdx in range(len(probs)):
        moves.append(move_list[jdx])
        confs.append(probs[jdx])  # Just report confidence
    d['levelup_moveset'] = []
    for jdx in np.argsort(confs)[::-1]:
        noise = np.random.random()
        noise -= n_evolutions  # adjust level at which move is learned by std dev per evolution upcoming
        lvl = int(np.round(noise * lvl_move_std[jdx] + lvl_move_avg[jdx]))
        lvl = max(1, min(100, lvl))
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
            mv = mon_to_infrequent_moves[nn]
        d['levelup_moveset'].append([lvl, mv, confs[jdx]])
    # Sort by level learn order.
    d['levelup_moveset'].sort(key=lambda x: x[0])

    # TMHM moveset.
    probs = torch.sigmoid(y[idx:idx+len(tmhm_move_list)]).numpy()
    idx += len(tmhm_move_list)
    d['mon_tmhm_moveset'] = [[tmhm_move_list[jdx], probs[jdx]]
                             for jdx in range(len(tmhm_move_list))]

    # Egg groups.
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
           ev_types_list, mon_to_infrequent_ev_types, ev_items_list, mon_to_infrequent_ev_items, min_evo_lvl,
           mon_metadata,
           z_mean, z_std, int_data_ranges,
           type_list, abilities_list, mon_to_infrequent_abilities,
           moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
           tmhm_moves_list, max_tmhm_moves,
           egg_group_list, growth_rates_list, gender_ratios_list):

    if base_mon['evo_stages'] == 0:
        return [x[0].detach().cpu()], [base_mon]

    base_mon['evolution'] = []

    # About 95% of evolutions are one-to-one, so sample how many we'll have here based on that.
    n_splits = 1
    while np.random.random() < 0.05:  # 5% chance of splitting
        n_splits += 1

    y_evo_embs = []
    y_mons = []
    ev_vals = []
    ev_types = []
    for split_idx in range(n_splits):
        if split_idx == 0:
            evo_model.eval()
        else:
            evo_model.train()  # re-introduces noise from VAE to get slightly different evolutions

        h_mu, h_std, y_evo_emb, y_type_pred, y_item_pred, y_level_pred = evo_model.forward(x)
        ev_type_logits = y_type_pred.cpu().detach()[0]
        ev_type = ev_types_list[int(np.argmax(ev_type_logits))]

        y_evo_base = autoencoder_model.decode(y_evo_emb)
        # Set data ranges for integer data based on base 'mon.
        int_data_ranges_base = copy.deepcopy(int_data_ranges)
        for idx in range(len(int_data)):
            if int_data[idx] in stat_data or int_data[idx] in ev_data or int_data[idx] in other_int_data_base_lowerbound:
                int_data_ranges_base[idx][0] = base_mon[int_data[idx]]
            if int_data[idx] in other_int_data_base_upperbound:
                int_data_ranges_base[idx][1] = base_mon[int_data[idx]]
        # Ensure primary type is preserved on evolution, except Normal.
        preserve_primary_type = base_mon['type1'] if base_mon['type1'] != "TYPE_NORMAL" else None
        y_mon = create_valid_mon(y_evo_base[0].cpu().detach(), mon_metadata,
                                 z_mean, z_std, int_data_ranges_base,
                                 type_list, preserve_primary_type,
                                 abilities_list, mon_to_infrequent_abilities,
                                 moves_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                                 tmhm_moves_list,
                                 egg_group_list, growth_rates_list, gender_ratios_list)
        y_mon['evo_stages'] = base_mon['evo_stages'] - 1
        y_mon['evo_from'] = base_mon['species']

        # If we've already evolved by level in this split, so we need a new evo type.
        if ev_type == 'EVO_LEVEL' and 'EVO_LEVEL' in ev_types:
            ev_type_logits[ev_types_list.index('EVO_LEVEL')] = -float('inf')
            ev_type = ev_types_list[int(np.argmax(ev_type_logits))]

        # Assign infrequent evo type to whatever original mon this sampled one's stats are closest to.
        if ev_type == 'INFREQUENT':
            # Decode unique move as the one whose owner has the closest stats.
            nn = None
            min_sp = 0
            for a in mon_to_infrequent_ev_types:
                sp = sum(
                    [abs(mon_metadata[a][stat_data[sidx]] - ((y_mon[stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]))
                     for sidx in range(len(stat_data))])
                if nn is None or sp < min_sp:
                    nn = a
                    min_sp = sp
            ev_type = mon_to_infrequent_ev_types[nn]

        # Must be on second or further split to do these special evolutions.
        if split_idx == 0 and ev_type in ['EVO_LEVEL_SILCOON', 'EVO_LEVEL_CASCOON',
                                          'EVO_LEVEL_NINJASK', 'EVO_LEVEL_SHEDINJA',
                                          'EVO_LEVEL_ATK_EQ_DEF', 'EVO_LEVEL_ATK_LT_DEF', 'EVO_LEVEL_ATK_GT_DEF']:
            ev_type = 'EVO_LEVEL'
        # Must be on the third split for this special ev, otherwise back down.
        elif split_idx == 1 and ev_type == 'EVO_LEVEL_ATK_EQ_DEF':
            ev_type = random.choice(['EVO_LEVEL_ATK_LT_DEF', 'EVO_LEVEL_ATK_GT_DEF'])

        if ev_type == 'EVO_LEVEL':
            ev_val = int(np.round(y_level_pred[0].item() * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
        elif ev_type in ['EVO_LEVEL_SILCOON', 'EVO_LEVEL_CASCOON']:
            # Guaranteed to have split_idx - 1 valid.
            # Make previous split evolve by this method and set both to this level.
            prev_ev_type = 'EVO_LEVEL_CASCOON' if ev_type == 'EVO_LEVEL_SILCOON' else 'EVO_LEVEL_SILCOON'
            ev_types[split_idx - 1] = prev_ev_type
            base_mon['evolution'][-1][1] = prev_ev_type
            ev_val = int(np.round(y_level_pred[0].item() * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            ev_vals[split_idx - 1] = ev_val
            base_mon['evolution'][-1][2] = ev_val
        elif ev_type in ['EVO_LEVEL_NINJASK', 'EVO_LEVEL_SHEDINJA']:
            # Guaranteed to have split_idx - 1 valid.
            # Make previous split evolve by this method and set both to this level.
            prev_ev_type = 'EVO_LEVEL_SHEDINJA' if ev_type == 'EVO_LEVEL_NINJASK' else 'EVO_LEVEL_NINJASK'
            ev_types[split_idx - 1] = prev_ev_type
            base_mon['evolution'][-1][1] = prev_ev_type
            ev_val = int(np.round(y_level_pred[0].item() * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            ev_vals[split_idx - 1] = ev_val
            base_mon['evolution'][-1][2] = ev_val
        elif ev_type == 'EVO_FRIENDSHIP' or ev_type == 'EVO_FRIENDSHIP_DAY' or ev_type == 'EVO_FRIENDSHIP_NIGHT':
            # Zero value always for friendship-based evs.
            ev_val = 0
        elif ev_type in ['EVO_LEVEL_ATK_LT_DEF', 'EVO_LEVEL_ATK_GT_DEF']:
            # Guaranteed to have split_idx - 1 valid.
            prev_ev_type = 'EVO_LEVEL_ATK_LT_DEF' if ev_type == 'EVO_LEVEL_ATK_GT_DEF' else 'EVO_LEVEL_ATK_GT_DEF'
            ev_types[split_idx - 1] = prev_ev_type
            base_mon['evolution'][-1][1] = prev_ev_type
            ev_val = int(np.round(y_level_pred[0].item() * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            ev_vals[split_idx - 1] = ev_val
            base_mon['evolution'][-1][2] = ev_val
        elif ev_type == 'EVO_LEVEL_ATK_EQ_DEF':
            # Guaranteed to have split_idx - 2 valid.
            prev_ev_types = random.choice([['EVO_LEVEL_ATK_LT_DEF', 'EVO_LEVEL_ATK_GT_DEF'],
                                           ['EVO_LEVEL_ATK_GT_DEF', 'EVO_LEVEL_ATK_LT_DEF']])
            ev_val = int(np.round(y_level_pred[0].item() * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            for jdx in range(2):
                ev_types[split_idx - (1 + jdx)] = prev_ev_types[jdx]
                base_mon['evolution'][-(1 + jdx)][1] = prev_ev_types[jdx]
                ev_vals[split_idx - (1 + jdx)] = ev_val
                base_mon['evolution'][-(1 + jdx)][2] = ev_val
        elif ev_type == 'EVO_BEAUTY':
            ev_val = int(np.round(y_level_pred[0].item() * 100))
            ev_val = max(1, min(ev_val, 170))  # Only one beauty evolver, so we know the upper bound from the game.
        elif ev_type == 'EVO_ITEM':
            ev_item_logits = y_item_pred.cpu().detach()[0]
            ev_val = ev_items_list[int(np.argmax(ev_item_logits))]

            # If we've already chosen this item while evolving by type, need a new distinct item.
            if np.any([ev_vals[idx] == ev_val for idx in range(len(ev_vals))
                       if ev_types[idx] == 'EVO_ITEM']):
                ev_item_logits[ev_items_list.index(ev_val)] = -float('inf')
                ev_val = ev_items_list[int(np.argmax(ev_item_logits))]

            # Infrequent items.
            if ev_val == 'INFREQUENT':
                # Decode unique move as the one whose owner has the closest stats.
                nn = None
                min_sp = 0
                for a in mon_to_infrequent_ev_items:
                    sp = sum(
                        [abs(mon_metadata[a][stat_data[sidx]] - (
                                    (y_mon[stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]))
                         for sidx in range(len(stat_data))])
                    if nn is None or sp < min_sp:
                        nn = a
                        min_sp = sp
                ev_val = mon_to_infrequent_ev_items[nn]

            # Stones force secondary type change from some items.
            forced_type2 = {'ITEM_FIRE_STONE': 'TYPE_FIRE', 'ITEM_DRAGON_SCALE': 'TYPE_DRAGON',
                            'ITEM_METAL_COAT': 'TYPE_STEEL', 'ITEM_WATER_STONE': 'TYPE_WATER',
                            'ITEM_THUNDER_STONE': 'TYPE_ELECTRIC', 'ITEM_LEAF_STONE': 'TYPE_GRASS'}
            for it in forced_type2:
                if ev_val == it:
                    if y_mon['type1'] != forced_type2[it]:
                        y_mon['type2'] = forced_type2[it]

        else:
            ev_val = None
            print("Unhandled ev type '%s'" % ev_type)

        base_mon['evolution'].append([global_vars['pop_idx'], ev_type, ev_val])
        global_vars['pop_idx'] += 1

        y_evo_embs.append(y_evo_emb)
        y_mons.append(y_mon)
        ev_vals.append(ev_val)
        ev_types.append(ev_type)

    evs = [evolve(y_evo_embs[idx], evo_model, autoencoder_model, y_mons[idx],
                  ev_types_list, mon_to_infrequent_ev_types,
                  ev_items_list, mon_to_infrequent_ev_items, 2*ev_vals[idx] if 'EVO_LEVEL' in ev_types[idx] else 5,
                  mon_metadata,
                  z_mean, z_std, int_data_ranges,
                  type_list, abilities_list, mon_to_infrequent_abilities,
                  moves_list,
                  max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                  tmhm_moves_list, max_tmhm_moves,
                  egg_group_list, growth_rates_list, gender_ratios_list)
           for idx in range(n_splits)]
    flat_evolved_list = [item for sublist in [ev[1] for ev in evs] for item in sublist]
    flat_evolved_embs_list = [item for sublist in [ev[0] for ev in evs] for item in sublist]
    return [x.detach().cpu()] + flat_evolved_embs_list, [base_mon] + flat_evolved_list


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
        infrequent_abilities = d["infrequent_abilities_map"]
        ae_x = torch.tensor(np.asarray(d["x_input"]), dtype=torch.float32).to(device)
    print("... done")

    print("Noting ranges of various data values and getting needed averages for sampling...")
    int_data_ranges = [[int(min([mon_metadata[m][int_data[didx]] * z_std[didx] + z_mean[didx] for m in mon_list])),
                        int(max([mon_metadata[m][int_data[didx]] * z_std[didx] + z_mean[didx] for m in mon_list]))]
                       for didx in range(len(int_data))]
    max_learned_moves = max([len(mon_levelup_moveset[m]) for m in mon_list])
    avg_learned_moves = np.average([len(mon_levelup_moveset[m]) for m in mon_list])
    max_tmhm_moves = max([len(mon_tmhm_moveset[m]) for m in mon_list])
    avg_tmhm_moves = np.average([len(mon_tmhm_moveset[m]) for m in mon_list])
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
    mon_to_infrequent_ev_types = {infrequent_ev_types[m][0]: m for m in infrequent_ev_types}
    mon_to_infrequent_abilities = {infrequent_abilities[m][0]: m for m in infrequent_abilities}
    mon_to_infrequent_ev_items = {infrequent_ev_items[m][0]: m for m in infrequent_ev_items}
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
    autoencoder_model.eval()
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
    evo_model.eval()
    print("... done")

    print("Sampling...")
    new_mon = []
    new_mon_embs = np.zeros((len(mon_list), ae_hidden_dim))
    while len(new_mon) < len(mon_list):
        # Sample from a distribution based on the center of the real 'mon embeddings.
        z = torch.mean(embs_mu.cpu(), dim=0) + torch.std(embs_mu.cpu(), dim=0) * torch.randn(ae_hidden_dim)
        y = autoencoder_model.decode(z.unsqueeze(0).to(device))
        m = create_valid_mon(y[0].cpu().detach(), mon_metadata,
                             z_mean, z_std, int_data_ranges,
                             type_list, None,
                             abilities_list, mon_to_infrequent_abilities,
                             moves_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                             tmhm_moves_list,
                             egg_group_list, growth_rates_list, gender_ratios_list)
        global_vars['pop_idx'] += 1

        # Evolve.
        if not args.no_evolve:
            evs_emb, evs = evolve(z.unsqueeze(0).to(device), evo_model, autoencoder_model, m,
                         ev_types_list, mon_to_infrequent_ev_types, ev_items_list, mon_to_infrequent_ev_items, 5,
                         mon_metadata,
                         z_mean, z_std, int_data_ranges,
                         type_list, abilities_list, mon_to_infrequent_abilities,
                         moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                         tmhm_moves_list, max_tmhm_moves,
                         egg_group_list, growth_rates_list, gender_ratios_list)
        else:
            evs_emb = [z]
            evs = [m]

        # Add to population.
        if len(new_mon) + len(evs) <= len(mon_list):
            for idx in range(len(evs_emb)):
                new_mon_embs[len(new_mon) + idx, :] = evs_emb[idx]
            new_mon.extend(evs)
        else:
            global_vars['pop_idx'] -= len(evs)
    print("... done; sampled %d mon" % len(mon_list))

    print("Choosing dynamic threshold for learned and TMHM moves to fit averages...")
    lvl_thresh = 0.5
    thresh_bounds = [0., 1.]
    last_avg = None
    done = False
    while not done:
        avg_n_mv = 0
        for m in new_mon:
            n_mv = 1  # Always keep highest conf move regardless of thresh.
            for _, _, conf in m['levelup_moveset'][1:]:
                if conf > lvl_thresh:
                    n_mv += 1
            avg_n_mv += n_mv
        avg_n_mv /= len(new_mon)
        if last_avg is not None and np.isclose(last_avg, avg_n_mv):  # done
            done = True
        elif np.isclose(avg_n_mv, avg_learned_moves):  # done
            done = True
        elif avg_n_mv > avg_learned_moves:  # thresh is too low
            too_low_thresh = lvl_thresh
            lvl_thresh = (thresh_bounds[1] + lvl_thresh) / 2
            thresh_bounds[0] = too_low_thresh
        else:  # thresh is too high
            too_high_thresh = lvl_thresh
            lvl_thresh = (thresh_bounds[0] + lvl_thresh) / 2
            thresh_bounds[1] = too_high_thresh
        last_avg = avg_n_mv
    print("... done; achieved average %.2f(target %.2f) levelup moves with threshold %.4f" %
          (last_avg, avg_learned_moves, lvl_thresh))

    tmhm_thresh = 0.5
    thresh_bounds = [0., 1.]
    last_avg = None
    done = False
    while not done:
        avg_n_mv = 0
        for m in new_mon:
            n_mv = 0
            base_tmhm_moveset = get_mon_by_species_id(new_mon, m['evo_from'])['mon_tmhm_moveset'] if 'evo_from' in m else None
            for idx in range(len(tmhm_moves_list)):
                # Can be learned if this or the base can learn it.
                if (m['mon_tmhm_moveset'][idx][1] > tmhm_thresh or
                        (base_tmhm_moveset is not None and base_tmhm_moveset[idx][1] > tmhm_thresh)):
                    n_mv += 1
            avg_n_mv += n_mv
        avg_n_mv /= len(new_mon)
        if last_avg is not None and np.isclose(last_avg, avg_n_mv):  # done
            done = True
        elif np.isclose(avg_n_mv, avg_tmhm_moves):  # done
            done = True
        elif avg_n_mv > avg_tmhm_moves:  # thresh is too low
            too_low_thresh = tmhm_thresh
            tmhm_thresh = (thresh_bounds[1] + tmhm_thresh) / 2
            thresh_bounds[0] = too_low_thresh
        else:  # thresh is too high
            too_high_thresh = tmhm_thresh
            tmhm_thresh = (thresh_bounds[0] + tmhm_thresh) / 2
            thresh_bounds[1] = too_high_thresh
        last_avg = avg_n_mv
    print("... done; achieved average %.2f(target %.2f) tmhm moves with threshold %.4f" %
          (last_avg, avg_tmhm_moves, tmhm_thresh))

    n_to_show = 0  # DEBUG
    for m in new_mon:  # Limit levelup and tmhm moves by new thresholds.
        m['levelup_moveset'].sort(key=lambda x: x[2])  # Sort by confidence
        keep_highest = [0] if 'evo_from' not in m else [0, 1]
        m['levelup_moveset'] = [[m['levelup_moveset'][idx][0], m['levelup_moveset'][idx][1]]
                                for idx in range(len(m['levelup_moveset']))
                                if idx in keep_highest or m['levelup_moveset'][idx][2] > lvl_thresh]
        base_tmhm_moveset = []
        if 'evo_from' in m:
            # Move not learned by base version is learned by this version at maximum of evo level
            base = get_mon_by_species_id(new_mon, m['evo_from'])
            base_tmhm_moveset = base['mon_tmhm_moveset']
            for lidx in range(len(m['levelup_moveset'])):
                lvl, mv = m['levelup_moveset'][lidx]
                learned_by_base = np.any([base['levelup_moveset'][ljdx][1] == mv
                                          for ljdx in range(len(base['levelup_moveset']))])
                if not learned_by_base:  # base version can't learn this, so learn upon evolving
                    for species, type, ev_lvl in base['evolution']:
                        if species == m['species'] and type == 'EVO_LEVEL':
                            m['levelup_moveset'][lidx] = [max(lvl, ev_lvl), mv]
        m['levelup_moveset'].sort(key=lambda x: x[0])  # Re-sort by level.
        m['levelup_moveset'][0][0] = 1  # Learn first move at birth regardless.
        m['mon_tmhm_moveset'] = [m['mon_tmhm_moveset'][idx][0] for idx in range(len(m['mon_tmhm_moveset']))
                                 if m['mon_tmhm_moveset'][idx][1] > tmhm_thresh or
                                 # By now, base tmhm moveset is just a list of moves, not including confs.
                                 m['mon_tmhm_moveset'][idx][0] in base_tmhm_moveset]

        # DEBUG
        if 'evolution' in m:
            n_to_show += len(m['evolution']) + 1
        if n_to_show > 0:
            n_to_show -= 1
            print(m)
            _ = input()  # DEBUG

    # Visualize sampled mon embeddings based on their NNs.
    print("Drawing TSNE visualization of new sampled mon embeddings...")
    tsne = TSNE(random_state=1, n_iter=1000, metric="euclidean",
                learning_rate=1000)  # changed from default bc of outliers
    mon_icons = [d['mon_metadata'][new_mon[idx]['nn']]["icon"] for idx in range(len(new_mon))]
    if ae_hidden_dim > 2:
        embs_2d = tsne.fit_transform(new_mon_embs)
    elif ae_hidden_dim == 2:
        embs_2d = new_mon_embs
    else:  # ae_hidden_dim == 1
        embs_2d = np.zeros(shape=(len(new_mon_embs), 2))
        embs_2d[:, 0] = new_mon_embs[:, 0]
        embs_2d[:, 1] = new_mon_embs[:, 0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(embs_2d[:, 0], embs_2d[:, 1], alpha=.1)
    paths = mon_icons
    for x0, y0, path in zip(embs_2d[:, 0], embs_2d[:, 1], paths):
        img = plt.imread(path)
        ab = AnnotationBbox(OffsetImage(img[:32, :32, :]),
                            (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.savefig("%s.sampled_mon_embeddings.pdf" % args.output_fn,
                bbox_inches='tight')
    print("... done")

    # Replace species ids with species names for the purpose of swapping/ROM rewriting.
    print("Replacing global ids with species ids for ROM map, then writing data to file...")
    for idx in range(len(new_mon)):
        new_mon[idx]['species'] = mon_list[new_mon[idx]['species']]
        if 'evolution' in new_mon[idx]:
            for jdx in range(len(new_mon[idx]['evolution'])):
                new_mon[idx]['evolution'][jdx][0] = mon_list[new_mon[idx]['evolution'][jdx][0]]

    # Output sampled mon structure into .json expected by the swapping script.
    d = {'mon_metadata': {}, 'mon_evolution': {}, 'mon_levelup_moveset': {}, 'mon_tmhm_moveset': {}}
    data_to_ignore = ['levelup', 'name_chars', 'tmhm', 'evo_stages', 'levelup_moveset', 'evolution', 'n_evolutions']
    for m in new_mon:
        d['mon_metadata'][m['species']] = {}
        for k in mon_metadata[m['species']].keys():
            if k not in d.keys() and k not in data_to_ignore:
                d['mon_metadata'][m['species']][k] = mon_metadata[m['species']][k]
        for k in m.keys():
            if k not in d.keys() and k not in data_to_ignore:
                d['mon_metadata'][m['species']][k] = m[k]
        d['mon_evolution'][m['species']] = m['evolution'] if 'evolution' in m else []
        d['mon_levelup_moveset'][m['species']] = m['levelup_moveset']
        d['mon_tmhm_moveset'][m['species']] = m['mon_tmhm_moveset']

    # Write to file.
    with open(args.output_fn, 'w') as f:
        json.dump(d, f, indent=2)
    print("... done; wrote to '%s'" % args.output_fn)

    # Quickly analyze type distribution.
    tc = {}
    for m in new_mon:
        t = (m['type1'], m['type2'])
        if t not in tc:
            tc[t] = 0
        tc[t] += 1
    tcs = {k: v / float(len(new_mon)) for k, v in sorted(tc.items(), key=lambda item: item[1], reverse=True)}
    print(tcs)
    tc = {}
    for m in mon_metadata:
        t = (mon_metadata[m]['type1'], mon_metadata[m]['type2'])
        if t not in tc:
            tc[t] = 0
        tc[t] += 1
    tcs = {k: v / float(len(mon_metadata)) for k, v in sorted(tc.items(), key=lambda item: item[1], reverse=True)}
    print(tcs)


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
    parser.add_argument('--no_evolve', action='store_true',
                        help='disable evolutions during sampling')
    args = parser.parse_args()

    main(args)
