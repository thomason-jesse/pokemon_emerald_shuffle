import argparse
import copy
import json
import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from train_mon_network import Autoencoder, int_data
from train_evo_network import EvoNet
from train_sprite_network import SpriteNet

stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']
ev_data = ['evYield_HP', 'evYield_Attack', 'evYield_Defense',
           'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense']
other_int_data_base_lowerbound = ['expYield']
other_int_data_base_upperbound = ['catchRate']
global_vars = {'pop_idx': 0}
forced_type2 = {'ITEM_FIRE_STONE': 'TYPE_FIRE', 'ITEM_DRAGON_SCALE': 'TYPE_DRAGON',
                'ITEM_METAL_COAT': 'TYPE_STEEL', 'ITEM_WATER_STONE': 'TYPE_WATER',
                'ITEM_THUNDER_STONE': 'TYPE_ELECTRIC', 'ITEM_LEAF_STONE': 'TYPE_GRASS'}

# Bayes hyperparameters.
alpha = 0.55  # weight that type1 distributions contribute.
beta = 0.45  # weight that type2 distributions contribute.
uniform_weight = 0.05  # how much uniform distribution to mix in to smooth out real data.
move_conf_jitter = 0.1  # how much weight to put on random noise in move learnsets off type-based averages


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
                     egg_groups_list,
                     growth_rates_list, preserve_growth_rate,
                     gender_ratios_list):
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
    try:
        type_v = list(y[idx:idx + len(type_list)].detach())
    except AttributeError:
        type_v = y[idx:idx + len(type_list)]
    if preserve_primary_type is None:
        type1_idx = int(np.argmax(type_v))
    else:
        type1_idx = type_list.index(preserve_primary_type)
    type_v[type1_idx] = -float('inf')
    type2_idx = int(np.argmax(type_v))
    type_v[type2_idx] = -float('inf')
    type3_idx = int(np.argmax(type_v))
    idx += len(type_list)
    if type_list[type1_idx] == 'TYPE_NONE':
        d['type1'] = d['type2'] = d['type3'] = type_list[type2_idx]
    elif type_list[type2_idx] == 'TYPE_NONE':
        d['type1'] = d['type2'] = d['type3'] = type_list[type1_idx]
    elif type_list[type3_idx] == 'TYPE_NONE':
        d['type1'] = type_list[type1_idx]
        d['type2'] = d['type3'] = type_list[type2_idx]
    else:
        d['type1'] = type_list[type1_idx]
        d['type2'] = type_list[type2_idx]
        d['type3'] = type_list[type3_idx]

    # N evolutions.
    n_evolutions = np.argmax(y[idx:idx+3])
    idx += 3
    d['evo_stages'] = int(n_evolutions)

    # Abilities.
    try:
        ability_v = list(y[idx:idx + len(abilities_list)].detach())
    except AttributeError:
        ability_v = y[idx:idx + len(abilities_list)]
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
    try:
        probs = torch.sigmoid(y[idx:idx + len(move_list)]).numpy()
    except TypeError:
        probs = np.asarray(y[idx:idx + len(move_list)])
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
    try:
        probs = torch.sigmoid(y[idx:idx+len(tmhm_move_list)]).numpy()
    except TypeError:
        probs = np.asarray(y[idx:idx + len(tmhm_move_list)])
    idx += len(tmhm_move_list)
    d['mon_tmhm_moveset'] = [[tmhm_move_list[jdx], probs[jdx]]
                             for jdx in range(len(tmhm_move_list))]

    # Egg groups.
    try:
        egg_v = list(y[idx:idx + len(egg_groups_list)].detach())
    except AttributeError:
        egg_v = y[idx:idx + len(egg_groups_list)]
    egg1_idx = int(np.argmax(egg_v))
    egg_v[egg1_idx] = -float('inf')
    egg2_idx = int(np.argmax(egg_v))
    idx += len(egg_groups_list)
    d['eggGroups'] = [egg_groups_list[egg1_idx], egg_groups_list[egg2_idx]]

    # Growth rate.
    if preserve_growth_rate is None:
        gr_idx = np.argmax(y[idx:idx + len(growth_rates_list)])
    else:
        gr_idx = growth_rates_list.index(preserve_growth_rate)
    idx += len(growth_rates_list)
    d['growthRate'] = growth_rates_list[gr_idx]

    # Gender ratio.
    gd_idx = np.argmax(y[idx:idx + len(gender_ratios_list)])
    idx += len(gender_ratios_list)
    d['genderRatio'] = gender_ratios_list[gd_idx]

    return d


def evolve(x, evo_model, autoencoder_model, base_mon, ae_rand_projection,
           ev_types_list, mon_to_infrequent_ev_types, ev_items_list, mon_to_infrequent_ev_items, min_evo_lvl,
           mon_metadata,
           z_mean, z_std, int_data_ranges,
           type_list, abilities_list, mon_to_infrequent_abilities,
           moves_list, max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
           tmhm_moves_list, max_tmhm_moves,
           egg_group_list, growth_rates_list, gender_ratios_list,
           p_ev_type, p_ev_item_g_type, evo_lvl_slope, evo_lvl_intercept,
           int_data_inc_mus_per_type, int_data_inc_stds_per_type,
           p_levelup_g_type, p_tmhm_g_type):

    if base_mon['evo_stages'] == 0:
        try:
            return [x[0].detach().cpu()], [base_mon]
        except AttributeError:
            return [np.matmul(x, ae_rand_projection)], [base_mon]

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
        if evo_model is not None:  # neural draw
            if split_idx > 0:
                x_in = x + torch.rand_like(x) * 0.1  # introduce some gaussian noise to the neural vec to be decoded
            else:
                x_in = x

            y_evo_emb, y_type_pred, y_item_pred, y_level_pred = evo_model.forward(x_in)

            ev_type_logits = y_type_pred.cpu().detach()[0]
            ev_type = ev_types_list[int(np.argmax(ev_type_logits))]

            y_evo_base = autoencoder_model.decode(y_evo_emb, ev_types_list,
                                                  p_ev_type)
            ev_item_logits = y_item_pred.cpu().detach()[0]

            y_evo_vec = y_evo_base[0].cpu().detach()

        else:  # bayes draw
            y_evo_vec, ev_type_logits, ev_item_logits, y_level_pred = \
                sample_bayesian_evo_vector(x, base_mon, ev_vals, ev_types,
                                           type_list, ev_types_list, ev_items_list,
                                           abilities_list, moves_list, tmhm_moves_list,
                                           p_ev_type, p_ev_item_g_type, evo_lvl_slope, evo_lvl_intercept,
                                           int_data_inc_mus_per_type, int_data_inc_stds_per_type,
                                           p_levelup_g_type, p_tmhm_g_type)
            ev_type = ev_types_list[int(np.argmax(ev_type_logits))]
            y_evo_emb = y_evo_vec

        # Set data ranges for integer data based on base 'mon.
        int_data_ranges_base = copy.deepcopy(int_data_ranges)
        for idx in range(len(int_data)):
            if (int_data[idx] in stat_data or int_data[idx] in ev_data or
                    int_data[idx] in other_int_data_base_lowerbound):
                int_data_ranges_base[idx][0] = base_mon[int_data[idx]]
            if int_data[idx] in other_int_data_base_upperbound:
                int_data_ranges_base[idx][1] = base_mon[int_data[idx]]
        # Ensure primary type is preserved on evolution, except Normal.
        preserve_primary_type = base_mon['type1'] if base_mon['type1'] != "TYPE_NORMAL" else None
        y_mon = create_valid_mon(y_evo_vec, mon_metadata,
                                 z_mean, z_std, int_data_ranges_base,
                                 type_list, preserve_primary_type,
                                 abilities_list, mon_to_infrequent_abilities,
                                 moves_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                                 tmhm_moves_list,
                                 egg_group_list,
                                 growth_rates_list, base_mon['growthRate'],
                                 gender_ratios_list)
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
            try:
                ev_val = int(np.round(y_level_pred[0].item() * 100))
            except (IndexError, TypeError):
                ev_val = int(np.round(y_level_pred * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
        elif ev_type in ['EVO_LEVEL_SILCOON', 'EVO_LEVEL_CASCOON']:
            # Guaranteed to have split_idx - 1 valid.
            # Make previous split evolve by this method and set both to this level.
            prev_ev_type = 'EVO_LEVEL_CASCOON' if ev_type == 'EVO_LEVEL_SILCOON' else 'EVO_LEVEL_SILCOON'
            ev_types[split_idx - 1] = prev_ev_type
            base_mon['evolution'][-1][1] = prev_ev_type
            try:
                ev_val = int(np.round(y_level_pred[0].item() * 100))
            except (IndexError, TypeError):
                ev_val = int(np.round(y_level_pred * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            ev_vals[split_idx - 1] = ev_val
            base_mon['evolution'][-1][2] = ev_val
        elif ev_type in ['EVO_LEVEL_NINJASK', 'EVO_LEVEL_SHEDINJA']:
            # Guaranteed to have split_idx - 1 valid.
            # Make previous split evolve by this method and set both to this level.
            prev_ev_type = 'EVO_LEVEL_SHEDINJA' if ev_type == 'EVO_LEVEL_NINJASK' else 'EVO_LEVEL_NINJASK'
            ev_types[split_idx - 1] = prev_ev_type
            base_mon['evolution'][-1][1] = prev_ev_type
            try:
                ev_val = int(np.round(y_level_pred[0].item() * 100))
            except (IndexError, TypeError):
                ev_val = int(np.round(y_level_pred * 100))
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
            try:
                ev_val = int(np.round(y_level_pred[0].item() * 100))
            except (IndexError, TypeError):
                ev_val = int(np.round(y_level_pred * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            ev_vals[split_idx - 1] = ev_val
            base_mon['evolution'][-1][2] = ev_val
        elif ev_type == 'EVO_LEVEL_ATK_EQ_DEF':
            # Guaranteed to have split_idx - 2 valid.
            prev_ev_types = random.choice([['EVO_LEVEL_ATK_LT_DEF', 'EVO_LEVEL_ATK_GT_DEF'],
                                           ['EVO_LEVEL_ATK_GT_DEF', 'EVO_LEVEL_ATK_LT_DEF']])
            try:
                ev_val = int(np.round(y_level_pred[0].item() * 100))
            except (IndexError, TypeError):
                ev_val = int(np.round(y_level_pred * 100))
            ev_val = max(min_evo_lvl, min(ev_val, 100))
            for jdx in range(2):
                ev_types[split_idx - (1 + jdx)] = prev_ev_types[jdx]
                base_mon['evolution'][-(1 + jdx)][1] = prev_ev_types[jdx]
                ev_vals[split_idx - (1 + jdx)] = ev_val
                base_mon['evolution'][-(1 + jdx)][2] = ev_val
        elif ev_type == 'EVO_BEAUTY':
            try:
                ev_val = int(np.round(y_level_pred[0].item() * 100))
            except (IndexError, TypeError):
                ev_val = int(np.round(y_level_pred * 100))
            ev_val = max(1, min(ev_val, 170))  # Only one beauty evolver, so we know the upper bound from the game.
        elif ev_type == 'EVO_ITEM':
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

    evs = [evolve(y_evo_embs[idx], evo_model, autoencoder_model, y_mons[idx], ae_rand_projection,
                  ev_types_list, mon_to_infrequent_ev_types,
                  ev_items_list, mon_to_infrequent_ev_items, 2*ev_vals[idx] if 'EVO_LEVEL' in ev_types[idx] else 5,
                  mon_metadata,
                  z_mean, z_std, int_data_ranges,
                  type_list, abilities_list, mon_to_infrequent_abilities,
                  moves_list,
                  max_learned_moves, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                  tmhm_moves_list, max_tmhm_moves,
                  egg_group_list, growth_rates_list, gender_ratios_list,
                  p_ev_type, p_ev_item_g_type, evo_lvl_slope, evo_lvl_intercept,
                  int_data_inc_mus_per_type, int_data_inc_stds_per_type,
                  p_levelup_g_type, p_tmhm_g_type)
           for idx in range(n_splits)]
    flat_evolved_list = [item for sublist in [ev[1] for ev in evs] for item in sublist]
    flat_evolved_embs_list = [item for sublist in [ev[0] for ev in evs] for item in sublist]
    try:
        x_to_return = x.detach().cpu()
    except AttributeError:
        x_to_return = np.matmul(x, ae_rand_projection)
    return [x_to_return] + flat_evolved_embs_list, [base_mon] + flat_evolved_list


# Given a vec representation of a mon, y, draw an evolution type and subsequent data changes.
# Note that this function keeps some things constant that can normally change during evolution, for example
# abilities and egg groups.
def sample_bayesian_evo_vector(y, base_mon, existing_ev_vals, existing_ev_types,
                               type_list, ev_types_list, ev_items_list,
                               abilities_list, moves_list, tmhm_moves_list,
                               p_ev_type, p_ev_item_g_type, evo_lvl_slope, evo_lvl_intercept,
                               int_data_inc_mus_per_type, int_data_inc_stds_per_type,
                               p_levelup_g_type, p_tmhm_g_type):

    # Decide on an evolution type based on prior distribution.
    ev_type = None
    while (ev_type is None or  # Choose if we haven't
           ('EVO_LEVEL' in ev_type and ev_type in existing_ev_types) or  # Choose at most one kind of each evo_level_*
           ('EVO_FRIENDSHIP' in ev_type and ev_type in existing_ev_types)):  # Choose at most one kind of evo_friendship
        ev_type = np.random.choice(ev_types_list, p=p_ev_type)
    ev_type_logits = [1 if ev_types_list[idx] == ev_type else 0 for idx in range(len(ev_types_list))]

    # Pick the evolution item conditioned on base mon types.
    p_ev_item = [alpha * p_ev_item_g_type[type_list.index(base_mon['type1'])][idx] +
                 beta / 2. * p_ev_item_g_type[type_list.index(base_mon['type2'])][idx] +
                 beta / 2. * p_ev_item_g_type[type_list.index(base_mon['type3'])][idx] for idx in range(len(ev_items_list))]
    ev_item = None
    while ev_item is None or (ev_type == 'EVO_ITEM' and ev_item in existing_ev_vals):  # Choose unique evo item
        ev_item = np.random.choice(ev_items_list, p=p_ev_item)
    ev_item_logits = [1 if ev_items_list[idx] == ev_item else 0 for idx in range(len(ev_items_list))]

    # Create target vector for evolved form.
    y_evo = y[:]
    y_evo_type1 = base_mon['type1']
    y_evo_type2 = base_mon['type2']

    # Set types based on evolution style.
    # Set type2 to chosen evo_item if it forces a type change.
    if ev_type == 'EVO_ITEM' and ev_item in forced_type2:
        y_evo_type2 = forced_type2[ev_item]
        # Set type2 <- type3 if any other kind of evo and this is final stage.
    elif base_mon['evo_stages'] == 1:
        y_evo_type2 = base_mon['type3']
    y_evo_type3 = base_mon['type3'] if base_mon['type3'] not in [y_evo_type1, y_evo_type2] else 'TYPE_NONE'

    idx_base = len(int_data)
    for idx in range(len(type_list)):
        if type_list[idx] == y_evo_type1:
            y_evo[idx_base + idx] = 1
        elif type_list[idx] == y_evo_type2:
            y_evo[idx_base + idx] = 0.75
        elif type_list[idx] == y_evo_type3:
            y_evo[idx_base + idx] = 0.5
        else:
            y_evo[idx_base + idx] = 0
    # If primary type is redundant with secondary, promote NONE for decoding.
    if y_evo_type1 == y_evo_type2:
        y_evo[idx_base + type_list.index('TYPE_NONE')] = 1

    # Add to base stats conditioned on target evo type.
    z = np.random.normal(size=len(int_data))
    for idx in range(len(int_data)):
        y_evo[idx] += (z[idx] * (alpha * int_data_inc_stds_per_type[type_list.index(y_evo_type1)][idx] +
                                 beta * int_data_inc_stds_per_type[type_list.index(y_evo_type2)][idx]) +
                       (alpha * int_data_inc_mus_per_type[type_list.index(y_evo_type1)][idx] +
                        beta * int_data_inc_mus_per_type[type_list.index(y_evo_type2)][idx]))

    # Add to move confidence by weighting in the distribution of new types with additional jitter.
    levelup_p = [alpha * p_levelup_g_type[type_list.index(y_evo_type1)][idx] +
                 beta * p_levelup_g_type[type_list.index(y_evo_type2)][idx] for idx in range(len(moves_list))]
    uniform_rand = np.random.uniform(size=len(moves_list))
    uniform_rand /= sum(uniform_rand)
    levelup_p = [(1 - move_conf_jitter) * levelup_p[idx] + move_conf_jitter * uniform_rand[idx]
                 for idx in range(len(moves_list))]
    idx_base = len(int_data) + len(type_list) + 3 + len(abilities_list)
    for idx in range(len(moves_list)):
        y_evo[idx_base + idx] = 0.5 * y_evo[idx_base + idx] + 0.5 * levelup_p[idx]

    # Same for TMHM.
    tmhm_p = [alpha * p_tmhm_g_type[type_list.index(y_evo_type1)][idx] +
              beta * p_tmhm_g_type[type_list.index(y_evo_type2)][idx] for idx in range(len(tmhm_moves_list))]
    uniform_rand = np.random.uniform(size=len(tmhm_moves_list))
    uniform_rand /= sum(uniform_rand)
    tmhm_p = [(1 - move_conf_jitter) * tmhm_p[idx] + move_conf_jitter * uniform_rand[idx]
              for idx in range(len(tmhm_moves_list))]
    idx_base += len(moves_list)
    for idx in range(len(tmhm_moves_list)):
        y_evo[idx_base + idx] = 0.5 * y_evo[idx_base + idx] + 0.5 * tmhm_p[idx]

    # Pick the level to evolve at given new stat total.
    stat_total = sum([y_evo[int_data.index(s)] for s in stat_data])  # need raw logits in vec that are z normalized
    y_level_pred = evo_lvl_slope * stat_total + evo_lvl_intercept

    return y_evo, ev_type_logits, ev_item_logits, y_level_pred


# Infer type, move scores, stats, etc. just based on statistical distribution of original data.
def sample_bayesian_mon_vector(type_list, abilities_list, moves_list, tmhm_moves_list,
                               egg_group_list, growth_rates_list, gender_ratios_list,
                               p_type1, p_type2_g_type1,
                               int_data_mus_per_type, int_data_stds_per_type,
                               evo_lr,
                               p_ability_g_type, p_levelup_g_type, p_tmhm_g_type,
                               p_egg_g_type, growth_rate_c, p_gender):
    d = dict()

    # Draw types first, on which most things will be conditioned.
    # Type1 is drawn from the prior distribution of in-game types.
    # Type2 is drawn from the posterior distribution given type1.
    # Type3 is drawn given type1; type2/3 function together but 3 becomes secondary (official) in final levelup ev.
    d['type1'] = np.random.choice(type_list, p=p_type1)
    d['type2'] = np.random.choice(type_list, p=p_type2_g_type1[type_list.index(d['type1'])])
    d['type3'] = np.random.choice(type_list, p=p_type2_g_type1[type_list.index(d['type1'])])

    # Draw numeric stats.
    # For each type, we have a mean and stdev per stat.
    # We draw a z score from a normal distribution for each stat and assign it based on these type means.
    # For each stat s, we draw z_s and assign s as
    #   z_s*(alpha*type_1(s_std) + beta*type_2(s_std)) + (alpha*type_1(s_mu) + beta*type_2(s_mu))
    z = np.random.normal(size=len(int_data))
    type2 = d['type2'] if d['type2'] != 'TYPE_NONE' else d['type1']
    type3 = d['type3'] if d['type3'] != 'TYPE_NONE' else type2
    for idx in range(len(int_data)):
        d[int_data[idx]] = (z[idx] * (alpha * int_data_stds_per_type[type_list.index(d['type1'])][idx] +
                                      beta / 2. * int_data_stds_per_type[type_list.index(type2)][idx] +
                                      beta / 2. * int_data_stds_per_type[type_list.index(type3)][idx]) +
                            (alpha * int_data_mus_per_type[type_list.index(d['type1'])][idx] +
                             beta / 2. * int_data_mus_per_type[type_list.index(type2)][idx] +
                             beta / 2. * int_data_mus_per_type[type_list.index(type3)][idx]))

    # Determine number of evolutions based on stat total.
    stat_total = sum([d[s] for s in stat_data])
    # sub [0, 0.75] stddev from total bc classifier predicts low.
    n_evo = int(evo_lr.predict([[stat_total - len(stat_data) * np.random.rand() * 0.75]]))
    d['evo_stages'] = int(np.round(n_evo))

    # Determine abilities conditioned on types.
    ability_p = [alpha * p_ability_g_type[type_list.index(d['type1'])][idx] +
                 beta / 2. * p_ability_g_type[type_list.index(type2)][idx] +
                 beta / 2. * p_ability_g_type[type_list.index(type3)][idx] for idx in range(len(abilities_list))]
    abilities = list(np.random.choice(abilities_list, p=ability_p, replace=False, size=2))
    d['abilities'] = abilities

    # Determine levelup move confidences conditioned on types with a little jitter noise.
    levelup_p = [alpha * p_levelup_g_type[type_list.index(d['type1'])][idx] +
                 beta / 2. * p_levelup_g_type[type_list.index(type2)][idx] +
                 beta / 2. * p_levelup_g_type[type_list.index(type3)][idx] for idx in range(len(moves_list))]
    uniform_rand = np.random.uniform(size=len(moves_list))
    uniform_rand /= sum(uniform_rand)
    levelup_p = [(1 - move_conf_jitter) * levelup_p[idx] + move_conf_jitter * uniform_rand[idx]
                 for idx in range(len(moves_list))]
    d['levelup_confs'] = levelup_p

    # Determine TMHM move learning confidence conditioned on types with a little jitter noise.
    tmhm_p = [alpha * p_tmhm_g_type[type_list.index(d['type1'])][idx] +
              beta / 2. * p_tmhm_g_type[type_list.index(type2)][idx] +
              beta / 2. * p_tmhm_g_type[type_list.index(type3)][idx] for idx in range(len(tmhm_moves_list))]
    uniform_rand = np.random.uniform(size=len(tmhm_moves_list))
    uniform_rand /= sum(uniform_rand)
    tmhm_p = [(1 - move_conf_jitter) * tmhm_p[idx] + move_conf_jitter * uniform_rand[idx]
              for idx in range(len(tmhm_moves_list))]
    d['tmhm_confs'] = tmhm_p

    # Determine egg groups given types.
    egg_p = [alpha * p_egg_g_type[type_list.index(d['type1'])][idx] +
             beta / 2. * p_egg_g_type[type_list.index(type2)][idx] +
             beta / 2. * p_egg_g_type[type_list.index(type3)][idx] for idx in range(len(egg_group_list))]
    eggGroups = list(np.random.choice(egg_group_list, p=egg_p, replace=False, size=2))
    d['eggGroups'] = eggGroups

    # Trained classifier for predicting growth rates given stat totals.
    growthRate = growth_rate_c.predict([[stat_total, type_list.index(d['type1']), type_list.index(d['type2'])]])
    d['growthRate'] = growthRate

    # Gender ratio draw from prior distribution.
    d['genderRatio'] = np.random.choice(gender_ratios_list, p=p_gender)

    # Put together vector representing information from draw.
    y = []

    # Numeric stats.
    for idx in range(len(int_data)):
        y.append(d[int_data[idx]])

    # Types.
    for idx in range(len(type_list)):
        if type_list[idx] == d['type1']:
            y.append(1)
        elif type_list[idx] == d['type2']:
            y.append(0.75)
        elif type_list[idx] == d['type3']:
            y.append(0.5)
        else:
            y.append(0)

    # N evolutions.
    for idx in range(3):
        if d['evo_stages'] == idx:
            y.append(1)
        else:
            y.append(0)

    # Abilities.
    for idx in range(len(abilities_list)):
        if abilities_list[idx] in abilities:
            y.append(1)
        else:
            y.append(0)

    # Levelup moveset.
    for idx in range(len(moves_list)):
        y.append(d['levelup_confs'][idx])

    # TMHM moveset.
    for idx in range(len(tmhm_moves_list)):
        y.append(d['tmhm_confs'][idx])

    # Egg groups.
    for idx in range(len(egg_group_list)):
        if egg_group_list[idx] in eggGroups:
            y.append(1)
        else:
            y.append(0)

    # Growth rate.
    for idx in range(len(growth_rates_list)):
        if d['growthRate'] == growth_rates_list[idx]:
            y.append(1)
        else:
            y.append(0)

    # Gender ratio.
    for idx in range(len(gender_ratios_list)):
        if d['genderRatio'] == gender_ratios_list[idx]:
            y.append(1)
        else:
            y.append(0)

    return y


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
    learned_moves_counts = sorted([len(mon_levelup_moveset[m]) for m in mon_list])
    max_tmhm_moves = max([len(mon_tmhm_moveset[m]) for m in mon_list])
    tmhm_moves_counts = sorted([len(mon_tmhm_moveset[m]) for m in mon_list])
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

    p_type1 = p_type2_g_type1 = int_data_mus_per_type = int_data_stds_per_type = None
    evo_lr = None
    p_ability_g_type = p_levelup_g_type = p_tmhm_g_type = None
    p_egg_g_type = growth_rate_c = p_gender = None
    p_ev_type = p_ev_item_g_type = None
    evo_lvl_slope = evo_lvl_intercept = None
    int_data_inc_mus_per_type = int_data_inc_stds_per_type = None
    if args.draw_type == 'bayes':
        print("Getting distributions from main data needed for bayesian drawing...")
        # Type1 prior distribution.
        p_type1 = [sum([1 for m in mon_list if mon_metadata[m]['type1'] == t]) for t in type_list]
        p_type1 = [c / sum(p_type1) for c in p_type1]
        p_type1 = [(1 - uniform_weight) * p + uniform_weight * (1. / (len(type_list) - 1)) for p in p_type1]
        p_type1[type_list.index("TYPE_NONE")] = 0.  # can't assign type1 = TYPE_NONE during draw.

        # Type2 | Type1 conditional distribution.
        # Because we draw from neural metadata, TYPE_NONE as type2 just indicates type1=type2.
        p_type2_g_type1 = [[sum([1 for m in mon_list
                                if mon_metadata[m]['type1'] == t1 and mon_metadata[m]['type2'] == t2])
                            for t2 in type_list] for t1 in type_list]
        p_type2_g_type1 = [[c / sum(p_type2_g_type1[idx]) if sum(p_type2_g_type1[idx]) > 0
                            else 1. / len(type_list)
                            for c in p_type2_g_type1[idx]] for idx in range(len(type_list))]
        p_type2_g_type1 = [[(1 - uniform_weight) * p + uniform_weight * (1. / len(type_list))
                            for p in p_type2_g_type1[idx]] for idx in range(len(type_list))]

        # Means and stddevs of integer data conditioned on either type.
        int_data_mus_per_type = []
        int_data_stds_per_type = []
        for t in type_list:
            type_mus = []
            type_stds = []
            for idx in range(len(int_data)):
                d = [mon_metadata[m][int_data[idx]] for m in mon_list
                     if t == mon_metadata[m]['type1']]
                d.extend([mon_metadata[m][int_data[idx]] for m in mon_list
                          if t == mon_metadata[m]['type2'] or
                          (mon_metadata[m]['type2'] == "TYPE_NONE" and t == mon_metadata[m]['type1'])])
                type_mus.append(np.average(d))
                type_stds.append(np.std(d))
            int_data_mus_per_type.append(type_mus)
            int_data_stds_per_type.append(type_stds)

        # Linear regression between stat totals and number of evolutions for estimating based on stat draws.
        stat_totals = [sum([mon_metadata[m][s] for s in stat_data]) for m in mon_list]
        n_evos = [mon_metadata[m]['n_evolutions'] for m in mon_list]
        evo_lr = LogisticRegression(random_state=0).fit([[s] for s in stat_totals], n_evos)
        # Print confusion matrix.
        # cm = [[0 for i in set(n_evos)] for j in set(n_evos)]
        # for idx in range(len(mon_list)):
        #     cm[mon_metadata[mon_list[idx]]['n_evolutions']][int(evo_lr.predict([[stat_totals[idx]]]))] += 1
        # print(cm)

        # Ability frequency conditioned on type.
        p_ability_g_type = [[sum([2 if mon_metadata[m]['type2'] == 'TYPE_NONE' and mon_metadata[m]['type1'] == t else
                                  1 if t in [mon_metadata[m]['type1'], mon_metadata[m]['type2']] else 0
                                  for m in mon_list if a in mon_metadata[m]['abilities']])
                             for a in abilities_list] for t in type_list]
        p_ability_g_type = [[c / sum(p_ability_g_type[idx]) if sum(p_ability_g_type[idx]) > 0
                             else 1. / len(abilities_list)
                            for c in p_ability_g_type[idx]] for idx in range(len(type_list))]
        p_ability_g_type = [[(1 - uniform_weight) * p + uniform_weight * (1. / len(abilities_list))
                            for p in p_ability_g_type[idx]] for idx in range(len(type_list))]

        # Levelup move frequency conditioned on type.
        # This count ignores 'INFREQUENT', but the smoothing function that adds some uniform weight should mostly
        # take care of that without making this code horrendous.
        p_levelup_g_type = [[[2 if (mon_metadata[m]['type2'] == 'TYPE_NONE' and mon_metadata[m]['type1'] == t
                                   and l in [e[1] for e in mon_metadata[m]['levelup']]) else
                             1 if l in [e[1] for e in mon_metadata[m]['levelup']] else 0
                             for m in mon_list if t in [mon_metadata[m]['type1'], mon_metadata[m]['type2']]]
                             for l in moves_list] for t in type_list]
        p_levelup_g_type = [[sum(cl) / (2 * len(cl))  # prob per-move of mon learning conditioned on type
                             for cl in p_levelup_g_type[idx]] for idx in range(len(type_list))]
        p_levelup_g_type = [[(1 - uniform_weight) * p + uniform_weight  # add uniform weight to each move as prob
                             for p in p_levelup_g_type[idx]] for idx in range(len(type_list))]

        # TMHM frequency conditioned on type.
        p_tmhm_g_type = [[[2 if (mon_metadata[m]['type2'] == 'TYPE_NONE' and mon_metadata[m]['type1'] == t
                                 and l in mon_metadata[m]['tmhm']) else
                           1 if l in mon_metadata[m]['tmhm'] else 0
                           for m in mon_list if t in [mon_metadata[m]['type1'], mon_metadata[m]['type2']]]
                          for l in tmhm_moves_list] for t in type_list]
        p_tmhm_g_type = [[sum(cl) / (2 * len(cl))  # prob per-move of mon learning conditioned on type
                          for cl in p_tmhm_g_type[idx]] for idx in range(len(type_list))]
        p_tmhm_g_type = [[(1 - uniform_weight) * p + uniform_weight  # add uniform weight to each move as prob
                          for p in p_tmhm_g_type[idx]] for idx in range(len(type_list))]

        # Egg group conditioned on type.
        p_egg_g_type = [[sum([2 if mon_metadata[m]['type2'] == 'TYPE_NONE' and mon_metadata[m]['type1'] == t else
                              1 if t in [mon_metadata[m]['type1'], mon_metadata[m]['type2']] else 0
                              for m in mon_list if e in mon_metadata[m]['eggGroups']])
                         for e in egg_group_list] for t in type_list]
        p_egg_g_type = [[c / sum(p_egg_g_type[idx]) if sum(p_egg_g_type[idx]) > 0 else 1. / len(egg_group_list)
                         for c in p_egg_g_type[idx]] for idx in range(len(type_list))]
        p_egg_g_type = [[(1 - uniform_weight) * p + uniform_weight * (1. / len(egg_group_list))
                         for p in p_egg_g_type[idx]] for idx in range(len(type_list))]

        # Growth rate a learned classification from stat total.
        growth_rate_c = svm.SVC()
        growth_rates = [mon_metadata[m]['growthRate'] for m in mon_list]
        type1s = [type_list.index(mon_metadata[m]['type1']) for m in mon_list]
        type2s = [type_list.index(mon_metadata[m]['type2']) if mon_metadata[m]['type1'] != 'TYPE_NONE'
                  else type_list.index(mon_metadata[m]['type1']) for m in mon_list]
        growth_rate_c.fit([[stat_totals[idx], type1s[idx], type2s[idx]] for idx in range(len(mon_list))], growth_rates)
        # Print confusion matrix.
        # cm = [[0 for i in set(growth_rates_list)] for j in set(growth_rates_list)]
        # for idx in range(len(mon_list)):
        #     cm[growth_rates_list.index(mon_metadata[mon_list[idx]]['growthRate'])]\
        #         [growth_rates_list.index(growth_rate_c.predict([[stat_totals[idx], type1s[idx], type2s[idx]]]))] += 1
        # print(growth_rates_list)
        # print(cm)

        # Prior distribution on gender ratio.
        p_gender = [sum([1 for m in mon_list if mon_metadata[m]['genderRatio'] == r]) for r in gender_ratios_list]
        p_gender = [c / sum(p_gender) for c in p_gender]
        p_gender = [(1 - uniform_weight) * p + uniform_weight * (1. / len(gender_ratios_list)) for p in p_gender]

        # Prior distribution on evolution types.
        p_ev_type = [sum([[e[1] for e in mon_evolution[m]].count(evt) for m in mon_evolution])
                     for evt in ev_types_list]
        p_ev_type[ev_types_list.index('INFREQUENT')] = sum([len([e[1] for e in mon_evolution[m]
                                                                 if e[1] in infrequent_ev_types])
                                                            for m in mon_evolution])
        p_ev_type = [c / sum(p_ev_type) for c in p_ev_type]
        p_ev_type = [(1 - uniform_weight) * p + uniform_weight * (1. / len(ev_types_list)) for p in p_ev_type]

        # Posterior distribution of evolution items given mon type.
        # Here again we're going to let INFREQUENT items be handled by the uniform distribution weighting, especially
        # since it's just almost always zero.
        p_ev_item_g_type = [[sum([2 if mon_metadata[m]['type2'] == 'TYPE_NONE' and mon_metadata[m]['type1'] == t else
                                  1 if t in [mon_metadata[m]['type1'], mon_metadata[m]['type2']] else 0
                                  for m in mon_evolution
                                  if np.any([e[1] == 'EVO_ITEM' and e[2] == evi for e in mon_evolution[m]])])
                             for evi in ev_items_list] for t in type_list]
        p_ev_item_g_type = [[c / sum(p_ev_item_g_type[idx]) if sum(p_ev_item_g_type[idx]) > 0
                             else 1. / len(ev_items_list)
                             for c in p_ev_item_g_type[idx]] for idx in range(len(type_list))]
        p_ev_item_g_type = [[(1 - uniform_weight) * p + uniform_weight * (1. / len(ev_items_list))
                             for p in p_ev_item_g_type[idx]] for idx in range(len(type_list))]

        # Logistic regression classifier for predicting evolution level conditioned on stat total.
        evo_lvl = []
        for m in mon_evolution:
            evo_lvl.extend([[e[0], float(e[2]) / 100.] for e in mon_evolution[m] if e[1] == 'EVO_LEVEL'])
        evo_lvl_slope, evo_lvl_intercept, r_value, p_value, std_err = stats.linregress(
            [stat_totals[mon_list.index(evo_lvl[idx][0])] for idx in range(len(evo_lvl))],
            [el[1] for el in evo_lvl])
        # print(r_value, p_value, std_err)

        # Means and stddevs of integer data increases on either type when leveling up.
        int_data_inc_mus_per_type = []
        int_data_inc_stds_per_type = []
        for t in type_list:
            type_mus = []
            type_stds = []
            for idx in range(len(int_data)):
                d = []
                for m in mon_evolution:
                    for mon_evo, _, _ in mon_evolution[m]:
                        if mon_metadata[mon_evo]['type1'] == t:
                            d.append(mon_metadata[mon_evo][int_data[idx]] - mon_metadata[m][int_data[idx]])
                for m in mon_evolution:
                    for mon_evo, _, _ in mon_evolution[m]:
                        if (mon_metadata[mon_evo]['type2'] == t or (mon_metadata[mon_evo]['type2'] == 'TYPE_NONE' and
                                                                    mon_metadata[mon_evo]['type1'] == t)):
                            d.append(mon_metadata[mon_evo][int_data[idx]] - mon_metadata[m][int_data[idx]])
                type_mus.append(np.average(d))
                type_stds.append(np.std(d))
                int_data_inc_mus_per_type.append(type_mus)
            int_data_inc_stds_per_type.append(type_stds)

        print("... done")

    if args.draw_type == 'neural':
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
        evo_model = EvoNet(ae_hidden_dim, ae_hidden_dim,
                           n_evo_types, n_evo_items,
                           device).to(device)
        evo_model.load_state_dict(torch.load(args.input_evo_model_fn, map_location=torch.device(device)))
        evo_model.eval()
        print("... done")
    else:
        # Random projection matrix to create 'embeddings'
        ae_rand_projection = np.random.rand(ae_input_dim, ae_hidden_dim)
        embs_mu = np.matmul(ae_x, ae_rand_projection)
        embds_std = np.zeros_like(ae_x)

    print("Sampling...")
    new_mon = []
    new_mon_embs = np.zeros((len(mon_list), ae_hidden_dim))
    prop_types = {}
    while len(new_mon) < len(mon_list):

        if args.draw_type == 'neural':
            # Sample from a distribution based on the center of the real 'mon embeddings.
            z = torch.mean(embs_mu.cpu(), dim=0) + torch.std(embs_mu.cpu(), dim=0) * torch.randn(ae_hidden_dim)
            y = autoencoder_model.decode(z.unsqueeze(0).to(device))
            mon_vec = y[0].cpu().detach()
        else:
            # Sample a new mon based on a bayesian approach.
            mon_vec = sample_bayesian_mon_vector(type_list, abilities_list, moves_list, tmhm_moves_list,
                                                 egg_group_list, growth_rates_list, gender_ratios_list,
                                                 p_type1, p_type2_g_type1,
                                                 int_data_mus_per_type, int_data_stds_per_type,
                                                 evo_lr,
                                                 p_ability_g_type, p_levelup_g_type, p_tmhm_g_type,
                                                 p_egg_g_type, growth_rate_c, p_gender)
            z = np.matmul(mon_vec, ae_rand_projection)

        m = create_valid_mon(mon_vec, mon_metadata,
                             z_mean, z_std, int_data_ranges,
                             type_list, None,
                             abilities_list, mon_to_infrequent_abilities,
                             moves_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                             tmhm_moves_list,
                             egg_group_list,
                             growth_rates_list, None,
                             gender_ratios_list)
        global_vars['pop_idx'] += 1

        # Evolve.
        if not args.no_evolve:
            if args.draw_type == 'neural':
                evs_emb, evs = evolve(z.unsqueeze(0).to(device), evo_model, autoencoder_model, m, ae_rand_projection,
                                      ev_types_list, mon_to_infrequent_ev_types, ev_items_list,
                                      mon_to_infrequent_ev_items, 5, mon_metadata,
                                      z_mean, z_std, int_data_ranges,
                                      type_list, abilities_list, mon_to_infrequent_abilities,
                                      moves_list, max_learned_moves, lvl_move_avg, lvl_move_std,
                                      mon_to_infrequent_moves, tmhm_moves_list, max_tmhm_moves,
                                      egg_group_list, growth_rates_list, gender_ratios_list,
                                      p_ev_type, p_ev_item_g_type, evo_lvl_slope, evo_lvl_intercept,
                                      int_data_inc_mus_per_type, int_data_inc_stds_per_type,
                                      p_levelup_g_type, p_tmhm_g_type)
            else:
                # Bayes-based evolution procedure.
                evs_emb, evs = evolve(mon_vec, None, None, m, ae_rand_projection,
                                      ev_types_list, mon_to_infrequent_ev_types, ev_items_list,
                                      mon_to_infrequent_ev_items, 5, mon_metadata,
                                      z_mean, z_std, int_data_ranges,
                                      type_list, abilities_list, mon_to_infrequent_abilities,
                                      moves_list, max_learned_moves, lvl_move_avg, lvl_move_std,
                                      mon_to_infrequent_moves, tmhm_moves_list, max_tmhm_moves,
                                      egg_group_list, growth_rates_list, gender_ratios_list,
                                      p_ev_type, p_ev_item_g_type, evo_lvl_slope, evo_lvl_intercept,
                                      int_data_inc_mus_per_type, int_data_inc_stds_per_type,
                                      p_levelup_g_type, p_tmhm_g_type)
        else:
            evs_emb = [z]
            evs = [m]

        # Check whether adding this throws off type balance too much.
        type_balance = True
        for ev in evs:
            t = (ev['type1'], ev['type2'])
            # The biggest proportion in the real distribution is NORMAL/NORMAL at just over 12%.
            if t in prop_types and prop_types[t] / float(len(new_mon)) > 0.12:
                type_balance = False
                break

        # Add to population.
        if type_balance and len(new_mon) + len(evs) <= len(mon_list):
            for idx in range(len(evs_emb)):
                new_mon_embs[len(new_mon) + idx, :] = evs_emb[idx]
            new_mon.extend(evs)
            for ev in evs:
                t = (ev['type1'], ev['type2'])
                if t not in prop_types:
                    prop_types[t] = 0
                prop_types[t] += 1
        else:
            global_vars['pop_idx'] -= len(evs)
    print("... done; sampled %d mon" % len(mon_list))

    print("Assigning movesets...")
    lvl_confs = [sum([conf for _, _, conf in m['levelup_moveset']])
                 for m in new_mon]
    lvl_n_moves = [0] * len(new_mon)
    for idx in range(len(mon_list)):
        lvl_n_moves[np.argsort(lvl_confs)[idx]] = learned_moves_counts[idx]
    tmhm_confs = [sum([conf for _, conf in m['mon_tmhm_moveset']])
                  for m in new_mon]
    tmhm_n_moves = [0] * len(new_mon)
    for idx in range(len(mon_list)):
        tmhm_n_moves[np.argsort(tmhm_confs)[idx]] = tmhm_moves_counts[idx]

    n_to_show = 0  # DEBUG
    for jdx in range(len(new_mon)):  # Limit levelup and tmhm moves by new thresholds.
        m = new_mon[jdx]
        m['levelup_moveset'].sort(key=lambda x: x[2], reverse=True)  # Sort by confidence
        m['levelup_moveset'] = [[m['levelup_moveset'][idx][0], m['levelup_moveset'][idx][1]]
                                for idx in range(len(m['levelup_moveset']))][:lvl_n_moves[jdx]]
        base_tmhm_moveset = []
        if 'evo_from' in m:
            # Move not learned by base version is learned by this version at maximum of evo level
            base = get_mon_by_species_id(new_mon, m['evo_from'])
            base_tmhm_moveset = base['mon_tmhm_moveset']  # Just a list by now.
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
        m['mon_tmhm_moveset'].sort(key=lambda x: x[1], reverse=True)  # Sort by confidence
        tmhm_moveset = base_tmhm_moveset[:]
        tmhm_moveset.extend([m['mon_tmhm_moveset'][idx][0] for idx in range(len(m['mon_tmhm_moveset']))
                                      if m['mon_tmhm_moveset'][idx][0] not in base_tmhm_moveset])
        m['mon_tmhm_moveset'] = tmhm_moveset[:tmhm_n_moves[jdx]]

        # DEBUG
        # if 'evolution' in m:
        #     if m['evo_stages'] == 2:
        #         n_to_show += m['evo_stages']
        #         n_to_show += len(m['evolution']) + 1
        # if n_to_show > 0:
        #     n_to_show -= 1
        #     print(m)
        #     _ = input()  # DEBUG
    print("... done")

    # Visualize sampled mon embeddings based on their NNs.
    # print("Drawing TSNE visualization of new sampled mon embeddings...")
    # tsne = TSNE(random_state=1, n_iter=1000, metric="euclidean",
    #             learning_rate=1000)  # changed from default bc of outliers
    # mon_icons = [mon_metadata[new_mon[idx]['nn']]["icon"] for idx in range(len(new_mon))]
    # if ae_hidden_dim > 2:
    #     embs_2d = tsne.fit_transform(new_mon_embs)
    # elif ae_hidden_dim == 2:
    #     embs_2d = new_mon_embs
    # else:  # ae_hidden_dim == 1
    #     embs_2d = np.zeros(shape=(len(new_mon_embs), 2))
    #     embs_2d[:, 0] = new_mon_embs[:, 0]
    #     embs_2d[:, 1] = new_mon_embs[:, 0]
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(embs_2d[:, 0], embs_2d[:, 1], alpha=.1)
    # paths = mon_icons
    # for x0, y0, path in zip(embs_2d[:, 0], embs_2d[:, 1], paths):
    #     img = plt.imread(path)
    #     ab = AnnotationBbox(OffsetImage(img[:32, :32, :]),
    #                         (x0, y0), frameon=False)
    #     ax.add_artist(ab)
    # plt.savefig("%s.sampled_mon_embeddings.pdf" % args.output_fn,
    #             bbox_inches='tight')
    # print("... done")

    # Load the sprite network into memory.
    sprite_model = SpriteNet(ae_hidden_dim, device).to(device)
    if args.input_sprite_model_fn:
        sprite_model.load_state_dict(torch.load(args.input_sprite_model_fn, map_location=torch.device(device)))
    else:
        print("WARNING: not creating fresh sprites; will load NN sprites over normal ones")

    # Get input sprites for each new mon.
    in_front = np.zeros((len(new_mon), 3, 64, 64))
    in_back = np.zeros((len(new_mon), 3, 64, 64))
    in_icon = np.zeros((len(new_mon), 3, 32, 32))
    front_static = []
    back_static = []
    icon_static = []
    for idx in range(len(mon_list)):
        dists = [np.linalg.norm(new_mon_embs[idx, :] - embs_mu[jdx, :].cpu().numpy())
                 for jdx in range(len(mon_list))]
        nn_emb_idx = np.argsort(dists)[0]

        mon_fn_name = mon_list[nn_emb_idx][len('SPECIES_'):].lower()
        if args.input_sprite_model_fn:
            front = transforms.ToTensor()(
                Image.open('orig/graphics/%s.front.png' % mon_fn_name).convert("RGB").crop((0, 0, 64, 64)))
            in_front[idx, :, :, :] = front
            back = transforms.ToTensor()(
                Image.open('orig/graphics/%s.back.png' % mon_fn_name).convert("RGB").crop((0, 0, 64, 64)))
            in_back[idx, :, :, :] = back
            icon = transforms.ToTensor()(
                Image.open('orig/graphics/%s.icon.png' % mon_fn_name).convert("RGB").crop((0, 0, 32, 32)))
            in_icon[idx, :, :, :] = icon
        else:
            front_static.append('orig/graphics/%s.front.png' % mon_fn_name)
            back_static.append('orig/graphics/%s.back.png' % mon_fn_name)
            icon_static.append('orig/graphics/%s.icon.png' % mon_fn_name)
    in_front = torch.tensor(in_front).float().to(device)
    in_back = torch.tensor(in_back).float().to(device)
    in_icon = torch.tensor(in_icon).float().to(device)

    # Load type->names
    type_to_name_words = None
    if args.name_by_type_word:
        with open(args.name_by_type_word, 'r') as f:
            type_to_name_words = json.load(f)

    # Replace species ids with species names for the purpose of swapping/ROM rewriting.
    print("Replacing global ids with species ids for ROM map, generating and writing sprites, and then writing data to file...")
    out_fns = []
    for idx in range(len(new_mon)):
        if type_to_name_words is not None:
            candidates1 = type_to_name_words[new_mon[idx]['type1']]
            candidates2 = type_to_name_words[new_mon[idx]['type2']]
            m = [c for c in candidates1 if c in candidates2]
            if len(m) > 0:  # a word matches both types, so pick it
                new_mon[idx]['name'] = m[0]
            else:  # no word matches both types, so assign by primary
                new_mon[idx]['name'] = candidates1[0]

            for t in type_to_name_words:
                if new_mon[idx]['name'] in type_to_name_words[t]:
                    del type_to_name_words[t][type_to_name_words[t].index(new_mon[idx]['name'])]

            new_mon[idx]['name'] = new_mon[idx]['name'].upper()
            if len(new_mon[idx]['name']) <= 7:
                new_mon[idx]['name'] += 'MON'
            new_mon[idx]['name'] = new_mon[idx]['name'][:10]
        else:
            new_mon[idx]['name'] = "MON %d" % new_mon[idx]['species']
        new_mon[idx]['species'] = mon_list[new_mon[idx]['species']]
        if 'evolution' in new_mon[idx]:
            for jdx in range(len(new_mon[idx]['evolution'])):
                new_mon[idx]['evolution'][jdx][0] = mon_list[new_mon[idx]['evolution'][jdx][0]]

        # Sprites.
        species_dir = new_mon[idx]['species'][len('SPECIES_'):].lower()
        if species_dir == 'castform':
            suffixes = ['_normal_form', '_rainy_form', '_sunny_form', '_snowy_form']
        else:
            suffixes = None
        if species_dir == 'unown':
            subdirs = list('abcdefghijklmnopqrstuvwxyz') + ['question_mark'] + ['exclamation_mark']
        else:
            subdirs = None
        if not os.path.isdir('../graphics/pokemon/%s/' % species_dir):
            print("WARNING: missing graphics/pokemon dir '%s'" % species_dir)
        if args.input_sprite_model_fn:
            front = sprite_model(torch.tensor(new_mon_embs[idx]).unsqueeze(0).float().to(device), in_front[idx, :])
            back = sprite_model(torch.tensor(new_mon_embs[idx]).unsqueeze(0).float().to(device), in_back[idx, :])
            icon = sprite_model(torch.tensor(new_mon_embs[idx]).unsqueeze(0).float().to(device), in_icon[idx, :])

            imf = transforms.ToPILImage()(front.squeeze(0).detach().cpu())
            imb = transforms.ToPILImage()(back.squeeze(0).detach().cpu())
            im = transforms.ToPILImage()(icon.squeeze(0).detach().cpu())
            im = im.resize((32, 32))
            icon_target = Image.new('RGB', (32, 64))
            icon_target.paste(im, (0, 0, 32, 32))
            icon_target.paste(im, (0, 32, 32, 64))
            front_anim_target = Image.new('RGB', (64, 128))
            front_anim_target.paste(imf, (0, 0, 64, 64))
            front_anim_target.paste(imf, (0, 64, 64, 128))
            if suffixes is not None:
                for suf in suffixes:
                    imf.save('../graphics/pokemon/%s/front%s.png' % (species_dir, suf), mode='RGB')
                    imb.save('../graphics/pokemon/%s/back%s.png' % (species_dir, suf), mode='RGB')
                    front_anim_target.save('../graphics/pokemon/%s/anim_front%s.png' % (species_dir, suf), mode='RGB')
                    out_fns.append('../graphics/pokemon/%s/front%s.png' % (species_dir, suf))
                    out_fns.append('../graphics/pokemon/%s/back%s.png' % (species_dir, suf))
                    out_fns.append('../graphics/pokemon/%s/anim_front%s.png' % (species_dir, suf))
                icon_target.save('../graphics/pokemon/%s/icon.png' % species_dir, mode='RGB')
                out_fns.append('../graphics/pokemon/%s/icon.png' % species_dir)
            elif subdirs is not None:
                for subdir in subdirs:
                    imf.save('../graphics/pokemon/%s/%s/front.png' % (species_dir, subdir), mode='RGB')
                    out_fns.append('../graphics/pokemon/%s/%s/front.png' % (species_dir, subdir))
                    front_anim_target.save('../graphics/pokemon/%s/%s/anim_front.png' % (species_dir, subdir), mode='RGB')
                    out_fns.append('../graphics/pokemon/%s/%s/anim_front.png' % (species_dir, subdir))
                    imb.save('../graphics/pokemon/%s/%s/back.png' % (species_dir, subdir), mode='RGB')
                    out_fns.append('../graphics/pokemon/%s/%s/back.png' % (species_dir, subdir))
                    icon_target.save('../graphics/pokemon/%s/%s/icon.png' % (species_dir, subdir), mode='RGB')
                    out_fns.append('../graphics/pokemon/%s/%s/icon.png' % (species_dir, subdir))
            else:
                imf.save('../graphics/pokemon/%s/front.png' % species_dir, mode='RGB')
                out_fns.append('../graphics/pokemon/%s/front.png' % species_dir)
                front_anim_target.save('../graphics/pokemon/%s/anim_front.png' % species_dir, mode='RGB')
                out_fns.append('../graphics/pokemon/%s/anim_front.png' % species_dir)
                imb.save('../graphics/pokemon/%s/back.png' % species_dir, mode='RGB')
                out_fns.append('../graphics/pokemon/%s/back.png' % species_dir)
                icon_target.save('../graphics/pokemon/%s/icon.png' % species_dir, mode='RGB')
                out_fns.append('../graphics/pokemon/%s/icon.png' % species_dir)
        else:
            if suffixes is not None:
                for suf in suffixes:
                    os.system('cp %s ../graphics/pokemon/%s/front%s.png' % (front_static[idx], species_dir, suf))
                    os.system('cp %s ../graphics/pokemon/%s/anim_front%s.png' % (front_static[idx], species_dir, suf))
                    os.system('cp %s ../graphics/pokemon/%s/back%s.png' % (back_static[idx], species_dir, suf))
                os.system('cp %s ../graphics/pokemon/%s/icon.png' % (icon_static[idx], species_dir))
            elif subdirs is not None:
                for subdir in subdirs:
                    os.system('cp %s ../graphics/pokemon/%s/%s/front.png' % (front_static[idx], species_dir, subdir))
                    os.system('cp %s ../graphics/pokemon/%s/%s/anim_front.png' % (front_static[idx], species_dir, subdir))
                    os.system('cp %s ../graphics/pokemon/%s/%s/back.png' % (back_static[idx], species_dir, subdir))
                    os.system('cp %s ../graphics/pokemon/%s/%s/icon.png' % (icon_static[idx], species_dir, subdir))
            else:
                os.system('cp %s ../graphics/pokemon/%s/front.png' % (front_static[idx], species_dir))
                os.system('cp %s ../graphics/pokemon/%s/anim_front.png' % (front_static[idx], species_dir))
                os.system('cp %s ../graphics/pokemon/%s/back.png' % (back_static[idx], species_dir))
                os.system('cp %s ../graphics/pokemon/%s/icon.png' % (icon_static[idx], species_dir))
    print("... done")

    if len(out_fns) > 0:
        print("Running pngtopnm; will produce lots of output...")
    for out_fn in out_fns:
        print(out_fn)
        os.system('pngtopnm %s | pnmquant 16 | pnmtopng > tmp.png' % out_fn)
        os.system('mv tmp.png %s' % out_fn)
    if len(out_fns) > 0:
        print("... done")

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
    parser.add_argument('--draw_type', type=str, required=True,
                        help='one of "neural" or "bayes"')
    parser.add_argument('--input_meta_mon_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--input_meta_network_fn', type=str, required=True,
                        help='the input file for the network metadata')
    parser.add_argument('--input_mon_model_fn', type=str, required=False,
                        help='the trained mon autoencoder weights')
    parser.add_argument('--input_evo_model_fn', type=str, required=False,
                        help='the trained mon evolver weights')
    parser.add_argument('--input_sprite_model_fn', type=str, required=False,
                        help='the trained sprite creation network')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output json for the sampled mon')
    parser.add_argument('--no_evolve', action='store_true',
                        help='disable evolutions during sampling')
    parser.add_argument('--name_by_type_word', type=str, required=False,
                        help='json mapping types to potential names')
    args = parser.parse_args()

    assert args.draw_type == 'bayes' or args.input_mon_model_fn is not None
    assert args.draw_type == 'bayes' or args.input_evo_model_fn is not None

    main(args)
