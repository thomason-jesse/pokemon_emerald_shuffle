import argparse
import json
import numpy as np
import random
import os

# Paths.
base_stats_orig_fn = 'orig/base_stats.h'
evolution_fn = '../src/data/pokemon/evolution.h'
trainer_parties_fn = 'orig/trainer_parties.h'
wild_encounters_fn = 'orig/wild_encounters.h'


# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship']
stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']
ev_data = ['evYield_HP', 'evYield_Attack', 'evYield_Defense',
           'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense']

# Bayes hyperparameters.
alpha = 0.60  # weight that type1 distributions contribute.
beta = 0.40  # weight that type2 distributions contribute.
uniform_weight = 0.001  # how much uniform distribution to mix in to smooth out real data.


def create_valid_mon(y, mon_metadata, type1, type2, type3,
                     z_mean, z_std, int_data_ranges,
                     type_list, preserve_primary_type,
                     abilities_list, mon_to_infrequent_abilities,
                     move_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                     tmhm_move_list,
                     egg_groups_list,
                     growth_rates_list, preserve_growth_rate,
                     gender_ratios_list,
                     levelup_moveset, mon_tmhm_moveset):  # TODO: directly set move learnsets elsewhere
    idx = 0
    d = {}

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
    idx += len(type_list)
    d['type1'] = type1
    d['type2'] = type2 if type2 != 'TYPE_NONE' else type1
    d['type3'] = type3 if type3 != 'TYPE_NONE' else type2

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


    # movesets
    d['levelup_moveset'] = levelup_moveset
    d['mon_tmhm_moveset'] = mon_tmhm_moveset

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


def sample_new_mon_given_type_map(abilities_list, moves_list, tmhm_moves_list,
                                  egg_group_list, growth_rates_list, gender_ratios_list, type_list,
                                  p_ability_g_type, p_levelup_g_type, p_tmhm_g_type, p_levelup_depth_prior,
                                  p_egg_g_type, p_gender,
                                  mon_metadata, mon_evolution, mon_evolution_sr, type_map, mon_n_appearances,
                                  z_mean, z_std, int_data_ranges, mon_to_infrequent_abilities,
                                  lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                                  levelup_move_levels, learnset_move_tech_tree, tmhm_move_tech_tree):
    mon_map = {}
    new_mon = {}
    mon_to_map = list(mon_metadata.keys())
    while len(mon_to_map) > 0:
        a = mon_to_map.pop(0)
        print(a)  # DEBUG
        # If not at the base of an evolution chain, skip.
        if a in mon_evolution_sr:
            mon_to_map.append(a)
            continue

        # If a is part of an evolution chain, go up to the top.
        q = [[a]]
        current_level_to_index = [a]
        next_level_to_index = []
        n_target_moves = 0
        n_tmhm_moves = 0
        while len(current_level_to_index) > 0 or len(next_level_to_index) > 0:
            if len(current_level_to_index) > 0:
                a = current_level_to_index.pop()
                n_target_moves = max(len(mon_metadata[a]['levelup']), n_target_moves)
                n_tmhm_moves = max(len(mon_metadata[a]['tmhm']), n_tmhm_moves)
                q_level = []
                if a in mon_evolution:
                    for a_up in mon_evolution[a]:
                        q_level.append(a_up[0])
                        del mon_to_map[mon_to_map.index(a_up[0])]
                    q.append(q_level)
                    next_level_to_index.extend(q_level[:])
            else:
                current_level_to_index = next_level_to_index[:]
                next_level_to_index = []

        upper_level_types = set()
        upper_level_abilities = set()
        upper_level_stats_original = []  # one entry for each upper level ev.
        upper_level_stats_sampled = []  # one entry for each upper level ev.
        upper_level_learnsets_sampled = []  # one entry for each upper level ev.
        upper_level_tmhm_sampled = []  # one entry for each upper level ev.
        while len(q) > 0:  # pop the top level of the evo chart and build up
            top_species_list = q.pop()
            print(top_species_list)  # DEBUG
            current_level_types = set()
            current_level_abilities = set()
            new_upper_level_stats_sampled = []
            new_upper_level_stats_original = []
            new_upper_level_learnsets_sampled = []
            new_upper_level_tmhm_sampled = []
            for a in top_species_list:

                print(a)  # DEBUG
                species = mon_metadata[a]['species']
                type1 = type_map[mon_metadata[a]['type1']]
                type2 = 'TYPE_NONE' if mon_metadata[a]['type2'] == 'TYPE_NONE' else type_map[mon_metadata[a]['type2']]
                possible_type3 = upper_level_types - {type1, type2}
                if len(possible_type3) > 0:
                    type3 = np.random.choice(list(possible_type3))
                else:
                    type3 = 'TYPE_NONE'
                growthRate = mon_metadata[a]['growthRate']
                print(species, type1, type2, type3, growthRate)
                current_level_types.add(type1)
                if type2 != 'TYPE_NONE':
                    current_level_types.add(type2)

                # Get original stats.
                int_data_base = {}
                stats_values = []
                for idx in range(len(int_data)):
                    if int_data[idx] in stat_data:
                        stats_values.append(mon_metadata[a][int_data[idx]])
                    else:
                        int_data_base[int_data[idx]] = mon_metadata[a][int_data[idx]]
                stats_original = stats_values[:]

                # Sample new stats.
                sampled_stats = []
                if len(upper_level_stats_original) == 0:
                    # We are currently at the uppermost evolution level, so these stats are "original"
                    mu = np.average(stats_values)
                    stddev = np.std(stats_values)
                    for idx in range(len(int_data)):
                        if int_data[idx] in stat_data:
                            int_data_base[int_data[idx]] = mu + np.random.normal() * stddev
                            sampled_stats.append(int_data_base[int_data[idx]])
                    print(species, 'high level sampled', sampled_stats)  # DEBUG

                    # We are currently at the uppermost evolution level, so set the learnsets and tmhm set.
                    type2_l = type1 if type2 == 'TYPE_NONE' else type2
                    type3_l = type2_l if type3 == 'TYPE_NONE' else type3
                    learn_p = [(alpha * p_levelup_g_type[type_list.index(type1)][idx] +
                                beta / 2. * p_levelup_g_type[type_list.index(type2_l)][idx] +
                                beta / 2. * p_levelup_g_type[type_list.index(type3_l)][idx]) *
                               p_levelup_depth_prior[idx] for idx in range(len(moves_list))]
                    learn_p = [learn_p[idx] / sum(learn_p) for idx in range(len(learn_p))]
                    learnset = {}
                    while n_target_moves - len(learnset) > 0:
                        # Sample a top level move m to build a learnset from.
                        learn_p_sampler = [learn_p[idx] if moves_list[idx] not in learnset else 0
                                           for idx in range(len(learn_p))]
                        learn_p_sampler = [learn_p_sampler[idx] / sum(learn_p_sampler)
                                           for idx in range(len(learn_p))]
                        p_sum = 0
                        for idx in np.argsort(learn_p_sampler)[::-1]:
                            if p_sum > 0.33:  # nucleus sampling
                                learn_p_sampler[idx] = 0
                            p_sum += learn_p_sampler[idx]
                        learn_p_sampler = [learn_p_sampler[idx] / sum(learn_p_sampler)
                                           for idx in range(len(learn_p_sampler))]
                        m = np.random.choice(moves_list, p=learn_p_sampler)

                        # Expand learnset
                        mu = np.average(levelup_move_levels[m])
                        std = np.std(levelup_move_levels[m])
                        learnset[m] = max(0, min(100, int(mu + np.random.normal() * std + 0.5)))
                        learned_zero = learnset[m] == 0
                        last_move = m
                        while not learned_zero:
                            potential_prereqs = set([learnset_move_tech_tree[last_move][jdx][0]
                                                     for jdx in range(len(learnset_move_tech_tree[last_move]))])
                            if len(potential_prereqs.intersection(set(learnset.keys()))) > 0:  # already have prereq
                                break
                            g_type_probs = [
                                alpha * p_levelup_g_type[type_list.index(type1)][moves_list.index(b)] +
                                beta / 2. * p_levelup_g_type[type_list.index(type2_l)][moves_list.index(b)] +
                                beta / 2. * p_levelup_g_type[type_list.index(type3_l)][moves_list.index(b)]
                                for b, prob, mu, std in learnset_move_tech_tree[last_move]]
                            g_type_probs = [g_type_probs[idx] / sum(g_type_probs) for idx in range(len(g_type_probs))]
                            probs = [g_type_probs[idx] * learnset_move_tech_tree[last_move][idx][1]
                                     for idx in range(len(learnset_move_tech_tree[last_move]))]
                            probs = [probs[idx] / sum(probs) for idx in range(len(probs))]
                            p_sum = 0
                            for idx in np.argsort(probs)[::-1]:
                                if p_sum > .67:  # nucleus sampling
                                    probs[idx] = 0
                                p_sum += probs[idx]
                            probs = [probs[idx] / sum(probs) for idx in range(len(probs))]
                            jidx = np.random.choice(len(probs), p=probs)
                            next_move, _, next_mu, next_std = learnset_move_tech_tree[last_move][jidx]
                            learnset[next_move] = max(0,
                                                      min(learnset[last_move],
                                                          learnset[last_move] - int(
                                                              next_mu + np.random.normal() * next_std + 0.5)))
                            last_move = next_move
                            learned_zero = learnset[next_move] == 0
                    levelup_moveset = [[learnset[m], m] for m in learnset]
                    # Sort by level learn order.
                    levelup_moveset.sort(key=lambda x: x[0])
                    levelup_moveset[0][0] = 0  # always learn move0 at level 0
                    print(levelup_moveset)  # DEBUG

                    # TMHM moveset.
                    tmhm_moves = set()
                    tmhm_p = [(alpha * p_tmhm_g_type[type_list.index(type1)][idx] +
                                beta / 2. * p_tmhm_g_type[type_list.index(type2_l)][idx] +
                                beta / 2. * p_tmhm_g_type[type_list.index(type3_l)][idx])
                              for idx in range(len(tmhm_moves_list))]
                    tmhm_p = [tmhm_p[idx] / sum(tmhm_p) for idx in range(len(tmhm_p))]
                    score_per_b = {}
                    for b in tmhm_move_tech_tree:
                        score_by_p = 0
                        for m in learnset:
                            for prereq, prob, _, _ in tmhm_move_tech_tree[b]:
                                if prereq == m:
                                    score_by_p += prob
                        score_per_b[b] = score_by_p * tmhm_p[tmhm_moves_list.index(b)]
                    score_probs_b = {b: score_per_b[b] / sum(score_per_b.values())
                                     if sum(score_per_b.values()) > 0 else 1. / len(score_per_b)
                                     for b in score_per_b}
                    for (k, v) in [(k, v) for k, v in sorted(score_probs_b.items(),
                                                             key=lambda item: item[1], reverse=True)][:n_tmhm_moves]:
                        tmhm_moves.add(k)
                    mon_tmhm_moveset = list(tmhm_moves)
                    print(tmhm_moves)  # DEBUG

                else:
                    # We are at a lower evolution level, so we will set stats based on the difference between
                    # the original upper and current stat levels.
                    diffs = [[upper_level_stats_original[idx][jdx] - stats_original[jdx]
                              for idx in range(len(upper_level_stats_original))]
                             for jdx in range(len(stats_original))]
                    mus = [np.average([upper_level_stats_sampled[jdx][idx]
                                       for jdx in range(len(upper_level_stats_sampled))])
                           for idx in range(len(stat_data))]
                    mu = np.average(mus)
                    stddev = np.std(mus)
                    print(species, 'lower level diffs', diffs, mu, stddev)  # DEBUG
                    for idx in range(len(int_data)):
                        if int_data[idx] in stat_data:
                            int_data_base[int_data[idx]] = \
                                np.average([upper_level_stats_sampled[jdx][idx]
                                            for jdx in range(len(upper_level_stats_sampled))]) - \
                                max(0, mu + np.random.normal() * stddev)
                            sampled_stats.append(int_data_base[int_data[idx]])
                    print(species, 'lower level sampled', sampled_stats)  # DEBUG

                    # For levelup, start with intersection of higher-level, then pepper in lower-level learning until
                    # we hit the target, then reduce the level learned by like 4/5 or something.
                    learnset = {upper_level_learnsets_sampled[0][idx][1]: upper_level_learnsets_sampled[0][idx][0]
                                for idx in range(len(upper_level_learnsets_sampled[0]))}
                    for idx in range(1, len(upper_level_learnsets_sampled)):
                        identified = set()
                        local_ms = {}
                        for lvl, m in upper_level_learnsets_sampled[idx]:
                            identified.add(m)
                            local_ms[m] = lvl
                        learnset = {m: min(learnset[m], local_ms[m])
                                    for m in learnset if m in identified}
                    exausted = False
                    while len(learnset) < n_target_moves and not exausted:  # pick up low level until we fill docket
                        for lvl in range(100):
                            for idx in range(len(upper_level_learnsets_sampled)):
                                for _lvl, m in upper_level_learnsets_sampled[idx]:
                                    if lvl == _lvl and m not in learnset:
                                        learnset[m] = _lvl
                            if len(learnset) >= n_target_moves:
                                break
                        exausted = True
                    levelup_moveset = [[int(learnset[m] * 4./5 + 0.5), m] for m in learnset]
                    levelup_moveset.sort(key=lambda x: x[0])
                    levelup_moveset[0][0] = 0  # always learn move0 at level 0
                    print('lower level learnset', levelup_moveset)  # DEBUG

                    n_target_moves = len(mon_metadata[a]['tmhm'])
                    tmhm_moves = {upper_level_tmhm_sampled[0][idx] for idx in range(len(upper_level_tmhm_sampled[0]))}
                    for idx in range(1, len(upper_level_tmhm_sampled)):
                        identified = set()
                        for m in upper_level_tmhm_sampled[idx]:
                            identified.add(m)
                        tmhm_moves = {m for m in tmhm_moves if m in identified}
                    n_exausted = 100
                    while len(tmhm_moves) < n_target_moves and n_exausted > 0:  # pick up low level moves until
                        idx = np.random.randint(0, len(upper_level_tmhm_sampled))
                        jdx = np.random.randint(0, len(upper_level_tmhm_sampled[idx]))
                        if upper_level_tmhm_sampled[idx][jdx] not in tmhm_moves:
                            tmhm_moves.add(upper_level_tmhm_sampled[idx][jdx])
                        n_exausted -= 1
                    levelup_moveset = [[int(learnset[m] * 4. / 5 + 0.5), m] for m in learnset]
                    levelup_moveset.sort(key=lambda x: x[0])
                    mon_tmhm_moveset = list(tmhm_moves)
                    print('lower level tmhm', tmhm_moves)  # DEBUG

                # Store for lower level generations.
                new_upper_level_stats_original.append(stats_original)
                new_upper_level_stats_sampled.append(sampled_stats)
                new_upper_level_learnsets_sampled.append(levelup_moveset)
                new_upper_level_tmhm_sampled.append(mon_tmhm_moveset)

                b_vec = sample_bayesian_mon_vector(
                    type_list, abilities_list, species, type1, type2, type3, growthRate, int_data_base, len(q),
                    moves_list, tmhm_moves_list, egg_group_list, growth_rates_list, gender_ratios_list, p_ability_g_type,
                    p_levelup_g_type, p_tmhm_g_type, p_egg_g_type, p_gender)
                mon_b = create_valid_mon(
                    b_vec, mon_metadata, type1, type2, type3,
                    z_mean, z_std, int_data_ranges, type_list, None, abilities_list,
                    mon_to_infrequent_abilities, moves_list, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
                    tmhm_moves_list, egg_group_list, growth_rates_list, None, gender_ratios_list,
                    levelup_moveset, mon_tmhm_moveset)
                mon_map[a] = a
                mon_b['species'] = mon_metadata[a]['species']

                if len(upper_level_abilities) > 0:  # this is not the upper level so replace abilities with sample
                    print(upper_level_abilities, list(upper_level_abilities))  # DEBUG
                    mon_b['abilities'] = list(upper_level_abilities)[np.random.randint(0, len(upper_level_abilities))]
                current_level_abilities.add(tuple(mon_b['abilities']))
                print('finished sampled mon abilities', mon_b['abilities'])  # DEBUG

                # Set evo level and forms based on original
                mon_b['evolution'] = []
                if a in mon_evolution:
                    for ev_mon, ev_type, ev_val in mon_evolution[a]:
                        if ev_type == 'EVO_ITEM' and ev_val not in ['ITEM_DEEP_SEA_TOOTH', 'ITEM_DEEP_SEA_SCALE']:
                            ev_mon_types = {new_mon[ev_mon]['type1'], new_mon[ev_mon]['type2']}
                            if type1 not in ev_mon_types:  # base type change, so adopt ev type
                                type_changed = new_mon[ev_mon]['type1']
                            else:
                                type_changed = type3 if type3 in ev_mon_types else \
                                    (type2 if type2 in ev_mon_types else type1)
                            if type_changed == 'TYPE_NONE':
                                type_changed = type1
                            if type_changed == 'TYPE_FLYING' or type_changed == 'TYPE_PSYCHIC':
                                ev_val = 'ITEM_SUN_STONE'
                            elif type_changed == 'TYPE_NORMAL' or type_changed == 'TYPE_GROUND':
                                ev_val = 'ITEM_KINGS_ROCK'
                            elif type_changed == 'TYPE_FIRE' or type_changed == 'TYPE_FIGHTING':
                                ev_val = 'ITEM_FIRE_STONE'
                            elif type_changed == 'TYPE_GRASS' or type_changed == 'TYPE_BUG':
                                ev_val = 'ITEM_LEAF_STONE'
                            elif type_changed == 'TYPE_POISON' or type_changed == 'TYPE_GHOST' \
                                    or type_changed == 'TYPE_DARK':
                                ev_val = 'ITEM_MOON_STONE'
                            elif type_changed == 'TYPE_ICE' or type_changed == 'TYPE_WATER':
                                ev_val = 'ITEM_WATER_STONE'
                            elif type_changed == 'TYPE_STEEL' or type_changed == 'TYPE_ROCK':
                                ev_val = 'ITEM_METAL_COAT'
                            elif type_changed == 'TYPE_ELECTRIC' or type_changed == 'TYPE_DRAGON':
                                ev_val = 'ITEM_THUNDER_STONE'
                            else:
                                print('WARNING: no evo_item set!', type_changed)  # DEBUG
                                _ = input()  # DEBUG
                        if ev_val.isdigit():
                            mon_b['evolution'].append([ev_mon, ev_type, int(ev_val)])
                        else:
                            mon_b['evolution'].append([ev_mon, ev_type, ev_val])

                new_mon[mon_b['species']] = mon_b

            upper_level_types = current_level_types
            upper_level_abilities = current_level_abilities
            upper_level_stats_sampled = new_upper_level_stats_sampled
            upper_level_stats_original = new_upper_level_stats_original
            upper_level_learnsets_sampled = new_upper_level_learnsets_sampled
            upper_level_tmhm_sampled = new_upper_level_tmhm_sampled

    return mon_map, new_mon  # trivial a->a mapping; need to write out a's new data


# Infer type, move scores, stats, etc. just based on statistical distribution of original data.
def sample_bayesian_mon_vector(type_list, abilities_list,
                               species, type1, type2, type3, growthRate,
                               int_data_base, evo_stages,
                               moves_list, tmhm_moves_list,
                               egg_group_list, growth_rates_list, gender_ratios_list,
                               p_ability_g_type, p_levelup_g_type, p_tmhm_g_type,
                               p_egg_g_type, p_gender):
    d = dict()
    d['evo_stages'] = evo_stages
    d['type1'] = type1
    if type2 == 'TYPE_NONE':
        type2 = type1
    d['type2'] = type2
    if type3 == 'TYPE_NONE':
        type3 = type2
    d['type3'] = type3
    d['species'] = species

    # Set numeric stats.
    for idx in range(len(int_data)):
        d[int_data[idx]] = int_data_base[int_data[idx]]

    # Determine abilities conditioned on types.
    ability_p = [alpha * p_ability_g_type[type_list.index(type1)][idx] +
                 beta / 2. * p_ability_g_type[type_list.index(type2)][idx] +
                 beta / 2. * p_ability_g_type[type_list.index(type3)][idx] for idx in range(len(abilities_list))]
    abilities = list(np.random.choice(abilities_list, p=ability_p, replace=False, size=2))
    d['abilities'] = abilities

    # TODO: use levelup_p, tech tree, and avg # moves to decide binary vector moveset + use it upstairs for lower tiers.
    levelup_p = [alpha * p_levelup_g_type[type_list.index(d['type1'])][idx] +
                 beta / 2. * p_levelup_g_type[type_list.index(type2)][idx] +
                 beta / 2. * p_levelup_g_type[type_list.index(type3)][idx] for idx in range(len(moves_list))]
    uniform_rand = np.random.uniform(size=len(moves_list))
    uniform_rand /= sum(uniform_rand)
    d['levelup_confs'] = levelup_p

    # TODO: use tmhm_p, tech tree, and avg # moves to decide binary vector moveset. (needs TMHM tech tree stored.)
    tmhm_p = [alpha * p_tmhm_g_type[type_list.index(d['type1'])][idx] +
              beta / 2. * p_tmhm_g_type[type_list.index(type2)][idx] +
              beta / 2. * p_tmhm_g_type[type_list.index(type3)][idx] for idx in range(len(tmhm_moves_list))]
    uniform_rand = np.random.uniform(size=len(tmhm_moves_list))
    uniform_rand /= sum(uniform_rand)
    d['tmhm_confs'] = tmhm_p

    # Determine egg groups given types.
    egg_p = [alpha * p_egg_g_type[type_list.index(d['type1'])][idx] +
             beta / 2. * p_egg_g_type[type_list.index(type2)][idx] +
             beta / 2. * p_egg_g_type[type_list.index(type3)][idx] for idx in range(len(egg_group_list))]
    eggGroups = list(np.random.choice(egg_group_list, p=egg_p, replace=False, size=2))
    d['eggGroups'] = eggGroups

    # Keep growth rate of base mon.
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
    type_map = {}
    assert args.type_mapping in ['random', 'fixed', 'optimized']
    if args.partial_type_map is not None:
        tms = args.partial_type_map.strip().split(',')
        for tm in tms:
            ps = tm.split('-')
            type_map[ps[0]] = ps[1]

    print("Reading processed mon model metadata from '%s'..." % args.input_meta_network_fn)
    with open(args.input_meta_network_fn, 'r') as f:
        d = json.load(f)
        mon_metadata = d["mon_metadata"]
        z_mean = d["z_mean"]
        z_std = d["z_std"]
        mon_evolution = d["mon_evolution"]
        infrequent_moves_map = d["infrequent_moves_map"]
        abilities_list = d["abilities_list"]
        moves_list = d["moves_list"]
        tmhm_moves_list = d["tmhm_moves_list"]
        egg_group_list = d["egg_group_list"]
        growth_rates_list = d["growth_rates_list"]
        gender_ratios_list = d["gender_ratios_list"]
        infrequent_abilities = d["infrequent_abilities_map"]

        # Add infrequent moves back into moves_list.
        moves_list.extend(list(infrequent_moves_map.keys()))
        del moves_list[moves_list.index('INFREQUENT')]
    print("... done")

    print("Reading in mon metadata and evolutions; then z-scoring and reversing structures.")
    with open(args.input_fn_source, 'r') as f:
        d = json.load(f)
        mon_metadata_source = d["mon_metadata"]
        type_list = d["type_list"]
        mon_evolution_source = d["mon_evolution"]
        mon_evolution_source = {m: [entry[0] for entry in mon_evolution_source[m]]
                                for m in mon_evolution_source}
        mon_levelup_moveset = d["mon_levelup_moveset"]
        mon_tmhm_moveset = d["mon_tmhm_moveset"]
    # Perform z score normalization on numeric metadata.
    mon_list = list(mon_metadata_source.keys())
    n_mon = len(mon_list)

    print("Reading in mon tech tree info.")
    with open(args.input_move_tech_tree, 'r') as f:
        d = json.load(f)
        depths_per_move = d["depths_per_move"]
        levelup_move_levels = d["levelup_move_levels"]
        learnset_move_tech_tree = d["learnset_move_tech_tree"]
        tmhm_move_tech_tree = d["tmhm_move_tech_tree"]
    # Perform z score normalization on numeric metadata.
    mon_list = list(mon_metadata_source.keys())
    n_mon = len(mon_list)

    # Create reversed evolutions list.
    mon_evolution_sr = {}
    for a in mon_evolution_source:
        for b in mon_evolution_source[a]:
            if b in mon_evolution_sr:
                print("WARNING: multiple base species for %d" % b)
            mon_evolution_sr[b] = a
    print("... done")

    # Read in wild and trainer files to determine mon importance.
    print("Counting up number of appearances in-game...")
    mon_n_appearances = [0] * n_mon
    for fn, weight in [[trainer_parties_fn, 10], [wild_encounters_fn, 1]]:
        with open(fn, 'r') as f:
            d = f.read()
            for idx in range(n_mon):
                mon_n_appearances[idx] += d.count(mon_list[idx]) * weight
    print("... done; counted %d instances from %d to %d" %
          (sum(mon_n_appearances), min(mon_n_appearances), max(mon_n_appearances)))

    print("Noting ranges of various data values and getting needed averages for sampling...")
    int_data_ranges = [[int(min([mon_metadata[m][int_data[didx]] * z_std[didx] + z_mean[didx] for m in mon_list])),
                        int(max([mon_metadata[m][int_data[didx]] * z_std[didx] + z_mean[didx] for m in mon_list]))]
                       for didx in range(len(int_data))]
    lvl_move_entries = {}
    for a in mon_levelup_moveset:
        for level, move in mon_levelup_moveset[a]:
            if move not in lvl_move_entries:
                lvl_move_entries[move] = []
            lvl_move_entries[move].append(level)
    lvl_move_avg = [np.average(lvl_move_entries[move]) for move in moves_list]
    lvl_move_std = [np.std(lvl_move_entries[move]) for move in moves_list]
    mon_to_infrequent_moves = {infrequent_moves_map[m][0]: m for m in infrequent_moves_map}
    mon_to_infrequent_abilities = {infrequent_abilities[m][0]: m for m in infrequent_abilities}
    print("... done")

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

    # get levelup prob based on tech tree depth
    p_levelup_depth_prior = [np.power(np.e, np.average(depths_per_move[l])) for l in moves_list
                             if l in learnset_move_tech_tree]
    p_levelup_depth_prior /= sum(p_levelup_depth_prior)

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

    # Prior distribution on gender ratio.
    p_gender = [sum([1 for m in mon_list if mon_metadata[m]['genderRatio'] == r]) for r in gender_ratios_list]
    p_gender = [c / sum(p_gender) for c in p_gender]
    p_gender = [(1 - uniform_weight) * p + uniform_weight * (1. / len(gender_ratios_list)) for p in p_gender]

    print("... done")

    # First, assign a random type -> type map.
    type_list_to_assign = [t for t in type_list if t not in type_map]
    type_list_to_image = [t for t in type_list if not np.any([t == type_map[u]
                                                              for u in type_map])]
    idxs = list(range(len(type_list_to_assign)))
    if args.type_mapping == 'random':
        while np.any([type_list_to_assign[idx] == type_list_to_image[idxs[idx]]
                      for idx in range(len(type_list_to_assign))]):
            random.shuffle(idxs)
        print("Random initial type map:")
    elif args.type_mapping == 'fixed':
        print("Fixed initial type map:")
    for idx in range(len(type_list_to_assign)):
        type_map[type_list_to_assign[idx]] = type_list_to_image[idxs[idx]]
    for t in type_map:
        print("\t%s->%s" % (t, type_map[t]))

    mon_map, new_mon = sample_new_mon_given_type_map(
        abilities_list, moves_list, tmhm_moves_list, egg_group_list, growth_rates_list, gender_ratios_list, type_list,
        p_ability_g_type, p_levelup_g_type, p_tmhm_g_type, p_levelup_depth_prior, p_egg_g_type, p_gender,
        mon_metadata, mon_evolution, mon_evolution_sr, type_map, mon_n_appearances, z_mean, z_std, int_data_ranges,
        mon_to_infrequent_abilities, lvl_move_avg, lvl_move_std, mon_to_infrequent_moves,
        levelup_move_levels, learnset_move_tech_tree, tmhm_move_tech_tree)

    front_static = []
    back_static = []
    icon_static = []
    normal_color_static = []
    shiny_color_static = []
    new_mon_keys = list(new_mon.keys())
    for idx in range(len(new_mon_keys)):
        # Sprite comes from nearest stat neighbor.
        dists = [np.linalg.norm([(new_mon[new_mon_keys[idx]][stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]
                                 - mon_metadata[mon_list[jdx]][stat_data[sidx]]
                                 for sidx in range(len(stat_data))], ord=1) for jdx in range(len(mon_list))]
        nn_emb_idx = np.argsort(dists)[0]
        mon_fn_name = mon_metadata[mon_list[nn_emb_idx]]['species'][len('SPECIES_'):].lower()
        front_static.append('orig/graphics/%s.front.png' % mon_fn_name)
        back_static.append('orig/graphics/%s.back.png' % mon_fn_name)
        icon_static.append('orig/graphics/%s.icon.png' % mon_fn_name)
        print(new_mon[new_mon_keys[idx]], 'sprite', mon_fn_name)  # DEBUG
        # Colors come from nearest stat neighbor with a matching type.
        dists = [np.power(np.linalg.norm([(new_mon[new_mon_keys[idx]][stat_data[sidx]] - z_mean[sidx]) / z_std[sidx]
                                          - mon_metadata[mon_list[jdx]][stat_data[sidx]]
                                          for sidx in range(len(stat_data))], ord=1), 1. /
                 (len({new_mon[new_mon_keys[idx]]['type1'], new_mon[new_mon_keys[idx]]['type2']}.intersection(
                     {mon_metadata[mon_list[jdx]]['type1'], mon_metadata[mon_list[jdx]]['type2']})) + 1))
                 for jdx in range(len(new_mon_keys))]
        nn_emb_idx = np.argsort(dists)[0]
        mon_fn_name = mon_metadata[mon_list[nn_emb_idx]]['species'][len('SPECIES_'):].lower()
        normal_color_static.append('orig/graphics/%s.normal.pal' % mon_fn_name)
        shiny_color_static.append('orig/graphics/%s.shiny.pal' % mon_fn_name)
        print('color', mon_fn_name)  # DEBUG

    # Load type->names
    type_to_name_words = None
    if args.name_by_type_word:
        with open(args.name_by_type_word, 'r') as f:
            type_to_name_words = json.load(f)

    # Replace species ids with species names for the purpose of swapping/ROM rewriting.
    for idx in range(len(new_mon_keys)):
        meta_to_edit_mon = new_mon[new_mon_keys[idx]]
        candidates1 = type_to_name_words[meta_to_edit_mon['type1']]
        candidates2 = type_to_name_words[meta_to_edit_mon['type2']]
        m = [c for c in candidates1 if c in candidates2]
        if len(m) > 0:  # a word matches both types, so pick it
            meta_to_edit_mon['name'] = m[0]
        else:  # no word matches both types, so assign by primary
            meta_to_edit_mon['name'] = candidates1[0]
        for t in type_to_name_words:
            if meta_to_edit_mon['name'] in type_to_name_words[t]:
                del type_to_name_words[t][type_to_name_words[t].index(meta_to_edit_mon['name'])]
        meta_to_edit_mon['name'] = meta_to_edit_mon['name'].upper()
        if len(meta_to_edit_mon['name']) <= 7:
            meta_to_edit_mon['name'] += 'MON'
        meta_to_edit_mon['name'] = meta_to_edit_mon['name'][:10]

        # Sprites.
        species_dir = meta_to_edit_mon['species'][len('SPECIES_'):].lower()
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

        if suffixes is not None:
            for suf in suffixes:
                # os.system('cp ../graphics/pokemon/%s/normal%s.pal orig/graphics/%s.normal.pal' %
                #           (species_dir, suf, species_dir))
                # os.system('cp ../graphics/pokemon/%s/shiny%s.pal orig/graphics/%s.shiny.pal' %
                #           (species_dir, suf, species_dir))

                os.system('cp %s ../graphics/pokemon/%s/front%s.png' % (front_static[idx], species_dir, suf))
                os.system('cp %s ../graphics/pokemon/%s/anim_front%s.png' % (front_static[idx], species_dir, suf))
                os.system('cp %s ../graphics/pokemon/%s/back%s.png' % (back_static[idx], species_dir, suf))
                os.system('cp %s ../graphics/pokemon/%s/normal%s.pal' % (normal_color_static[idx], species_dir, suf))
                os.system('cp %s ../graphics/pokemon/%s/shiny%s.pal' % (shiny_color_static[idx], species_dir, suf))
            os.system('cp %s ../graphics/pokemon/%s/icon.png' % (icon_static[idx], species_dir))
        elif subdirs is not None:
            for subdir in subdirs:
                # os.system('cp ../graphics/pokemon/%s/normal.pal orig/graphics/%s.normal.pal' %
                #           (species_dir, species_dir))
                # os.system('cp ../graphics/pokemon/%s/shiny.pal orig/graphics/%s.shiny.pal' %
                #           (species_dir, species_dir))

                os.system('cp %s ../graphics/pokemon/%s/%s/front.png' % (front_static[idx], species_dir, subdir))
                os.system(
                    'cp %s ../graphics/pokemon/%s/%s/anim_front.png' % (front_static[idx], species_dir, subdir))
                os.system('cp %s ../graphics/pokemon/%s/%s/back.png' % (back_static[idx], species_dir, subdir))
                os.system('cp %s ../graphics/pokemon/%s/%s/icon.png' % (icon_static[idx], species_dir, subdir))
                os.system('cp %s ../graphics/pokemon/%s/normal.pal' %
                          (normal_color_static[idx], species_dir))  # unown pal files at root
                os.system('cp %s ../graphics/pokemon/%s/shiny.pal' % (shiny_color_static[idx], species_dir,))
        else:
            # os.system('cp ../graphics/pokemon/%s/normal.pal orig/graphics/%s.normal.pal' % (species_dir, species_dir))
            # os.system('cp ../graphics/pokemon/%s/shiny.pal orig/graphics/%s.shiny.pal' % (species_dir, species_dir))

            os.system('cp %s ../graphics/pokemon/%s/front.png' % (front_static[idx], species_dir))
            os.system('cp %s ../graphics/pokemon/%s/anim_front.png' % (front_static[idx], species_dir))
            os.system('cp %s ../graphics/pokemon/%s/back.png' % (back_static[idx], species_dir))
            os.system('cp %s ../graphics/pokemon/%s/icon.png' % (icon_static[idx], species_dir))
            os.system('cp %s ../graphics/pokemon/%s/normal.pal' % (normal_color_static[idx], species_dir))
            os.system('cp %s ../graphics/pokemon/%s/shiny.pal' % (shiny_color_static[idx], species_dir))
    print("... done")

    # Write the resulting map.
    print("Writing results to JSON format...")
    d = {"mon_map": mon_map, "type_map": type_map}
    with open(args.output_fn_mon_map, 'w') as f:
        json.dump(d, f, indent=2)
    print("... done")

    # Output sampled mon structure into .json expected by the swapping script.
    d = {'mon_metadata': {}, 'mon_evolution': {}, 'mon_levelup_moveset': {}, 'mon_tmhm_moveset': {}}
    data_to_ignore = ['levelup', 'name_chars', 'tmhm', 'evo_stages', 'levelup_moveset', 'evolution', 'n_evolutions']
    for m in mon_map.values():
        m = new_mon[m]
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
    with open(args.output_fn_mon_data, 'w') as f:
        json.dump(d, f, indent=2)
    print("... done; wrote to '%s'" % args.output_fn_mon_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_fn_source', type=str, required=True,
                        help='the input file for the original ROM mon metadata')
    parser.add_argument('--input_meta_network_fn', type=str, required=True,
                        help='the input file for the network metadata')
    parser.add_argument('--input_move_tech_tree', type=str, required=True,
                        help='the input file for the precomputed move tech tree data')
    parser.add_argument('--name_by_type_word', type=str, required=True,
                        help='name to type word map for new names')
    parser.add_argument('--output_fn_mon_map', type=str, required=True,
                        help='the output file for the mon map')
    parser.add_argument('--output_fn_mon_data', type=str, required=True,
                        help='the output file for the mon data')
    parser.add_argument('--type_mapping', type=str, required=True,
                        help='"random", "fixed", "optimized", or a partial list with - and ,')
    parser.add_argument('--partial_type_map', type=str, required=False,
                        help='a (partial) list with - and ,')
    args = parser.parse_args()

    main(args)
