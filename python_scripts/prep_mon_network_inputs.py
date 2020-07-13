import argparse
import json
import numpy as np
import pandas as pd
import torch

# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship']  # don't actually care about predicting 'safariZoneFleeRate'


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
            if np.any([b in mon_evolution for b, _, _ in mon_evolution[a]]):
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
    egg_group_list = [eg for eg in eg_mon]
    print("infrequent abilities", infrequent_abilities)
    print("infrequent learned moves", infrequent_moves)
    for a in mon_metadata:
        for idx in range(len(mon_metadata[a]["levelup"])):
            level, move = mon_metadata[a]["levelup"][idx]
            if move in infrequent_moves:
                mon_metadata[a]["levelup"][idx] = [level, "INFREQUENT"]
        for idx in range(len(mon_metadata[a]["abilities"])):
            if mon_metadata[a]["abilities"][idx] in infrequent_abilities:
                mon_metadata[a]["abilities"][idx] = "INFREQUENT"

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

    # Hidden dimension.
    h = 128

    # Write model metadata.
    print("Writing model metadata to file...")
    with open(args.output_fn, 'w') as f:
        d = {"mon_list": mon_list,
             "mon_metadata": mon_metadata,
             "mon_evolution": mon_evolution,
             "z_mean": z_mean,
             "z_std": z_std,
             "input_dim": input_dim,
             "hidden_dim": h,
             "int_data": int_data,
             "type_list": type_list,
             "abilities_list": abilities_list,
             "infrequent_abilities_map": infrequent_abilities,
             "moves_list": moves_list,
             "infrequent_moves_map": infrequent_moves,
             "tmhm_moves_list": tmhm_moves_list,
             "moves_pos_weight": moves_pos_weight.detach().numpy().tolist(),
             "tmhm_pos_weight": tmhm_pos_weight.detach().numpy().tolist(),
             "name_len": name_len,
             "name_chars": name_chars,
             "egg_group_list": egg_group_list,
             "growth_rates_list": growth_rates_list,
             "gender_ratios_list": gender_ratios_list,
             "x_input": x.detach().numpy().tolist()
             }
        json.dump(d, f)
    print("... done; wrote data to %s" % args.output_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input file for the mon metadata')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the network metadata')
    args = parser.parse_args()


    main(args)
