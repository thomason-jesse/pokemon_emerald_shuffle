import argparse
import json
import math
import numpy as np
import os
import random
import sys


# Paths.
battle_moves_fn = '../src/data/battle_moves.h'
level_up_learnsets_fn = '../src/data/pokemon/level_up_learnsets.h'
tmhm_learnsets_fn = '../src/data/pokemon/tmhm_learnsets.h'


# Metadata consts.
int_data = ['power', 'accuracy', 'pp', 'secondaryEffectChance', 'priority']

# Distance coefficients.
# How many standard deviations to mark a move away from another when one of these differs.
effect_coeff = 1
type_coeff = 2
target_coeff = 1
flag_coeff = 1  # IOU * this coeff

# Real valued properties have been z-score normalized, so a diff of 1 corresponds to a std deviation.
# So, we can use 1 as a distance for other effect differences to say, roughly, that the move
# falls another standard deviation away on a discrete metric (e.g., effect type).
def get_move_distance(a, b, type_map):

    # Does the effect differ.
    effect_diff = effect_coeff if a["effect"] != b["effect"] else 0

    # Sum of numerical differences.
    numerical_diff = sum([abs(a[data_type] - b[data_type])
                          for data_type in int_data])

    # Does the type differ.
    if a["type"] in type_map:
        type_diff = type_coeff if type_map[a["type"]] != b["type"] else 0
    else:  # For things like "TYPE_MYSTERY"
        type_diff = type_coeff if a["type"] != b["type"] else 0

    # Does the target differ.
    target_diff = target_coeff if a["target"] != b["target"] else 0

    # IOU of the flags.
    if len(a["flags"]) > 0 or len(b["flags"]) > 0:
        flag_diff = 1 - (len(set(a["flags"]).intersection(set(b["flags"]))) / float(len(set(a["flags"]).union(set(b["flags"])))))
    else:
        flag_diff = 0
    flag_diff = flag_diff * flag_coeff

    return effect_diff + numerical_diff + type_diff + flag_diff


def main(args):

    # Read in type map.
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_map = d["mon_map"]
        mon_map = {str(a): str(mon_map[a]) for a in mon_map}
        type_map = d["type_map"]
        type_map = {str(t): str(type_map[t]) for t in type_map}

    # Read in move metadata.
    move_metadata = {}
    with open(battle_moves_fn, 'r') as f:
        curr_move = None
        for line in f.readlines():
            if "[MOVE_" in line:
                #    [MOVE_POUND] =\n
                ps = line.split("MOVE_")
                move = ps[1][:ps[1].index(']')]
                if move == "NONE":
                    continue
                ps_close = ps[1][ps[1].index(']'):]
                if ps[0] == "    [" and ps_close == "] =\n":
                    move = "MOVE_%s" % move
                    move_metadata[move] = {}
                    curr_move = move
            elif curr_move is not None:
                if "=" in line:
                    #        .power = 40,\n
                    ps = line.strip().split("=")
                    data_name = ps[0].strip().strip('.')
                    data_str = ps[1].strip().strip(',')
                    if data_name in int_data:
                        move_metadata[curr_move][data_name] = int(data_str)
                    elif data_name == "flags":
                        move_metadata[curr_move][data_name] = data_str.split(" | ") if data_str != "0" else []
                    else:
                        move_metadata[curr_move][data_name] = data_str
    move_list = list(move_metadata.keys())
    n_moves = len(move_list)
    print("Read in %d move metadata" % len(move_metadata))

    # Normalize numerical metadata.
    print("Calculating distance between moves...")
    for data_type in int_data:
        data_mean = np.mean([move_metadata[a][data_type] for a in move_list])
        data_std = np.std([move_metadata[a][data_type] for a in move_list])
        for a in move_list:
            move_metadata[a][data_type] = (move_metadata[a][data_type] - data_mean) / data_std

    # Calculate the distance between each move.
    dist = np.zeros(shape=(n_moves, n_moves))
    for idx in range(n_moves):
        for jdx in range(n_moves):
            dist[idx, jdx] = get_move_distance(move_metadata[move_list[idx]],
                                               move_metadata[move_list[jdx]],
                                               type_map)

    # Writer nearest neighbor lists.
    move_nns = {}
    for idx in range(n_moves):
        jdxs = np.argsort(dist[idx])  #get min jdxs
        move_nns[move_list[idx]] = [move_list[jdx] for jdx in jdxs]
    print("done... got %dx%d matrix of move distances" % (n_moves, n_moves))

    # Read in level up and TMHM learnsets.
    mon_moveset = {}  # mon -> list of moves
    n_levelup_moves = 0
    with open(level_up_learnsets_fn, 'r') as f:
        curr_mon = None
        for line in f.readlines():
            if "LevelUpLearnset[] = {" in line:
                #static const u16 sBulbasaurLevelUpLearnset[] = {
                ps = line.split(" u16 ")
                mon = ps[1][1:ps[1].index('[')].replace("LevelUpLearnset", "")
                mon = "SPECIES_%s" % mon.upper()
                if mon not in mon_moveset:
                    mon_moveset[mon] = set()
                curr_mon = mon
            elif curr_mon is not None:
                if "LEVEL_UP_MOVE" in line:
                    #    LEVEL_UP_MOVE( 1, MOVE_TACKLE),\n
                    ps = line.strip().split(",")
                    move = ps[1].strip().strip(',()')
                    mon_moveset[curr_mon].add(move)
                    n_levelup_moves += 1
    print("Read in %d level up moves" % n_levelup_moves)
    n_tmhm_moves = 0
    with open(tmhm_learnsets_fn, 'r') as f:
        curr_mon = None
        for line in f.readlines():
            if "[SPECIES_" in line:
                #    [SPECIES_BULBASAUR]   = TMHM_LEARNSET(TMHM(TM06_TOXIC)\n
                ps = line.split("[SPECIES_")
                mon = ps[1][:ps[1].index(']')]
                if mon == "NONE":
                    continue
                mon = "SPECIES_%s" % mon
                if mon not in mon_moveset:
                    mon_moveset[mon] = set()
                curr_mon = mon
                # move on this line
                if line.count('TMHM') > 1:
                    tmhm = line.split('TMHM')[2].strip().strip('()')
                    move = '_'.join(tmhm.split('_')[1:]).strip('(),')
                    mon_moveset[curr_mon].add("MOVE_%s" % move)
                    n_tmhm_moves += 1
            elif curr_mon is not None:
                if "TMHM" in line:
                    #                                        | TMHM(TM09_BULLET_SEED)\n
                    tmhm = line.split('TMHM')[1].strip().strip('()')
                    move = '_'.join(tmhm.split('_')[1:]).strip('(),')
                    mon_moveset[curr_mon].add("MOVE_%s" % move)
                    n_tmhm_moves += 1
    print("Read in %d TM/HM moves" % n_tmhm_moves)

    # For every move a 'mon can learn or be taught, select the best replacement.
    print("Creating map per mon from all moves to nearest neighbor under this distance...")
    mon_move_replace = {}  # indexed by mon, then move, value ->move.
    for mon in mon_map:
        mon_move_replace[mon] = {}
        for move in move_list:
            # Map this move to the nearest neighbor move that this mon can learn.
            for nn in move_nns[move]:
                if nn in mon_moveset[mon]:
                    mon_move_replace[mon][move] = nn
                    break
    print("... done")

    # Write to file.
    print("Writing to file...")
    with open(args.output_fn, 'w') as f:
        d = {"mon_map": mon_map, "type_map": type_map, "mon_move_map": mon_move_replace}
        json.dump(d, f, indent=2)
    print("... done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adds all moves' nearest neighbors to assignment file.")
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input JSON of mon and type maps')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output JSON for the move ranks')
    args = parser.parse_args()

    main(args)
