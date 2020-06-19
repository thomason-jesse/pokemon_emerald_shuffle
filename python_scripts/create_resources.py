import argparse
import copy
import json
import numpy as np


# Paths.
base_stats_orig_fn = 'orig/base_stats.h'
evolution_fn = 'orig/evolution.h'
level_up_learnsets_fn = 'orig/level_up_learnsets.h'
tmhm_learnsets_fn = 'orig/tmhm_learnsets.h'


# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship', 'safariZoneFleeRate']
stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']


def main(args):

    # Read in mon metadata.
    mon_metadata = {}
    type_list = []
    with open(base_stats_orig_fn, 'r') as f:
        curr_mon = None
        for line in f.readlines():
            if "SPECIES_" in line:
                #     [SPECIES_BULBASAUR] =\n
                ps = line.split("SPECIES_")
                mon = ps[1][:ps[1].index(']')]
                ps_close = ps[1][ps[1].index(']'):]
                if ps[0] == "    [" and ps_close == "] =\n":
                    species_mon = "SPECIES_%s" % mon
                    mon_metadata[species_mon] = {"species": species_mon,
                                                 "icon": "../graphics/pokemon/%s/icon.png" % mon.lower()}
                    # Handle weird sprites.
                    if species_mon == "SPECIES_UNOWN":
                        mon_metadata[species_mon]["icon"] = "../graphics/pokemon/unown/a/icon.png"
                    curr_mon = species_mon
            elif curr_mon is not None:
                if "=" in line:
                    #         .baseHP        = 45,\n
                    ps = line.strip().split("=")
                    data_name = ps[0].strip().strip('.')
                    data_str = ps[1].strip().strip(',')
                    if data_name in int_data:
                        mon_metadata[curr_mon][data_name] = int(data_str)
                    elif data_name == "abilities":
                        mon_metadata[curr_mon][data_name] = [ab.strip()
                                                             for ab in data_str.strip('{}').split(',')]
                    else:
                        mon_metadata[curr_mon][data_name] = data_str
                    if data_name == "type1" or data_name == "type2":
                        if data_str not in type_list:
                            type_list.append(data_str)
    print("Read in %d mon metadata and %d types" % (len(mon_metadata), len(type_list)))

    # Read in mon evolution chart.
    mon_evolution = {}
    with open(evolution_fn, 'r') as f:
        curr_mon = None
        for line in f.readlines()[2:-1]:
            if "=" in line:
                #     [SPECIES_BULBASAUR]  = {{EVO_LEVEL, 16, SPECIES_IVYSAUR}},\n
                ps = line.split("=")
                species_a = ps[0].strip().strip('[]')
                species_b = ps[1].split(',')[2].strip().strip('{}')
                curr_mon = species_a
                if species_a not in mon_evolution:
                    mon_evolution[species_a] = []
                mon_evolution[species_a].append(species_b)
            else:
                #                             {EVO_ITEM, ITEM_SUN_STONE, SPECIES_BELLOSSOM}},
                species_b = line.strip().split(',')[2].strip().strip('{}')
                mon_evolution[curr_mon].append(species_b)
    print("Read in %d mon evolutions" % len(mon_evolution))

    # Read in level up and TMHM learnsets.
    mon_levelup_moveset = {}  # lists of [level_learned, move_name]
    n_levelup_moves = 0
    with open(level_up_learnsets_fn, 'r') as f:
        curr_mon = None
        for line in f.readlines():
            if "LevelUpLearnset[] = {" in line:
                # static const u16 sBulbasaurLevelUpLearnset[] = {
                ps = line.split(" u16 ")
                mon = ps[1][1:ps[1].index('[')].replace("LevelUpLearnset", "")
                mon = "SPECIES_%s" % mon.upper()
                if mon == "SPECIES_NIDORANF":
                    mon = "SPECIES_NIDORAN_F"
                elif mon == "SPECIES_NIDORANM":
                    mon = "SPECIES_NIDORAN_M"
                elif mon == "SPECIES_MRMIME":
                    mon = "SPECIES_MR_MIME"
                elif mon == "SPECIES_HOOH":
                    mon = "SPECIES_HO_OH"
                if mon not in mon_levelup_moveset:
                    mon_levelup_moveset[mon] = []
                curr_mon = mon
            elif curr_mon is not None:
                if "LEVEL_UP_MOVE" in line:
                    #    LEVEL_UP_MOVE( 1, MOVE_TACKLE),\n
                    ps = line.strip().split(",")
                    level = int(ps[0][ps[0].index('(') + 1:].strip().strip(','))
                    move = ps[1].strip().strip(',()')
                    mon_levelup_moveset[curr_mon].append([level, move])
                    n_levelup_moves += 1
    print("Read in %d level up moves" % n_levelup_moves)
    mon_tmhm_moveset = {}  # list of TMHM moves
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
                if mon not in mon_tmhm_moveset:
                    mon_tmhm_moveset[mon] = []
                curr_mon = mon
                # move on this line
                if line.count('TMHM') > 1:
                    tmhm = line.split('TMHM')[2].strip().strip('()')
                    move = '_'.join(tmhm.split('_')[1:]).strip('(),')
                    mon_tmhm_moveset[curr_mon].append("MOVE_%s" % move)
                    n_tmhm_moves += 1
            elif curr_mon is not None:
                if "TMHM" in line:
                    #                                        | TMHM(TM09_BULLET_SEED)\n
                    tmhm = line.split('TMHM')[1].strip().strip('()')
                    move = '_'.join(tmhm.split('_')[1:]).strip('(),')
                    mon_tmhm_moveset[curr_mon].append("MOVE_%s" % move)
                    n_tmhm_moves += 1
    print("Read in %d TM/HM moves" % n_tmhm_moves)

    # Write the result.
    print("Writing data to JSON format...")
    d = {"mon_metadata": mon_metadata,
         "type_list": type_list,
         "mon_evolution": mon_evolution,
         "mon_levelup_moveset": mon_levelup_moveset,
         "mon_tmhm_moveset": mon_tmhm_moveset}
    with open(args.output_fn, 'w') as f:
        json.dump(d, f, indent=2)
    print("... done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read mon metadata and write to usable JSON.')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the mon data')
    args = parser.parse_args()

    main(args)
