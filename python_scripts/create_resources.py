import argparse
import copy
import json
import numpy as np


# Paths.
base_stats_orig_fn = 'orig/base_stats.h'
evolution_fn = 'orig/evolution.h'


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
                    mon_metadata[species_mon] = {}
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

    # Write the result.
    print("Writing data to JSON format...")
    d = {"mon_metadata": mon_metadata,
         "type_list": type_list,
         "mon_evolution": mon_evolution}
    with open(args.output_fn, 'w') as f:
        json.dump(d, f, indent=2)
    print("... done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read mon metadata and write to usable JSON.')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the mon data')
    args = parser.parse_args()

    main(args)
