import argparse
import json

# Paths.
base_stats_fns = ['orig/base_stats.h', '../src/data/pokemon/base_stats.h']
evolution_fns = ['orig/evolution.h', '../src/data/pokemon/evolution.h']
level_up_learnsets_fn = ['orig/level_up_learnsets.h', '../src/data/pokemon/level_up_learnsets.h']
tmhm_learnsets_fn = ['orig/tmhm_learnsets.h', '../src/data/pokemon/tmhm_learnsets.h']


def main(args):
    tab_str = '    '
    
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_metadata = d['mon_metadata']
        mon_evolution = d['mon_evolution']
        mon_levelup_moveset = d['mon_levelup_moveset']
        mon_tmhm_moveset = d['mon_tmhm_moveset']

    # Replace base stats.
    print("Replacing base stats by reading from '%s' and writing to '%s'" % (base_stats_fns[0], base_stats_fns[1]))
    n_edited_lines = n_lines = 0
    with open(base_stats_fns[1], 'w') as f_out:

        curr_species = None
        with open(base_stats_fns[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # New species entry
                #     [SPECIES_BULBASAUR] =
                if line.strip()[:len('[SPECIES_')] == '[SPECIES_':
                    curr_species = line.strip().strip('=[] ')

                # Metadata line to be overwritten.
                #         .baseHP        = 45,
                if curr_species is not None and '.' in line and '=' in line:
                    ps = line.strip().split('=')
                    stat = ps[0].strip().strip('.')
                    if stat == 'abilities':
                        #         .abilities = {ABILITY_OVERGROW, ABILITY_NONE},
                        line = '%s%s.%s = {%s, %s},\n' % (tab_str, tab_str, stat,
                                                          mon_metadata[curr_species]['abilities'][0],
                                                          mon_metadata[curr_species]['abilities'][1])
                        n_edited_lines += 1
                    elif stat == 'eggGroup1':
                        line = '%s%s.%s = %s,\n' % (tab_str, tab_str, stat,
                                                    str(mon_metadata[curr_species]['eggGroups'][0]))
                        n_edited_lines += 1
                    elif stat == 'eggGroup2':
                        line = '%s%s.%s = %s,\n' % (tab_str, tab_str, stat,
                                                    str(mon_metadata[curr_species]['eggGroups'][1]))
                        n_edited_lines += 1
                    elif stat in mon_metadata[curr_species]:
                        line = '%s%s.%s = %s,\n' % (tab_str, tab_str, stat, str(mon_metadata[curr_species][stat]))
                        n_edited_lines += 1

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # Replace evolutions.
    print("Replacing evolutions by reading from '%s' and writing to '%s'" % (evolution_fns[0], evolution_fns[1]))
    n_edited_lines = n_lines = 0
    with open(evolution_fns[1], 'w') as f_out:
        with open(evolution_fns[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # Data entry.
                #     [SPECIES_BULBASAUR]  = {{EVO_LEVEL, 16, SPECIES_IVYSAUR}},
                species = line.strip().split('=')[0].strip(' []')
                if species in mon_evolution:
                    if len(mon_evolution[species]) > 0:
                        n_edited_lines += 1
                        line = '%s[%s] = {%s},\n' % (tab_str, species,
                                                     ', '.join(['{%s}' % ', '.join([str(s)
                                                                                    for s in mon_evolution[species][idx]])
                                                                for idx in range(len(mon_evolution[species]))]))
                    else:
                        # Don't write any data for a mon that now doesn't evolve.
                        n_edited_lines += 1
                        continue

                # Metadata line to be ignored
                #                             {EVO_ITEM, ITEM_SUN_STONE, SPECIES_BELLOSSOM}},
                if '=' not in line and '},' in line:
                    n_edited_lines += 1
                    continue

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # Replace level up learnsets.
    # Replace evolutions.
    print("Replacing level up learnsets by reading from '%s' and writing to '%s'" %
          (level_up_learnsets_fn[0], level_up_learnsets_fn[1]))
    n_edited_lines = n_lines = 0
    with open(level_up_learnsets_fn[1], 'w') as f_out:
        with open(level_up_learnsets_fn[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # New species entry, go ahead and dump everything.
                # static const u16 sBulbasaurLevelUpLearnset[] = {
                if 'LevelUpLearnset' in line:
                    ps = line.strip().split()
                    curr_species = ps[3][1:-len('LevelUpLearnset[]')]

                    if 'Species' not in curr_species:

                        if curr_species == 'NidoranF':
                            curr_species = 'Nidoran_F'
                        elif curr_species == 'NidoranM':
                            curr_species = 'Nidoran_M'
                        elif curr_species == 'Mrmime':
                            curr_species = 'Mr_mime'
                        elif curr_species == 'HoOh':
                            curr_species = 'Ho_Oh'
                        curr_species = 'SPECIES_%s' % curr_species.upper()

                        # Create a line entry with all levelup moves formatted.
                        for lvl, mv in mon_levelup_moveset[curr_species]:
                            n_edited_lines += 1
                            line += '%sLEVEL_UP_MOVE( %s, %s),\n' % (tab_str, str(lvl), mv)

                    else:
                        curr_species = None

                # Line to be ignored.
                #     LEVEL_UP_MOVE( 1, MOVE_TACKLE),
                elif '#define' not in line and curr_species is not None and 'LEVEL_UP_MOVE' in line:
                    continue

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # Reaad in TMHM numbers to prepend to moves.
    print("Reading through TMHM file to get numbers ahead of moves...")
    tmhm_prefix = {}
    with open(tmhm_learnsets_fn[0], 'r') as f:
        for line in f.readlines():
            # Header line for new species
            #     [SPECIES_BULBASAUR]   = TMHM_LEARNSET(TMHM(TM06_TOXIC)
            if '= TMHM_LEARNSET(' in line and 'TMHM_LEARNSET(0)' not in line:
                ps = line.strip().split('TMHM(')[1].strip(')').split('_')
                prefix = ps[0]
                move = '_'.join(ps[1:])
                tmhm_prefix['MOVE_' + move] = prefix
            # Follow-up line
            #                                         | TMHM(TM09_BULLET_SEED)
            #                                         | TMHM(HM06_ROCK_SMASH)),
            if '| ' in line:
                ps = line.strip().split('TMHM(')[1].strip('),').split('_')
                prefix = ps[0]
                move = '_'.join(ps[1:])
                tmhm_prefix['MOVE_' + move] = prefix
    print("... done; read in %d move prefixes" % len(tmhm_prefix))

    # Replace TMHM learnsets.
    print("Replacing level up learnsets by reading from '%s' and writing to '%s'" %
          (tmhm_learnsets_fn[0], tmhm_learnsets_fn[1]))
    n_edited_lines = n_lines = 0
    with open(tmhm_learnsets_fn[1], 'w') as f_out:
        with open(tmhm_learnsets_fn[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # New species entry, go ahead and dump everything.
                #     [SPECIES_BULBASAUR]   = TMHM_LEARNSET(TMHM(TM06_TOXIC)
                if '= TMHM_LEARNSET(' in line and 'TMHM_LEARNSET(0)' not in line:
                    ps = line.strip().split()
                    curr_species = ps[0].strip('[]')

                    # Create a line entry with all TMHM moves formatted.
                    # TMHM(TM06_TOXIC) | TMHM(TM09_BULLET_SEED) | ...
                    n_edited_lines += len(mon_tmhm_moveset[curr_species])
                    line = '%s[%s] = TMHM_LEARNSET(%s),\n' % (tab_str, curr_species,
                                                              ' | \n'.join(['TMHM(%s_%s)' %
                                                                          (tmhm_prefix[mv], mv[len('MOVE_'):])
                                                                          for mv in mon_tmhm_moveset[curr_species]]))

                elif '|' in line:
                    continue

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # TODO: Replace Pokedex entries.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite wild and trainer mon.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input json file with sampled mon metadata')
    args = parser.parse_args()


    main(args)
