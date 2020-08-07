import argparse
import json

# Paths.
base_stats_fns = ['orig/base_stats.h', '../src/data/pokemon/base_stats.h']
evolution_fns = ['orig/evolution.h', '../src/data/pokemon/evolution.h']
level_up_learnsets_fns = ['orig/level_up_learnsets.h', '../src/data/pokemon/level_up_learnsets.h']
tmhm_learnsets_fns = ['orig/tmhm_learnsets.h', '../src/data/pokemon/tmhm_learnsets.h']
pokedex_fns = ['orig/pokedex_text.h', '../src/data/pokemon/pokedex_text.h']
names_fns = ['orig/species_names.h', '../src/data/text/species_names.h']

stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']


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
            data_dumped = False
            can_dump = False
            for line in f_in.readlines():
                n_lines += 1

                # Data entry.
                #     [SPECIES_BULBASAUR]  = {{EVO_LEVEL, 16, SPECIES_IVYSAUR}},
                if can_dump and not data_dumped:
                    line = ''
                    for species in mon_evolution:
                        if len(mon_evolution[species]) > 0:
                            n_edited_lines += 1
                            line += '%s[%s] = {%s},\n' % (tab_str, species,
                                                          ', '.join(['{%s}' % ', '.join([str(s)
                                                                                         for s in mon_evolution[species][idx]])
                                                                     for idx in range(len(mon_evolution[species]))]))
                    data_dumped = True

                # Metadata lines to be ignored
                #     [SPECIES_BULBASAUR]  = {{EVO_LEVEL, 16, SPECIES_IVYSAUR}},
                #                             {EVO_ITEM, ITEM_SUN_STONE, SPECIES_BELLOSSOM}},
                elif '[SPECIES_' in line or ('=' not in line and '},' in line):
                    n_edited_lines += 1
                    continue

                if line.strip() == '{':
                    can_dump = True

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # Replace level up learnsets.
    # Replace evolutions.
    print("Replacing level up learnsets by reading from '%s' and writing to '%s'" %
          (level_up_learnsets_fns[0], level_up_learnsets_fns[1]))
    n_edited_lines = n_lines = 0
    with open(level_up_learnsets_fns[1], 'w') as f_out:
        with open(level_up_learnsets_fns[0], 'r') as f_in:
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
    with open(tmhm_learnsets_fns[0], 'r') as f:
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
          (tmhm_learnsets_fns[0], tmhm_learnsets_fns[1]))
    n_edited_lines = n_lines = 0
    with open(tmhm_learnsets_fns[1], 'w') as f_out:
        with open(tmhm_learnsets_fns[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # New species entry, go ahead and dump everything.
                #     [SPECIES_BULBASAUR]   = TMHM_LEARNSET(TMHM(TM06_TOXIC)
                if '= TMHM_LEARNSET(' in line and 'TMHM_LEARNSET(0)' not in line:
                    ps = line.strip().split()
                    curr_species = ps[0].strip('[]')

                    # Create a line entry with all TMHM moves formatted.
                    # TMHM(TM06_TOXIC) | TMHM(TM09_BULLET_SEED) | ...
                    if len(mon_tmhm_moveset[curr_species]) > 0:
                        n_edited_lines += len(mon_tmhm_moveset[curr_species])
                        line = '%s[%s] = TMHM_LEARNSET(%s),\n' % (tab_str, curr_species,
                                                                  ' | \n'.join(['TMHM(%s_%s)' %
                                                                              (tmhm_prefix[mv], mv[len('MOVE_'):])
                                                                              for mv in mon_tmhm_moveset[curr_species]]))
                    else:
                        n_edited_lines += 1
                        line = '%s[%s] = TMHM_LEARNSET(0),\n' % (tab_str, curr_species)

                elif '|' in line:
                    continue

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # Read through pokedex to get line character counts.
    print("Reading through Pokedex original file to get max characters per line...")
    max_lc = 0
    with open(pokedex_fns[0], 'r') as f:
        for line in f.readlines():

            # Read text line
            #     "This is a newly discovered POKeMON.\n"
            #     "adding new rocks.");
            if '"' in line:
                lc = len(line.strip().strip(');').strip('"').strip())
                if lc > max_lc:
                    max_lc = lc
    max_lc -= 4
    print("... done; will write up to %d chars per line" % max_lc)

    # Replace Pokedex entries.
    print("Replacing pokedex entries reading from '%s' and writing to '%s'" %
          (pokedex_fns[0], pokedex_fns[1]))
    n_edited_lines = n_lines = 0
    with open(pokedex_fns[1], 'w') as f_out:
        with open(pokedex_fns[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # New species entry, go ahead and dump everything.
                # const u8 gBulbasaurPokedexText[] = _(
                if 'PokedexText[]' in line:
                    ps = line.strip().split()
                    curr_species = ps[2][1:-len('PokedexText[]')]

                    if 'Dummy' not in curr_species:

                        if curr_species == 'NidoranF':
                            curr_species = 'Nidoran_F'
                        elif curr_species == 'NidoranM':
                            curr_species = 'Nidoran_M'
                        elif curr_species == 'Mrmime':
                            curr_species = 'Mr_mime'
                        elif curr_species == 'HoOh':
                            curr_species = 'Ho_Oh'
                        curr_species = 'SPECIES_%s' % curr_species.upper()

                        # Create pokedex text.
                        text = 'Sim stats: %s. ' % mon_metadata[curr_species]['nn'][len('SPECIES_'):].replace('_', '-')
                        text += '; '.join(['%s %s' % (s[len('base'):], mon_metadata[curr_species][s])
                                          for s in stat_data]) + '. '
                        if curr_species in mon_evolution:
                            for _, ev_type, ev_val in mon_evolution[curr_species]:
                                if 'FRIENDSHIP' in ev_type:
                                    text += "Evolves by friendship"
                                    if 'DAY' in ev_type:
                                        text += " (day). "
                                    elif 'NIGHT' in ev_type:
                                        text += " (night). "
                                    else:
                                        text += '. '
                                elif type(ev_val) is int:
                                    if ev_type == 'EVO_BEAUTY':
                                        text += "Evolves when beauty %d. " % ev_val
                                    else:
                                        text += "Evolves lvl %d. " % ev_val
                                else:
                                    text += "Evolves with %s. " % ev_val[len('ITEM_'):].replace('_', ' ')
                        text += "Learns " + ', '.join([mon_levelup_moveset[curr_species][idx][1][len('MOVE_'):].replace('_', ' ')
                                                       for idx in
                                                       range(len(mon_levelup_moveset[curr_species]) - 1, -1, -1)])

                        # Dump as much as will fit into four lines.
                        for jdx in range(4):
                            n_edited_lines += 1
                            text_sub = text[:min(len(text), max_lc)]
                            text = text[min(len(text), max_lc):]
                            line += '%s"%s%s"%s\n' % (tab_str, text_sub, '' if jdx == 3 else '\\n', ');' if jdx == 3 else '')
                    else:
                        curr_species = None

                # Line to be ignored.
                #     LEVEL_UP_MOVE( 1, MOVE_TACKLE),
                elif curr_species is not None and '"' in line:
                    continue

                f_out.write(line)
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))

    # Replace species names.
    print("Replacing species names reading from '%s' and writing to '%s'" %
          (names_fns[0], names_fns[1]))
    n_edited_lines = n_lines = 0
    with open(names_fns[1], 'w') as f_out:
        with open(names_fns[0], 'r') as f_in:
            for line in f_in.readlines():
                n_lines += 1

                # New species entry.
                #     [SPECIES_BULBASAUR] = _("BULBASAUR"),
                if '=' in line:
                    species = line.strip().split('=')[0].strip().strip('[]')
                    if species in mon_metadata:
                        n_edited_lines += 1
                        line = '%s[%s] = _("%s"),\n' % (tab_str, species, mon_metadata[species]['name'])

                f_out.write(line.encode("ascii", errors="ignore").decode())
    print("... done; edited %d/%d lines" % (n_edited_lines, n_lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite wild and trainer mon.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input json file with sampled mon metadata')
    args = parser.parse_args()


    main(args)
