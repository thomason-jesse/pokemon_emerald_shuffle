import argparse
import json
import os


# Paths.
wild_encounters_orig_fn = 'orig/wild_encounters.h'
wild_encounters_target_fn = '../src/data/wild_encounters.h'
trainer_parties_orig_fn = 'orig/trainer_parties.h'
trainer_parties_target_fn = '../src/data/trainer_parties.h'
starter_choose_orig_fn = 'orig/starter_choose.c'
starter_choose_target_fn = '../src/starter_choose.c'

# Lendary encounter path orig -> target files.
legendary_fn = {"SPECIES_REGIROCK": ['SPECIES_REGIROCK_scripts.inc',
                                     '../data/maps/DesertRuins/scripts.inc'],
                "SPECIES_REGICE": ['SPECIES_REGICE_scripts.inc',
                                   '../data/maps/IslandCave/scripts.inc'],
                "SPECIES_REGISTEEL": ['SPECIES_REGISTEEL_scripts.inc',
                                      '../data/maps/AncientTomb/scripts.inc'],
                "SPECIES_KYOGRE": ['SPECIES_KYOGRE_scripts.inc',
                                   '../data/maps/MarineCave_End/scripts.inc'],
                "SPECIES_GROUDON": ['SPECIES_GROUDON_scripts.inc',
                                   '../data/maps/TerraCave_End/scripts.inc'],
                "SPECIES_RAYQUAZA": ['SPECIES_RAYQUAZA_scripts.inc',
                                     '../data/maps/SkyPillar_Top/scripts.inc'],
                "SPECIES_MEW": ['SPECIES_MEW_scripts.inc',
                                '../data/maps/FarawayIsland_Interior/scripts.inc'],
                "SPECIES_LUGIA": ['SPECIES_LUGIA_scripts.inc',
                                  '../data/maps/NavalRock_Bottom/scripts.inc'],
                "SPECIES_HO_OH": ['SPECIES_HO_OH_scripts.inc',
                                  '../data/maps/NavalRock_Top/scripts.inc'],
                "SPECIES_DEOXYS": ['SPECIES_DEOXYS_scripts.inc',
                                   '../data/maps/BirthIsland_Exterior/scripts.inc'],}
latios_latias_orig_fn = 'LATIOS_LATIAS_scripts.inc'
latios_latias_target_fn = '../data/maps/SouthernIsland_Interior/scripts.inc'


def main(args):
    
    mon_map = {}
    with open(args.input_fn, 'r') as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            ab = line.strip().split('->')
            if ab[0] in mon_map:
                print("WARNING: one-to-many mapping for '%s'" % ab[0])
            mon_map[ab[0]] = ab[1]

    # Replace wild encounters.

    wild_encounter_str = ''
    # First, just copy the contents of the original file.
    with open(wild_encounters_orig_fn, 'r') as f_orig:
        wild_encounter_str = f_orig.read()
    # Now replace all instances of A with B.
    for a in mon_map:
        wild_encounter_str = wild_encounter_str.replace("%s }," % a,
                                                        "%s_REPLACED }," % mon_map[a])
    wild_encounter_str = wild_encounter_str.replace("_REPLACED", "")
    # Write the result.
    with open(wild_encounters_target_fn, 'w') as f_target:
        f_target.write(wild_encounter_str)

    # Replace trainer parties.
    trainer_parties_str = ''
    # First, just copy the contents of the original file.
    # Strip out fixed movesets for now.
    with open(trainer_parties_orig_fn, 'r') as f_orig:
        for line in f_orig.readlines():
            # TODO: do something more clever than just erasing movesets.
            if ".moves" not in line:
                trainer_parties_str += line
    # Now replace all instances of A with B.
    for a in mon_map:
        trainer_parties_str = trainer_parties_str.replace("%s," % a,
                                                          "%s_REPLACED," % mon_map[a])
    trainer_parties_str = trainer_parties_str.replace("_REPLACED", "")
    # Write the result.
    with open(trainer_parties_target_fn, 'w') as f_target:
        f_target.write(trainer_parties_str)

    # Replace event encounters (one by one basis, looks like).
    for legendary in legendary_fn:
        src_fn, target_fn = legendary_fn[legendary]
        with open(target_fn, 'w') as f_target:
            with open(src_fn, 'r') as f_orig:
                contents = f_orig.read()
            contents = contents.replace("%s," % legendary,
                                        "%s_REPLACED," % mon_map[legendary])
            contents = contents.replace(", %s" % legendary,
                                        ", %s_REPLACED" % mon_map[legendary])
            contents = contents.replace("_REPLACED", "")
            f_target.write(contents)
    # Replace Latios/Latias as a special case (they're in the same file.)
    with open(latios_latias_target_fn, 'w') as f_target:
        with open(latios_latias_orig_fn, 'r') as f_orig:
            contents = f_orig.read()
        for legendary in ['SPECIES_LATIOS', 'SPECIES_LATIAS']:
            contents = contents.replace("%s," % legendary,
                                            "%s_REPLACED," % mon_map[legendary])
            contents = contents.replace("_REPLACED", "")
        f_target.write(contents)

    # Replace starters
    starter_choose_str = ''
    # First, just copy the contents of the original file.
    with open(starter_choose_orig_fn, 'r') as f_orig:
        starter_choose_str = f_orig.read()
    # Now replace all instances of A with B.
    for a in mon_map:
        starter_choose_str = starter_choose_str.replace("%s," % a,
                                                        "%s_REPLACED," % mon_map[a])
    starter_choose_str = starter_choose_str.replace("_REPLACED", "")
    # Write the result.
    with open(starter_choose_target_fn, 'w') as f_target:
        f_target.write(starter_choose_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite wild and trainer mon.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input file with the mon map')
    args = parser.parse_args()


    main(args)
