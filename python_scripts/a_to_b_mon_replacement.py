import argparse
import json

# Paths.
wild_encounters_orig_fn = 'orig/wild_encounters.h'
wild_encounters_target_fn = '../src/data/wild_encounters.h'
trainer_parties_orig_fn = 'orig/trainer_parties.h'
trainer_parties_target_fn = '../src/data/trainer_parties.h'
trainers_orig_fn = 'orig/trainers.h'
trainers_target_fn = '../src/data/trainers.h'
starter_choose_orig_fn = 'orig/starter_choose.c'
starter_choose_target_fn = '../src/starter_choose.c'
battle_setup_orig_fn = 'orig/battle_setup.c'
battle_setup_target_fn = '../src/battle_setup.c'
battle_tower_orig_fn = 'orig/battle_tower.c'
battle_tower_target_fn = '../src/battle_tower.c'

# Parameters
lvl_increase_for_move_strip = 0.1  # Give a 10% boost to pkm lvl if custom moves were removed.

# Encounter path orig -> target files.
legendary_fn = {"SPECIES_REGIROCK":
                    [['orig/SPECIES_REGIROCK_scripts.inc',
                      '../data/maps/DesertRuins/scripts.inc']],
                "SPECIES_REGICE":
                    [['orig/SPECIES_REGICE_scripts.inc',
                      '../data/maps/IslandCave/scripts.inc']],
                "SPECIES_REGISTEEL":
                    [['orig/SPECIES_REGISTEEL_scripts.inc',
                      '../data/maps/AncientTomb/scripts.inc']],
                "SPECIES_KYOGRE":
                    [['orig/SPECIES_KYOGRE_scripts.inc',
                      '../data/maps/MarineCave_End/scripts.inc']],
                "SPECIES_GROUDON":
                    [['orig/SPECIES_GROUDON_scripts.inc',
                      '../data/maps/TerraCave_End/scripts.inc']],
                "SPECIES_RAYQUAZA":
                    [['orig/SPECIES_RAYQUAZA_scripts.inc',
                      '../data/maps/SkyPillar_Top/scripts.inc']],
                "SPECIES_MEW":
                    [['orig/SPECIES_MEW_scripts.inc',
                      '../data/maps/FarawayIsland_Interior/scripts.inc']],
                "SPECIES_LUGIA":
                    [['orig/SPECIES_LUGIA_scripts.inc',
                      '../data/maps/NavelRock_Bottom/scripts.inc']],
                "SPECIES_HO_OH":
                    [['orig/SPECIES_HO_OH_scripts.inc',
                      '../data/maps/NavelRock_Top/scripts.inc']],
                "SPECIES_DEOXYS":
                    [['orig/SPECIES_DEOXYS_scripts.inc',
                      '../data/maps/BirthIsland_Exterior/scripts.inc']],
                # Not legendary, but encounter setups are the same.
                "SPECIES_VOLTORB":
                    [['orig/NEW_MAUVILLE_scripts.inc',
                      '../data/maps/NewMauville_Inside/scripts.inc']],
                "SPECIES_ELECTRODE":
                    [['orig/ELECTRODE_scripts.inc',
                      '../data/maps/AquaHideout_B1F/scripts.inc']],
                "SPECIES_SUDOWOODO":
                    [['orig/SUDOWOODO_scripts.inc',
                      '../data/maps/BattleFrontier_OutsideEast/scripts.inc']],
                "SPECIES_KECLEON":
                    [['orig/KECLEON_scripts.inc',
                      '../data/maps/Route120/scripts.inc'],
                     ['orig/KECLEON_2_scripts.inc',
                      '../data/scripts/kecleon.inc']],
                # Gift given mon where only one appears per script.
                "SPECIES_CASTFORM":
                    [['orig/CASTFORM_scripts.inc',
                      '../data/maps/Route119_WeatherInstitute_2F/scripts.inc']],
                "SPECIES_BELDUM":
                    [['orig/BELDUM_scripts.inc',
                      '../data/maps/MossdeepCity_StevensHouse/scripts.inc']]
                }
# Encounter/gift scripts where multiple species are in the file.
latios_latias_orig_fn = 'orig/LATIOS_LATIAS_scripts.inc'
latios_latias_target_fn = '../data/maps/SouthernIsland_Interior/scripts.inc'
fossils_orig_fn = 'orig/FOSSILS_scripts.inc'
fossils_target_fn = '../data/maps/RustboroCity_DevonCorp_2F/scripts.inc'

# Gym encounter path orig -> target files for replacing TM gifts by orig type.
# Also changes chats to mention the correct type, either from the Aide (gym) or leader (E4).
gym_fn = {"TYPE_ROCK":
              [['orig/TYPE_ROCK_scripts.inc',
                '../data/maps/RustboroCity_Gym/scripts.inc']],
          "TYPE_FIGHTING":
              [['orig/TYPE_FIGHTING_scripts.inc',
                '../data/maps/DewfordTown_Gym/scripts.inc']],
          "TYPE_ELECTRIC":
              [['orig/TYPE_ELECTRIC_scripts.inc',
                '../data/maps/MauvilleCity_Gym/scripts.inc']],
          "TYPE_FIRE":
              [['orig/TYPE_FIRE_scripts.inc',
                '../data/maps/LavaridgeTown_Gym_1F/scripts.inc']],
          "TYPE_NORMAL":
              [['orig/TYPE_NORMAL_scripts.inc',
                '../data/maps/PetalburgCity_Gym/scripts.inc']],
          "TYPE_FLYING":
              [['orig/TYPE_FLYING_scripts.inc',
                '../data/maps/FortreeCity_Gym/scripts.inc']],
          "TYPE_PSYCHIC":
              [['orig/TYPE_PSYCHIC_scripts.inc',
                '../data/maps/MossdeepCity_Gym/scripts.inc']],
          "TYPE_WATER":
              [['orig/TYPE_WATER_scripts.inc',
                '../data/maps/SootopolisCity_Gym_1F/scripts.inc'],
               ['orig/TYPE_WATER_2_scripts.inc',
                '../data/maps/EverGrandeCity_ChampionsRoom/scripts.inc']],
          "TYPE_STEEL":  # Steven's gift in Dewford
              [['orig/TYPE_STEEL_scripts.inc',
                '../data/maps/GraniteCave_StevensRoom/scripts.inc']],
          "TYPE_DARK":
              [['orig/TYPE_DARK_scripts.inc',
                '../data/maps/EverGrandeCity_SidneysRoom/scripts.inc']],
          "TYPE_GHOST":
              [['orig/TYPE_GHOST_scripts.inc',
                '../data/maps/EverGrandeCity_PhoebesRoom/scripts.inc']],
          "TYPE_ICE":
              [['orig/TYPE_ICE_scripts.inc',
                '../data/maps/EverGrandeCity_GlaciasRoom/scripts.inc']],
          "TYPE_DRAGON":
              [['orig/TYPE_DRAGON_scripts.inc',
                '../data/maps/EverGrandeCity_DrakesRoom/scripts.inc']],
         }

# TM gifts per type.
tm_gifts = {"TYPE_FIGHTING": "ITEM_TM01",  # FOCUS PUNCH
            "TYPE_FLYING": "ITEM_TM40",  # AERIAL ACE
            "TYPE_ELECTRIC": "ITEM_TM34",  # SHOCK WAVE
            "TYPE_POISON": "ITEM_TM06",  # TOXIC 
            "TYPE_GHOST": "ITEM_TM30",  # SHADOW BALL
            "TYPE_NORMAL": "ITEM_TM42",  # FACADE
            "TYPE_BUG": "ITEM_TM19",  # GIGA DRAIN (no BUG TMs in Gen III)
            "TYPE_PSYCHIC": "ITEM_TM29",  # PSYCHIC
            "TYPE_GROUND": "ITEM_TM26",  # EARTHQUAKE
            "TYPE_DRAGON": "ITEM_TM02",  # DRAGON CLAW
            "TYPE_GRASS": "ITEM_TM22",  # SOLAR BEAM 
            "TYPE_STEEL": "ITEM_TM47",  # STEEL WING
            "TYPE_ICE": "ITEM_TM13",  # ICE BEAM
            "TYPE_FIRE": "ITEM_TM50",  # OVERHEAT
            "TYPE_WATER": "ITEM_TM03",  # WATER PULSE
            "TYPE_DARK": "ITEM_TM49",  # SNATCH
            "TYPE_ROCK": "ITEM_TM39"  # ROCK TOMB
            }
# TM gifts move overrides by leader.
tm_gift_overrides = {"TYPE_FIGHTING": "MOVE_FOCUS_PUNCH",  # FOCUS PUNCH
                     "TYPE_FLYING": "MOVE_AERIAL_ACE",  # AERIAL ACE
                     "TYPE_ELECTRIC": "MOVE_SHOCK_WAVE",  # SHOCK WAVE
                     "TYPE_POISON": "MOVE_TOXIC",  # TOXIC 
                     "TYPE_GHOST": "MOVE_SHADOW_BALL",  # SHADOW BALL
                     "TYPE_NORMAL": "MOVE_FACADE",  # FACADE
                     "TYPE_BUG": "MOVE_GIGA_DRAIN",  # GIGA DRAIN (no BUG TMs in Gen III)
                     "TYPE_PSYCHIC": "MOVE_PSYCHIC",  # PSYCHIC
                     "TYPE_GROUND": "MOVE_EARTHQUAKE",  # EARTHQUAKE
                     "TYPE_DRAGON": "MOVE_DRAGON_CLAW",  # DRAGON CLAW
                     "TYPE_GRASS": "MOVE_SOLAR_BEAM",  # SOLAR BEAM 
                     "TYPE_STEEL": "MOVE_STEEL_WING",  # STEEL WING
                     "TYPE_ICE": "MOVE_ICE_BEAM",  # ICE BEAM
                     "TYPE_FIRE": "MOVE_OVERHEAT",  # OVERHEAT
                     "TYPE_WATER": "MOVE_WATER_PULSE",  # WATER PULSE
                     "TYPE_DARK": "MOVE_SNATCH",  # SNATCH
                     "TYPE_ROCK": "MOVE_ROCK_TOMB"  # ROCK TOMB
                     }
tm_gift_overrides_r = {tm_gift_overrides[t]: t for t in tm_gift_overrides}
leader_name_strs = {'Roxanne': "TYPE_ROCK", 'Brawly': "TYPE_FIGHTING",
                    'Wattson': "TYPE_ELECTRIC", 'Flannery': "TYPE_FIRE",
                    'Norman': "TYPE_NORMAL", 'Winona': "TYPE_FLYING",
                    'TateAndLiza': "TYPE_PSYCHIC", 'Juan': "TYPE_WATER"}


def main(args):
    
    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        mon_map = d["mon_map"]
        mon_map = {str(a): str(mon_map[a]) for a in mon_map}
        type_map = d["type_map"]
        type_map = {str(t): str(type_map[t]) for t in type_map}
        mon_move_map = d["mon_move_map"] if "mon_move_map" in d else None
    if mon_move_map is None:
        print("WARNING: will remove custom moves and increase levels by %.2f percent." % lvl_increase_for_move_strip)

    # Replace wild encounters.
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
        curr_mon = None
        curr_leader = False
        for line in f_orig.readlines():
            if "static const struct" in line:
                # static const struct TrainerMonNoItemDefaultMoves sParty_MayRustboroTorchic[] = {\n
                curr_leader = None
                for l in leader_name_strs:
                    if "_%s" % l in line:
                        curr_leader = l

            if ".species" in line:
                #    .species = SPECIES_MEDICHAM,
                curr_mon = line.split('=')[1].strip().strip(',')

            if mon_move_map is None and ".lvl =" in line:
                #    .lvl = 43,
                lvl_str = line.split("=")[1].strip().strip(',')
                new_lvl = int(int(lvl_str) * (1 + lvl_increase_for_move_strip) + 0.5)
                trainer_parties_str += line.replace(" %s," % lvl_str,
                                                    " %s," % str(new_lvl))

            elif ".moves" in line:
                if mon_move_map is not None and ".moves" in line:  # Select new custom moves
                    #    .moves = {MOVE_PSYCHIC, MOVE_NONE, MOVE_NONE, MOVE_NONE}
                    ps = line.split("=")
                    moves = [m.strip() for m in ps[1].strip().strip('{}').split(',')]
                    if curr_leader is None:
                        moves = [mon_move_map[mon_map[curr_mon]][m] if m != "MOVE_NONE" else "MOVE_NONE"
                                 for m in moves]
                    else:
                        for idx in range(len(moves)):
                            if moves[idx] != "MOVE_NONE":
                                if (moves[idx] in tm_gift_overrides_r and
                                        tm_gift_overrides_r[moves[idx]] == leader_name_strs[curr_leader]):
                                    gift_type = tm_gift_overrides_r[moves[idx]]
                                    moves[idx] = tm_gift_overrides[type_map[gift_type]]
                                else:
                                    moves[idx] = mon_move_map[mon_map[curr_mon]][moves[idx]]
                    new_line = ps[0] + " {" + ', '.join(moves) + "}\n"
                    trainer_parties_str += new_line
            else:
                trainer_parties_str += line
    # Now replace all instances of A with B.
    for a in mon_map:
        trainer_parties_str = trainer_parties_str.replace("%s," % a,
                                                          "%s_REPLACED," % mon_map[a])
    trainer_parties_str = trainer_parties_str.replace("_REPLACED", "")
    # Because we're stripping moves, need to change trainer classes to DefaultMoves
    if mon_move_map is None:
        trainer_parties_str = trainer_parties_str.replace("CustomMoves ", "DefaultMoves ")
    # Write the result.
    with open(trainer_parties_target_fn, 'w') as f_target:
        f_target.write(trainer_parties_str)
    # As a result of changing trainer classes, need to update the trainers.h file too.
    with open(trainers_target_fn, 'w') as f_target:
        with open(trainers_orig_fn, 'r') as f_orig:
            contents = f_orig.read()
        if mon_move_map is None:
            contents = contents.replace("CustomMoves ", "DefaultMoves ")
            contents = contents.replace(" | F_TRAINER_PARTY_CUSTOM_MOVESET,", ",")
            contents = contents.replace(" = F_TRAINER_PARTY_CUSTOM_MOVESET,", " = 0,")
        f_target.write(contents)

    # Replace event encounters (one by one basis, looks like).
    for legendary in legendary_fn:
        for src_fn, target_fn in legendary_fn[legendary]:
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
    # Replace Lileep/Anorith as a special case (they're in the same file.)
    with open(fossils_target_fn, 'w') as f_target:
        with open(fossils_orig_fn, 'r') as f_orig:
            contents = f_orig.read()
        for fossil in ['SPECIES_LILEEP', 'SPECIES_ANORITH']:
            contents = contents.replace("%s" % fossil,
                                        "%s_REPLACED" % mon_map[fossil])
            contents = contents.replace("_REPLACED", "")
        f_target.write(contents)

    # Replace starters
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

    # Replace special battle setup 'mon.
    # For now, this is just the special battle with Wally catching Ralts.
    with open(battle_setup_orig_fn, 'r') as f_orig:
        contents = f_orig.read()
    contents = contents.replace("SPECIES_RALTS,", "%s," % mon_map['SPECIES_RALTS'])
    with open(battle_setup_target_fn, 'w') as f_target:
        f_target.write(contents)

    # Replace battle tower 'mon.
    # For now, this is just Steven's 'mon for the double battle in Mossdeep.
    with open(battle_tower_orig_fn, 'r') as f_orig:
        contents = f_orig.read()
    for steven_mon in ['SPECIES_METANG', 'SPECIES_SKARMORY', 'SPECIES_AGGRON']:
        contents = contents.replace("%s," % steven_mon, "%s," % mon_map[steven_mon])
    with open(battle_tower_target_fn, 'w') as f_target:
        f_target.write(contents)

    # Replace gym TM gifts and GymGuideAdvice type advice.
    for tm_type in gym_fn:
        type_str = tm_type.split('_')[1]
        for source_fn, target_fn in gym_fn[tm_type]:
            with open(target_fn, 'w') as f_target:
                with open(source_fn, 'r') as f_orig:
                    for line in f_orig.readlines():
                        if "giveitem ITEM_" in line:
                            f_target.write("   giveitem %s\n" % tm_gifts[type_map[tm_type]])
                        elif "%s-type" % type_str in line:
                            f_target.write(line.replace("%s-type" % type_str,
                                                        "%s-type" % type_map[tm_type].split('_')[1]))
                        else:
                            f_target.write(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite wild and trainer mon.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input file with the mon map')
    args = parser.parse_args()


    main(args)
