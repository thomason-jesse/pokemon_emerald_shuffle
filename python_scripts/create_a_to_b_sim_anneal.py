import argparse
import json
import math
import numpy as np
import os
import random
import sys


# Paths.
base_stats_orig_fn = 'orig/base_stats.h'
evolution_fn = '../src/data/pokemon/evolution.h'
trainer_parties_fn = 'orig/trainer_parties.h'
wild_encounters_fn = 'orig/wild_encounters.h'


# Metadata consts.
int_data = ['baseHP', 'baseAttack', 'baseDefense', 'baseSpeed', 'baseSpAttack',
            'baseSpDefense', 'catchRate', 'expYield', 'evYield_HP', 'evYield_Attack',
            'evYield_Defense', 'evYield_Speed', 'evYield_SpAttack', 'evYield_SpDefense',
            'eggCycles', 'friendship', 'safariZoneFleeRate']
stat_data = ['baseHP', 'baseAttack', 'baseDefense',
             'baseSpeed', 'baseSpAttack', 'baseSpDefense']

# Penalty hyperparameters.
stat_coeff = 1  # Use z score norm on stats, so 1 std dev in sum stat difference
evol_coeff = 5  # Penalize standard deviations for missing an evolution.
type_coeff = 3  # Penalize standard deviations for deviating from the target types.

# Simulated annealing hyperparameters.
max_iterations = 100000
print_every_n_iterations = max_iterations / 10
swaps_for_neighbor = 1  # how many assignment swaps to perform per iteration.

# Analytical solution hyperparameters.
starter_species = ['SPECIES_TREECKO', 'SPECIES_TORCHIC', 'SPECIES_MUDKIP']
# all_starter_species = starter_species + ['SPECIES_CYNDAQUIL', 'SPECIES_TOTODILE', 'SPECIES_CHIKORITA']
# all_starter_species += ['SPECIES_SQUIRTLE', 'SPECIES_CHARMANDER', 'SPECIES_BULBASAUR']
assignment_beam = 20  # max assignments to try in analytical solution.


def get_a_to_b_penalty(a, b, mon_map, mon_metadata, mon_evolution, type_map,
                       penalize_same_type):
    # Average difference in z score normalized stats forms the main part of the penalty.
    if a != b and not (a in starter_species and b in starter_species):
        # Average difference per stat (encourages matching fast with fast, etc.).
        # sp = sum([abs(mon_metadata[a][s] - mon_metadata[b][s])
        #           for s in stat_data])
        # Difference total normalized by num stats (encourages stat total match only).
        sp = abs(sum([mon_metadata[a][s] for s in stat_data]) -
                 sum([mon_metadata[b][s] for s in stat_data]))
    else:  # 2 std stat penalty for mapping a mon to itself or for mapping a starter to a starter.
        sp = 2 * len(stat_data)
    sp /= len(stat_data)

    # Penalize inconsistency of evolution form.
    # Bulb -> Char
    # Ivy -> Charm
    # Ven -> Charz
    # ep=1 means this mon maps to b but its evolution does not map to B.
    ep = 0
    if a in mon_evolution:
        if b in mon_evolution:
            a_evols = mon_evolution[a]
            a_evols_mapped = [mon_map[a_evols[idx]] if a_evols[idx] in mon_map else None
                              for idx in range(len(a_evols))]
            match = False
            for idx in range(len(a_evols_mapped)):
                if a_evols_mapped[idx] is None or a_evols_mapped[idx] in mon_evolution[b]:
                    match = True
                    break
            if not match:
                ep = 1
        else:
            ep = 1

    # Penalize rough type map mismatch. tp=1 means full mismatch / 0 perfect match.
    tp = 0
    for type_str, weight in [["type1", 0.67], ["type2", 0.33]]:
        if (mon_metadata[b][type_str] != type_map[mon_metadata[a][type_str]] or
                # Penalize for mismatch in mapping (above) or mapping a type to itself (below).
                (penalize_same_type and type_map[mon_metadata[a][type_str]] == mon_metadata[a][type_str])):
            tp += weight

    total_penalty = (sp * stat_coeff) + (ep * evol_coeff) + (tp * type_coeff)
    return total_penalty, sp, ep, tp


def score_map(mon_map, mon_list, type_list, mon_metadata, mon_evolution, penalize_same_type,
              as_exp_str=False):

    # First, build type confusion matrix.
    type_to_type_counts = [[0 for t1 in type_list] for t2 in type_list]
    for a in mon_map:
        b = mon_map[a]
        type_to_type_counts[type_list.index(mon_metadata[a]["type1"])][type_list.index(mon_metadata[b]["type1"])] += 1
        type_to_type_counts[type_list.index(mon_metadata[a]["type2"])][type_list.index(mon_metadata[b]["type2"])] += 1
    # Normalize.
    for idx in range(len(type_list)):
        type_to_type_counts[idx] = [float(type_to_type_counts[idx][jdx]) / sum(type_to_type_counts[idx])
                                    for jdx in range(len(type_list))]
    # Assign apparent type map maxes.
    rough_type_to_type = {}
    not_in_domain = type_list[:]
    not_in_range = type_list[:]
    for _ in range(len(type_list)):
        maxes = [max(type_to_type_counts[idx]) for idx in range(len(type_list))]
        idx_to_assign = int(np.argmax(maxes))
        candidates = [jdx for jdx in range(len(type_list))
                      if np.isclose(type_to_type_counts[idx_to_assign][jdx], 
                                    maxes[idx_to_assign])]
        jdx_assigned = random.choice(candidates)
        rough_type_to_type[type_list[idx_to_assign]] = type_list[jdx_assigned]
        del not_in_domain[not_in_domain.index(type_list[idx_to_assign])]
        del not_in_range[not_in_range.index(type_list[jdx_assigned])]
        # Remove these entries from the count structure.
        type_to_type_counts[idx_to_assign] = [-1 for jdx in range(len(type_list))]
        for idx in range(len(type_list)):
            type_to_type_counts[idx][jdx_assigned] = -1
    for idx in range(len(not_in_domain)):
        rough_type_to_type[not_in_domain[idx]] = not_in_range[idx]
        # print("final %s->%s" % (not_in_domain[idx], not_in_range[idx]))

    # Show rough type to type map.
    exp_str = ''
    if as_exp_str:
        exp_str += "Type->Type\n"
        for idx in range(len(type_list)):
            exp_str += '%s->%s\n' % (type_list[idx], rough_type_to_type[type_list[idx]])
        exp_str += '\n'

    # Now, pan over all the mappings and calculate the penalty per with explanation.
    penalty = 0
    penalty_per_mon = [0 for _ in range(len(mon_list))]
    spt, ept, tpt = 0, 0, 0
    if as_exp_str:
        exp_str += "Mon->Mon\n"
    for mon_idx in range(len(mon_list)):
        a = mon_list[mon_idx]
        b = mon_map[a]

        mon_penalty, sp, ep, tp = get_a_to_b_penalty(a, b, mon_map, mon_metadata,
                                                     mon_evolution, rough_type_to_type,
                                                     penalize_same_type)
        
        penalty += mon_penalty
        penalty_per_mon[mon_idx] += mon_penalty
        spt += sp * stat_coeff
        ept += ep * evol_coeff
        tpt += tp * type_coeff

        if as_exp_str:
            exp_str += "%s->%s\t\tsp=%.2f(%.2f)\tep=%d(%d)\ttp=%d(%d)\n" % \
                       (a, b,
                        sp, sp * stat_coeff,
                        ep, ep * evol_coeff,
                        tp, tp * type_coeff)

    if as_exp_str:
        exp_str += "\nPenalty total: %d\tspt=%d(%.2f)\tept=%d(%.2f)\ttpt=%d(%.2f)" % \
                   (penalty,
                    spt, spt / float(penalty),
                    ept, ept / float(penalty),
                    tpt, tpt / float(penalty))
        return exp_str

    return penalty, penalty_per_mon, rough_type_to_type


def make_assignment_and_propagate(a_to_b_penalty, dom_idx, img_jdx, type_map,
                                  mon_map, mon_image, mon_list, mon_evolution, mon_metadata,
                                  penalize_same_type, debug=False):
    penalty_added = 0

    mon_map[mon_list[dom_idx]] = mon_list[img_jdx]  # assignment
    if debug:
        print("%s->%s (p=%.2f)" % (mon_list[dom_idx], mon_list[img_jdx], a_to_b_penalty[dom_idx, img_jdx]))
    penalty_added += a_to_b_penalty[dom_idx, img_jdx]
    mon_image.add(mon_list[img_jdx])
    # If a and b both evolve, force evolution mapping.
    propagate_evol = True
    while propagate_evol:
        a_evols = mon_evolution[mon_list[dom_idx]] if mon_list[dom_idx] in mon_evolution else []
        b_evols = mon_evolution[mon_list[img_jdx]] if mon_list[img_jdx] in mon_evolution else []
        a_evols = [a for a in a_evols if a not in mon_map]
        b_evols = [b for b in b_evols if b not in mon_image]
        if len(a_evols) > 0 and len(b_evols) > 0:
            # Find best map among candidates.
            best_a = best_b = None
            best_p = None
            for a in a_evols:
                for b in b_evols:
                    new_p = a_to_b_penalty[mon_list.index(a), mon_list.index(b)]
                    if best_p is None or new_p < best_p:
                        best_p = new_p
                        best_a = a
                        best_b = b
            dom_idx = mon_list.index(best_a)
            img_jdx = mon_list.index(best_b)
            mon_map[mon_list[dom_idx]] = mon_list[img_jdx]  # assignment
            if debug:
                print("%s->%s (p=%.2f)" % (mon_list[dom_idx], mon_list[img_jdx], a_to_b_penalty[dom_idx, img_jdx]))
            penalty_added += a_to_b_penalty[dom_idx, img_jdx]
            mon_image.add(mon_list[img_jdx])
        else:
            propagate_evol = False
    # Update evolution penalties by recalculating matrix.
    # TODO: this can be made more efficient by in-place updates.
    n_mon = len(mon_list)
    for idx in range(n_mon):
        a = mon_list[idx]
        for jdx in range(n_mon):
            b = mon_list[jdx]
            if a in mon_map or b in mon_image:
                mon_penalty = float("inf")  # already assigned
            else:
                mon_penalty, _, _, _ = get_a_to_b_penalty(a, b, mon_map, 
                                                          mon_metadata,
                                                          mon_evolution, type_map,
                                                          penalize_same_type)
            a_to_b_penalty[idx, jdx] = mon_penalty
    # if debug:
        # print("\tpenalty added %d" % penalty_added)
    return penalty_added


def main(args):
    type_map = {}
    if args.type_mapping not in ['random', 'fixed']:
        tms = args.type_mapping.strip().split(',')
        for tm in tms:
            ps = tm.split('-')
            type_map[ps[0]] = ps[1]
        args.type_mapping = 'random'

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
                    else:
                        mon_metadata[curr_mon][data_name] = data_str
                    if data_name == "type1" or data_name == "type2":
                        if data_str not in type_list:
                            type_list.append(data_str)
    # Perform z score normalization on numeric metadata.
    mon_list = list(mon_metadata.keys())
    for data_name in int_data:
        data_mean = np.mean([mon_metadata[a][data_name] for a in mon_list])
        data_std = np.std([mon_metadata[a][data_name] for a in mon_list])
        for a in mon_list:
            mon_metadata[a][data_name] = (mon_metadata[a][data_name] - data_mean) / data_std
    n_mon = len(mon_list)
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
    mon_evolution_r = {}
    for a in mon_evolution:
        for b in mon_evolution[a]:
            if b in mon_evolution_r:
                print("WARNING: multiple base species for %d" % b)
            mon_evolution_r[b] = a
    print("Read in %d mon evolutions" % len(mon_evolution))

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

    # Perform simulated annealing over an analytically chosen, greedy initial assignment.
    mon_map = {}

    # First, assign a random type -> type map.
    penalize_same_type = None
    type_list_to_assign = [t for t in type_list if t not in type_map]
    type_list_to_image = [t for t in type_list if not np.any([t == type_map[u]
                                                              for u in type_map])]
    idxs = list(range(len(type_list_to_assign)))
    if args.type_mapping == 'random':
        penalize_same_type = True
        while np.any([type_list_to_assign[idx] == type_list_to_image[idxs[idx]]
                      for idx in range(len(type_list_to_assign))]):
            random.shuffle(idxs)
        print("Random initial type map:")
    elif args.type_mapping == 'fixed':
        penalize_same_type = False
        print("Fixed initial type map:")
    for idx in range(len(type_list_to_assign)):
        type_map[type_list_to_assign[idx]] = type_list_to_image[idxs[idx]]
    for t in type_map:
        print("\t%s->%s" % (t, type_map[t]))

    # Create an initial penalty weights matrix.
    print("Running greedy solver given type map...")
    n_mon = len(mon_list)
    a_to_b_penalty = np.zeros(shape=(n_mon, n_mon))
    for idx in range(n_mon):
        a = mon_list[idx]
        for jdx in range(n_mon):
            b = mon_list[jdx]
            mon_penalty, _, _, _ = get_a_to_b_penalty(a, b, mon_map, 
                                                      mon_metadata, mon_evolution, type_map,
                                                      penalize_same_type)
            a_to_b_penalty[idx, jdx] = mon_penalty

    # Initially, assign A -> B map in order of A's importance, minimizing B assignments per.
    mon_image = set()
    for s in starter_species:  # Ensure starters are assigned first.
        mon_n_appearances[mon_list.index(s)] += max(mon_n_appearances)
    mon_by_importance = [mon_list[idx] for idx in
                         sorted(range(n_mon), key=lambda ky:mon_n_appearances[ky], reverse=True)]
    print("... First, assigning by importance (ranked list of mon appearance in-game)...")
    for a in mon_by_importance:
        dom_idx = mon_list.index(a)
        if mon_n_appearances[dom_idx] == 0:
            break
        # If this is an evolution and the base form isn't mapped, map the base instead.
        if a in mon_evolution_r:
            base = a
            while base in mon_evolution_r:
                base = mon_evolution_r[base]
            if base not in mon_map:
                # print("Was going to assign %s, but assigning base %s instead" % (a, base))  # DEBUG
                a = base
                dom_idx = mon_list.index(a)
        if a in mon_evolution:  # need to run beam search rather than taking the single best.
            k = assignment_beam
        else:
            k = 1
        best_jdx = None
        lowest_penalty = None
        kidxs = np.argpartition(a_to_b_penalty[dom_idx], k)[:k]  # get k min jdxs
        for kidx in kidxs:
            if a_to_b_penalty[dom_idx, kidx] == float("inf"):
                break
            ab_p_copy = a_to_b_penalty.copy()
            image_copy = mon_image.copy()
            mon_map_copy = {m: mon_map[m] for m in mon_map}
            kidx_p = make_assignment_and_propagate(ab_p_copy, dom_idx, kidx, type_map,
                                                   mon_map_copy, image_copy, mon_list,
                                                   mon_evolution, mon_metadata, penalize_same_type)
            if best_jdx is None or kidx_p < lowest_penalty:
                best_jdx = kidx
                lowest_penalty = kidx_p
        if best_jdx is not None:
            make_assignment_and_propagate(a_to_b_penalty, dom_idx, best_jdx, type_map,
                                          mon_map, mon_image, mon_list, mon_evolution, mon_metadata,
                                          penalize_same_type, debug=True)
    print("...... done")

    # For remainder, greedily select the current least penalty assignment 
    # and make it until finished.
    print("... Now assigning remainder based on minimizing penalty only...")
    all_assigned = len(mon_map.keys()) == len(mon_list)
    while not all_assigned:
        min_idx_jdx = np.argmin(a_to_b_penalty)
        dom_idx, img_jdx = int(min_idx_jdx // n_mon), int(min_idx_jdx % n_mon)
        make_assignment_and_propagate(a_to_b_penalty, dom_idx, img_jdx, type_map,
                                      mon_map, mon_image, mon_list, mon_evolution, mon_metadata,
                                      penalize_same_type)
        all_assigned = len(mon_map.keys()) == len(mon_list)

    curr_penalty, curr_mon_penalty, type2type = score_map(mon_map, mon_list, type_list,
                                                          mon_metadata, mon_evolution,
                                                          penalize_same_type)
    print("...... done")
    print("... done; init penalty %.2f" % curr_penalty)

    # Perform simulated annealing.
    if args.tune_w_sa == 1:
        print("Running simulated annealing to tune the solution...")
        for i in range(max_iterations):
            t = math.log(float(max_iterations) / (i + 1)) + 1
            if i % print_every_n_iterations == 0:
                print("%d/%d; t %.5f; penalty %.2f" % (i, max_iterations, t, curr_penalty))
                with open("%s.%d" % (args.output_fn, i), 'w') as f:
                    f.write(score_map(mon_map, mon_list, type_list, mon_metadata, mon_evolution,
                                      penalize_same_type, as_exp_str=True))

            # Create a nearest neighbor assignment.
            neighbor = {a: mon_map[a] for a in mon_map}
            for _ in range(swaps_for_neighbor):
                # select (a -> b & c -> d) to create (a -> d & c -> b)
                a, c = random.choices(mon_list, weights=curr_mon_penalty, k=2)
                b, d = neighbor[a], neighbor[c]
                neighbor[a] = d
                neighbor[c] = b
            
            # Score the neighbor and accept it based on temp.
            neighbor_penalty, n_mon_penalty, n_type2type = score_map(neighbor, mon_list, type_list,
                                                                     mon_metadata, mon_evolution,
                                                                     penalize_same_type)
            if (neighbor_penalty < curr_penalty or 
                random.random() < math.exp((curr_penalty - neighbor_penalty) / t)):
                mon_map = neighbor
                curr_penalty = neighbor_penalty
                curr_mon_penalty = n_mon_penalty
                type2type = n_type2type
        print("... done; final penalty %.2f" % curr_penalty)

    # Write the result.
    print("Writing results to JSON format...")
    d = {"mon_map": mon_map, "type_map": type2type}
    with open(args.output_fn, 'w') as f:
        json.dump(d, f, indent=2)
    print("... done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Write an A to B mon map.')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output file for the mon map')
    parser.add_argument('--type_mapping', type=str, required=True,
                        help='"random", "fixed", or a partial list with - and ,')
    parser.add_argument('--tune_w_sa', type=int, required=True,
                        help='whether to tune with simulated annealing after analytical solution')
    args = parser.parse_args()


    main(args)
