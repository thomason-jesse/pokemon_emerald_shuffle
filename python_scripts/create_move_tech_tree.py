import argparse
import json
import numpy as np


def main(args):

    # Read in mon movesets.
    with open(args.input_metadata_fn, 'r') as f:
        d = json.load(f)
        mon_metadata = d["mon_metadata"]
        mon_levelup_moveset = d["mon_levelup_moveset"]
        mon_tmhm_moveset = d["mon_tmhm_moveset"]

    moves = set()
    levelup_move_freq = {}
    levelup_move_levels = {}
    for a in mon_metadata:
        for lvl, move in mon_levelup_moveset[a]:
            moves.add(move)
            if move not in levelup_move_freq:
                levelup_move_freq[move] = 0
                levelup_move_levels[move] = []
            levelup_move_freq[move] += 1
            levelup_move_levels[move].append(lvl)
        for move in mon_tmhm_moveset[a]:
            moves.add(move + "+TMHM")
    print('Read in %d mon and indexed %d moves' % (len(mon_metadata), len(moves)))

    # For every move, build prereq list based on levelup moveset.
    move_tech_tree = {}
    for m in moves:
        mon_prereq_lvls = []
        mon_prereq_moves = []

        # First, build the prereq lists for each mon who has this move.
        for a in mon_metadata:
            pre_lvls = []
            pre_moves = []
            found_at = None
            for lvl, b in mon_levelup_moveset[a]:
                if b == m and found_at is None:
                    found_at = lvl
                if found_at is not None and lvl > found_at:
                    break
                pre_lvls.append(lvl)
                pre_moves.append(b)
            n_from_levelup = len(pre_lvls)
            for b in mon_tmhm_moveset[a]:
                if b + '+TMHM' == m:
                    found_at = 100
                    for idx in range(n_from_levelup):
                        pre_lvls.append(100)
                        pre_moves.append(pre_moves[idx])

            if found_at is not None and len(pre_moves) > 1:
                pre_lvls = [found_at - lvl for lvl in pre_lvls]
                mon_prereq_lvls.append(pre_lvls)
                mon_prereq_moves.append(pre_moves)

        # Check for intersections.
        chosen_prereqs = []
        remaining = set(range(len(mon_prereq_lvls)))
        while len(remaining) > 0:
            sm = {}
            for idx in remaining:
                for jdx in range(len(mon_prereq_lvls[idx])):
                    if m != mon_prereq_moves[idx][jdx]:
                        if mon_prereq_moves[idx][jdx] not in sm:
                            sm[mon_prereq_moves[idx][jdx]] = set()
                        sm[mon_prereq_moves[idx][jdx]].add(idx)
            ssm = [(k, v) for k, v in
                   sorted(sm.items(),
                          key=lambda item: len(item[1]) * len(mon_metadata) - levelup_move_freq[item[0]],
                          reverse=True)]
            chosen = ssm[0]
            chosen_prereqs.append((chosen[0],
                                   [mon_prereq_lvls[i][mon_prereq_moves[i].index(chosen[0])] for i in chosen[1]]))
            remaining = remaining - chosen[1]

        # Summary stats of the tech tree for the move.
        move_tech_tree[m] = []
        for b, lvls in chosen_prereqs:
            prob = len(lvls) / sum([len(_lvls) for _b, _lvls in chosen_prereqs])
            mu = np.average(lvls)
            std = np.std(lvls)
            move_tech_tree[m].append([b, prob, mu, std])

    # Randomly sample some movesets conditioned on target move to get depths.
    depths_per_move = {}
    for m in levelup_move_levels:
        depths_per_move[m] = []
        for _ in range(10000):
            mu = np.average(levelup_move_levels[m])
            std = np.std(levelup_move_levels[m])
            learnset = {m: max(0, min(100, int(mu + np.random.normal() * std + 0.5)))}
            valid_move = True
            learned_zero = learnset[m] == 0
            last_move = m
            while valid_move and not learned_zero:
                p = np.random.random()
                ps = 0
                next_move = next_mu = next_std = None
                valid_move = False
                for b, prob, mu, std in move_tech_tree[last_move]:
                    if b not in learnset:
                        valid_move = True
                    if p < prob + ps:
                        next_move = b
                        next_mu = mu
                        next_std = std
                        break
                    ps += prob
                if next_move is not None:
                    learnset[next_move] = max(0,
                                              min(learnset[last_move],
                                                  learnset[last_move] - int(next_mu + np.random.normal() * next_std + 0.5)))
                    last_move = next_move
                    learned_zero = learnset[next_move] == 0
            depths_per_move[m].append(len(learnset))
    depths_per_move = {m: np.average(depths_per_move[m]) for m in depths_per_move}
    print('\n'.join([str((k, np.average(v)))
                     for k, v in sorted(depths_per_move.items(), key=lambda item: item[1],
                                        reverse=False)]))

    with open(args.output_fn, 'w') as f:
        d = {'depths_per_move': depths_per_move,
             'levelup_move_levels': levelup_move_levels,
             'learnset_move_tech_tree': {m: move_tech_tree[m] for m in move_tech_tree if '+TMHM' not in m},
             'tmhm_move_tech_tree': {m[:-len('+TMHM')]: move_tech_tree[m] for m in move_tech_tree if '+TMHM' in m},
             }
        json.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates tech tree structure out of moves")
    parser.add_argument('--input_metadata_fn', type=str, required=True,
                        help='the input JSON of mon metadata')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output JSON for move tech trees and depths')
    args = parser.parse_args()

    main(args)
