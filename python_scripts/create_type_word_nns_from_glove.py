import argparse
import json
import numpy as np


def main(args):
    disallow_list = ['color', 'white', 'black', 'red', 'blue', 'green', 'yellow', 'purple', 'pink', 'brown']

    with open(args.input_fn, 'r') as f:
        d = json.load(f)
        type_list = d['type_list']
        n_mon = len(d['mon_metadata'])
    t_ws = [t[len('TYPE_'):].lower() for t in type_list]

    print("Loading GloVe embeddings from '%s'..." % args.glove_infile)
    glove_ws = []
    glove_vs = []
    with open(args.glove_infile, 'r') as f:
        for line in f.readlines():
            splitLines = line.split()
            word = splitLines[0]
            if len(word) <= 7 or str(word) in t_ws:
                wordEmbedding = np.array([float(value) for value in splitLines[1:]])
                glove_ws.append(str(word))
                glove_vs.append(wordEmbedding)
    print("... done; read %d word embeddings" % len(glove_ws))

    print("Calculating nearest %d neighbors for each type..." % n_mon)
    t_nns = {}
    for t in type_list:
        print("... for %s" % t)
        t_base = t[len('TYPE_'):].lower()
        v = glove_vs[glove_ws.index(t_base)]
        glove_sim = np.zeros(len(glove_ws))
        for idx in range(len(glove_ws)):
            w = glove_vs[idx]
            glove_sim[idx] = np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w))
            if (np.any([str(t_w) in str(glove_ws[idx]) for t_w in t_ws]) or
                np.any([str(t_w) in str(glove_ws[idx]) for t_w in disallow_list])):
                glove_sim[idx] = 0
            if str(glove_ws[idx])[-1] == 's' and str(glove_ws[idx])[:-1] in glove_ws:
                glove_sim[idx] = 0
        t_nns[t] = [glove_ws[idx] for idx in np.argsort(glove_sim)[::-1][:n_mon]]
        print("..... done; ")
    print("... done")

    print("Writing to file '%s'" % args.output_fn)
    with open(args.output_fn, 'w') as f:
        json.dump(t_nns, f)
    print("... done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rewrite wild and trainer mon.')
    parser.add_argument('--input_fn', type=str, required=True,
                        help='the input json file with orig mon metadata')
    parser.add_argument('--glove_infile', type=str, required=True,
                        help='the input glove text file')
    parser.add_argument('--output_fn', type=str, required=True,
                        help='the output json file')
    args = parser.parse_args()

    main(args)
