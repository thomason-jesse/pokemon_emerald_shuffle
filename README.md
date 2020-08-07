# Pokémon Emerald Neural Shuffle

This is an extension of a decompilation of Pokémon Emerald that adds tools for shuffling and generating entirely new Pokémon in the ROM.

I played Pokémon as a kid, and revisiting it via emulators has been interesting, mostly because the game is sort of very easy and predictable now. I took some steps to make that less true, and in the process built up large tool pipeline to shuffle and remix the games, all the way up to generating entirely new Pokémon using deep autoencoders.

## Get Started

We build on the decompilation project [here](https://github.com/pret/pokeemerald). Follow the installation and build instructions there before using these tools.

The other changes I have made in this repo do things like giving a national Pokédex at the beginning of the game, giving all the special island boat tickets after beating the Elite Four, and other similar measures to make the ROM stand alone with all its content.

## Shuffle Existing Pokémon

What if GYM leaders used different types? Use this tool to create a mapping from original types to target types throughout the ROM, enabling, for example, ROCK->GHOST to make a spooky ROXANNE. Unlike a completely random shuffling tool, this script attempts to align Pokémon my stat total and the desired type map while preserving evolution lines.

First, we run a script that gathers relevant Pokémon metadata from the ROM source.

```
cd python_scripts/
mkdir metadata/
python create_resources.py --output_fn metadata/mon.json
```

Create a directory to host your map from mon to mon.

```
mkdir mon_maps/
```

Now, for our example of making ROXANNE spooky, we'd run

```
python create_a_to_b_mapping.py \
 --input_fn_source metadata/mon.json \
 --input_fn_target metadata/mon.json \
 --output_fn mon_maps/spooky_rocks.json \
 --type_mapping random \
 --partial_type_map TYPE_ROCK-TYPE_GHOST
```

This command generates a one-to-one mapping from Pokémon to Pokémon, trying to assign ROCK types to similar GHOST types. Other type to type mappings are chosen randomly. Check out `mon_maps/spooky_rocks.json` to see the resulting map, which might for example take GEODUDE, GRAVELER, and GOLEM to GASTLY, HAUNTER, and GENGAR.

You can specify the entire type to type map by chaining your `--partial_type_map` with commas, like `TYPE_ROCK-TYPE_GHOST,TYPE_PSYCHIC-TYPE_FIGHTING`.

These type maps are reflected in GYMs and ELITE FOUR flavor-text, so you can tell what type a leader will use. The gift TMs from leaders also reflect the type maps, so spooky ROXANNE will gift SHADOW BALL.

If you'd rather not shuffle types, but just want a fresh experience, creating a map with `--type_mapping fixed` preserves types but assigns new Pokémon, effectively mixing a lot of Gen I and Gen II into the Hoenn region.

The `--tune_w_sa` option runs Simulated Annealing after performing an initial, greedy algorithm assignment to try to smooth out the mapping. It takes longer to run but generally seems to create more coherent maps.

Given your mon map, you can augment it with a move map.

```
python create_move_nn.py \
 --input_fn mon_maps/spooky_rocks.json \
 --input_metadata_fn metadata/mon.json \
 --output_fn mon_maps/spooky_rocks.moves.json
```

This resource is for replacing trainer parties in the ROM where move lists are specified. For each move each mon can learn, we find the moves nearest neighbor (based on effects, type, etc.) that can be learned by its destination. For example, ROXANE's GEODUDE knows ROCK THROW, which GASTLY can't learn, so we map ROCK THROW to NIGHT SHADE for GASTLY.

Now that you have your map, you can use it to rewrite ROM data.

```
python a_to_b_mon_replacement.py \
 --input_fn mon_maps/spooky_rocks.moves.json \
 --lvl_increase 0.1
```

This script overwrites wild Pokémon and trainer parties, sets new gift TMs, sets new gift Pokémon, and changes some flavortext in Gyms and the Elite Four to reflect the type mapping. The optional `--lvl_increase` parameter increases the level of trainer party Pokémon by this fraction, which I have found is helpful to make the game a bit tougher and counter-balance less finely tuned movesets in leaders.

![Shuffled game battle with ELECTRIKE and SQUIRTLE versus QUILAVA and CHARMELEON](/python_scripts/screenshots/shuffle.png)

These shuffled games have an edge over full randomizers that can result in ROMs that are sort of goofy and unplayable, but they don't require someone to sit down and hand-craft the entire thing to have a mostly new game to play.

## Generate Entirely New Pokémon

The problem with any kind of shuffling of an existing ROM is that I know all these Pokémon already. To replicate the nostalgia I am after, I needed to encounter Pokémon I had never seen before, taking us back to guessing at types and being surprised by movesets.

![A batle between two generated Pokémon, BURNINGMON and KILLMON](/python_scripts/screenshots/gen_battle.png)

![A party of six generated Pokémon](/python_scripts/screenshots/gen_party.png)

First, we train a neural autoencoder model to learn a low dimensional manifold that represents stats, move learnsets, types, abilities, and other Pokémon metadata. Once we have that learned network, we can sample entirely new Pokémon, and use the mapping tool above to insert them into the ROM in a coherent way that preserves stat totals and desired type maps.

First, we need to vectorize the metadata so it can be fed easily into our neural networks.

```
python prep_mon_network_inputs.py \
 --input_fn metadata/mon.json \
 --output_fn metadata/processed_h_128.json
```

Now, we can train a neural, variational autoencoder model from which we can later sample new Pokémon.

```
mkdir ae_models/
python train_mon_network.py \
 --input_fn metadata/processed_h_128.json \
 --output_fn ae_models/ae
```

Increase the number of `epochs` to train the model longer by modifying the sourcecode, and continue training the best model by providing the additional flag `--input_model_pref ae_models/ae.best` The best-performing model will live at `ae_models/ae.best.model`. These things are true for the evolution network training as well.

Given the learned embedding space from the autoencoder, we can now train a model that takes in a sampled Pokémon and produces an evolution embedding, as well as the nature of the evolution.

```
mkdir ev_models/
python train_evo_network.py \
 --input_meta_fn metadata/processed_h_128.json \
 --input_model_fn ae_models/ae.best.model \
 --output_fn ev_models/ev
```

Now that we can sample both base mon and evolve them, we can sample an entirely new Pokémon population.

```
mkdir sampled_mon/
python sample_new_mon.py \
 --input_meta_mon_fn metadata/mon.json \
 --input_meta_network_fn metadata/processed_h_128.json \
 --input_mon_model_fn ae_models/ae.best.model \
 --input_evo_model evo_models/ev.best.model \
 --output_fn sampled_mon/new_mon.json \
 --name_by_type_word metadata/type_word_nns.json
```

![The splash screen for generated BURNINGMON](/python_scripts/screenshots/gen_starter.png)

Now we rewrite the ROM with the newly generated metadata, essentially overwriting existing Pokémon in ROM memory with these newly sampled ones. One fun thing this script also does is make the Pokédex show something useful instead of just flavor text, letting you know the original Pokémon with the most similar base stats, how this sampled Pokémon evolves, and what moves it learns by leveling up.

```
python replace_rom_mon_data.py --input_fn sampled_mon/new_mon.json
```

![The Pokédex entry for generated BURNINGMON](/python_scripts/screenshots/gen_pokedex_starter.png)

![A party of six generated Pokémon](/python_scripts/screenshots/gen_party.png)

Now that this new Pokémon metadata exists, we can also use the existing mapping scripts to generate a coherent ROM. Otherwise, all the new Pokémon are randomly assigned and will be hanging out in inappropriate places in the game.

```
python create_a_to_b_mapping.py \
 --input_fn_source metadata/mon.json \
 --input_fn_target metadata/new_mon.json \
 --output_fn mon_maps/sampled.json \
 --type_mapping fixed \
 --tune_w_sa
```

By setting `--type_mapping fixed`, we'll keep the existing type themes in the game, with Team Magma using FIRE and GROUND types, etc. Optionally, we could set `--type_mapping optimal`, which greedily attempts to form a type map that corresponds to the distribution of types in the sampled Pokémon. For example, if we sample many WATER types, it would be better to map NORMAL to WATER than NORMAL to NORMAL, since there are more NORMAL types than anything else in the original ROM.

Like above, we need to also create a good move map so hand-crafted movesets are preserved in power.

```
python create_move_nn.py \
 --input_fn mon_maps/sampled.json \
 --input_metadata_fn metadata/new_mon.json \
 --output_fn mon_maps/sampled.moves.json
```

And finally, we swap appearances in the ROM based on this map.

```
python a_to_b_mon_replacement.py \
 --input_fn mon_maps/sampled.moves.json \
 --lvl_increase 0.1
```

That's that! A whole new population of Pokémon to find, learn about, and try to catch.

## Clear Room for Improvement

Putting all these tools together was A Lot, but there's a few clear-cut places where they could be made better.

### Generating Names

Name generation is currently just some GloVe nearest neighbors for each type, generated by `create_type_word_nns_from_glove.py` with logic to assign to Pokémon in `sample_new_mon.py`. Names are tricky because they don't really care any semantic information about the Pokémon, so generating them from the neural manifold doesn't make much sense to me. Still, doing something more clever with WordNet or something might be fun.

### Generating Sprites

![Generated LAWNMON and DRAINMON fight with horrible generated sprites](/python_scripts/screenshots/horrible_sprites.png)

So I actually did this! The `train_sprite_network.py` script contains my half-baked modelling attempts to make front, back, and icon sprites. I never got this working well enough to be remotely passable, so aa better network would be great, and probably actually working with the 4 bit colormap images instead of converting them to / from full 8-bit RGB might make the process easier. Instead, what `sample_new_mon.py` does currently is assign a base sprite to a generated mon based on its nearest neighbor in the learned embedding space, and the color palette for whatever mon it's overwriting in the ROM is adopted. So, essentially, they're sprites of original Pokémon that are somehow similar to the generated ones, with random coloration. That could just, like, obviously be done better.

### Better learned representations and evolutions

The `sample_new_mon.py` script really does a lot of the heavy lifting to make sampled Pokémon and their evolutions coherent. For example, it tries to ensure consistency with learned TM/HM moves, so that evolved forms don't lose the ability to learn things. It preserves at least one type when mon evolve, and ensures their base stats and EVs don't decrease. It forces type changes when things like WATER STONE are used for evolution. And, like, all of that wouldn't be necessary if the evolution network did a better job. Probably losses for evolution should be cooked into the actual training manifold, such that sampling new mon and evolving them happen simultaneously.

Also, I'm still not happy about the levelup and TMHM movesets that end up being learned. While `sample_new_mon.py` tries hard to at least make sure mon have a sensible number of moves, there's not always a clear correspondence between learned moves and a sampled Pokémon's other themes like type and ability. It might be that the embedding dimension for the autoencoder is actually too small, or that the network overfits too much. Who knows; not me.

### More complete ROM rewriting

There are just some pieces of the ROM that would feel better if they were replaced but they're either tedious to track down or I lost motivation. For example, NORMAN gives WALLY ZIGZAGOON (or whatever now lives in its memory slot) instead of what ZIGZAGOON has been mapped to. And it would be nice if flavortext that mentions mon by name used thier new, sampled names instead. Plus, the Pokédex still uses the original memory of mon nicknames, heights, footprints and sprite placement and animations, all of which could be replaced or learned as part of the autoencoder.
