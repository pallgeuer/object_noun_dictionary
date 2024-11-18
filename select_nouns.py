#!/usr/bin/env python3

# Imports
import os
import re
import csv
import sys
import math
import json
import copy
import argparse
import itertools
import contextlib
import collections
import dataclasses
from typing import Optional
import inflect
from nltk.stem import WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from nltk.corpus import wordnet31 as wordnet
from nltk.corpus.reader.wordnet import WordNetError

# Notes:
#  - We don't try to exclude proper nouns by checking for capitalisation because many genuses and such are capitalised and shouldn't be filtered out
#  - We don't use (non-plural) derivative forms of nouns as they are mostly not nouns anyway, and if they are, then they are often of an unwanted classification/category
#  - True for all 82192 noun synsets: re.sub(r'\.n\.[^.]+$', r'', synset.name()) == synset.lemma_names()[0].lower() => Stripping synset name is always lower case and matches first lemma name (has underscores, some have '.' in it)
#  - All 82192 noun synsets only use the characters: _'-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz (at last check the datasets also only uses these characters)
#  - WordNet lemmas already include British and/or alternative spellings some of the time
#  - StarDict dict.org keys have the characters: " #'-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Constants
ROOT_SYNSET = 'entity.n.01'
PLURAL_NOUNS_EXC = 'noun.exc'
MULT_SCALE = 1.5
FREQ_THRESHOLD = 2.5e-6
CLEMMA_FREQ_PRINT = 0.015
ALLOWED_CHARS_CASED = set(" '-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
ALLOWED_CHARS_SAFE = set(" -ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
ALLOWED_CHARS_CANON = set(" 0123456789abcdefghijklmnopqrstuvwxyz")
ALLOWED_CHARS_STARDICT = set(" -'abcdefghijklmnopqrstuvwxyz")
NOUN_LEXNAMES = {
	'noun.Tops',
	'noun.act',
	'noun.animal',
	'noun.artifact',
	'noun.attribute',
	'noun.body',
	'noun.cognition',
	'noun.communication',
	'noun.event',
	'noun.feeling',
	'noun.food',
	'noun.group',
	'noun.location',
	'noun.motive',
	'noun.object',
	'noun.person',
	'noun.phenomenon',
	'noun.plant',
	'noun.possession',
	'noun.process',
	'noun.quantity',
	'noun.relation',
	'noun.shape',
	'noun.state',
	'noun.substance',
	'noun.time',
}

# Main function
def main(argv):

	parser = argparse.ArgumentParser(description='Select nouns from WordNet.', add_help=False)
	parser.add_argument('-h', '--help', action='help', help='Show this help message and exit')
	parser.add_argument('--select', action='store_true', help='Recurse through the WordNet hierarchy and interactively select noun synset subtrees')
	parser.add_argument('--select_outfile', type=str, help='Save the selected nouns to an output text file')
	parser.add_argument('--select_root', type=str, default=ROOT_SYNSET, help='Root synset to select nouns from')
	parser.add_argument('--select_plurals', action='store_true', help='Generate the plural forms of the selected nouns as well')
	parser.add_argument('--select_lexnames', type=str, nargs='+', help='Lexical noun classes to limit the selection to (e.g. artifact, food, time)')
	parser.add_argument('--select_freqs', type=str, help='Load word frequencies file and include them in the selected output')
	parser.add_argument('--lextree', type=str, nargs='+', help='Show a tree of all synsets in the given lexical noun classes (e.g. animal, person, object)')
	parser.add_argument('--lextree_outfile', type=str, help='Save the tree to an output text file')
	parser.add_argument('--lextree_root', type=str, default=ROOT_SYNSET, help='Root synset to show a tree for')
	parser.add_argument('--synset_spec', type=str, nargs=2, help='Load a synset spec and save the corresponding synset list (tags: LEINY)')
	parser.add_argument('--curate', type=str, help='Curate nouns from a synset list')
	parser.add_argument('--curate_outfile', type=str, help='Save the curated nouns to an output CSV file')
	parser.add_argument('--curate_manual', type=str, help='File containing manual nouns to initialise lemma map with')
	parser.add_argument('--curate_alternates', type=str, help='Load StarDict dict.org noun alternates from a TSV file')
	parser.add_argument('--curate_spellings', type=str, nargs=2, metavar=('AMKEY', 'BRKEY'), help='Load American-British spellings as JSON dicts (may include plural words)')
	parser.add_argument('--curate_objects', type=str, help='Directory to load object classes from')
	parser.add_argument('--curate_general', type=str, help='Directory to load general classes from')
	parser.add_argument('--curate_freqs', type=str, help='Load word frequencies TSV file')
	args = parser.parse_args(argv)

	print()

	if args.lextree:
		action_lextree(outfile=args.lextree_outfile, root=args.lextree_root, lexnames=args.lextree)
	if args.select:
		action_select(outfile=args.select_outfile, root=args.select_root, lexnames=args.select_lexnames, plurals=args.select_plurals, freqs=args.select_freqs)
	if args.synset_spec:
		action_synset_spec(spec_file=args.synset_spec[0], list_file=args.synset_spec[1])
	if args.curate:
		action_curate(outfile=args.curate_outfile, lemmas_file=args.curate_manual, list_file=args.curate, alternates=args.curate_alternates, spellings=args.curate_spellings, objects=args.curate_objects, general=args.curate_general, freqs=args.curate_freqs)

#
# Lextree
#

# Action: Show a synset tree filtered by lexnames
def action_lextree(outfile, root, lexnames):

	load_wordnet()

	print(f"Showing WordNet hierarchy tree (with repeats) starting at: {root}")
	if lexnames:
		lexnames = {'noun.' + lexname for lexname in lexnames}
		assert not (lexnames - NOUN_LEXNAMES), format(lexnames - NOUN_LEXNAMES)
		print(f"Allowed lexnames: {', '.join(lexnames)}")
	root_synset = wordnet.synset(root)
	assert root_synset.pos() == wordnet.NOUN

	if outfile:
		print(f"Generating and writing output file: {outfile}")
	with contextlib.ExitStack() as stack:

		if outfile in (None, '', 'stdout', 'sys.stdout'):
			file = None
			print('-' * 120, file=file)
		else:
			file = stack.enter_context(open(outfile, 'w'))

		show_lextree(root_synset, lexnames=lexnames, file=file)

		if file is None:
			print('-' * 120, file=file)

	if outfile:
		print("Done")
	print()

# Worker: Show a synset tree filtered by lexnames
def show_lextree(synset, lexnames, file, pending=None, level=0):

	if pending is None:
		pending = [(level, synset)]

	if synset.lexname() in lexnames:
		for pending_level, pending_synset in pending:
			pending_lexname = pending_synset.lexname()
			color = 32 if pending_lexname in lexnames else 31
			other_lexnames = '|'.join(sorted({lexname.replace('noun.', '') for syn in wordnet.synsets(re.sub(r'\.n\.[^.]+$', r'', pending_synset.name()), pos=wordnet.NOUN) if (lexname := syn.lexname()) != pending_lexname}))
			if other_lexnames:
				other_lexnames = f" \033[33m[{other_lexnames}]\033[0m"
			print(f"{'|   ' * pending_level}[{pending_lexname.replace('noun.', '')}] \033[{color}m{pending_synset.name()}\033[0m:{other_lexnames} {pending_synset.definition().capitalize()}", file=file)
		pending.clear()

	hyponyms = synset.hyponyms() + synset.instance_hyponyms()
	for hypo in hyponyms:
		pending.append((level + 1, hypo))
		show_lextree(hypo, lexnames, file, pending=pending, level=level + 1)
		if pending:
			pending.pop()

#
# Select
#

# Action: Recursively allow user to interactively select which synsets to include
# noinspection PyUnusedLocal, PyUnreachableCode
def action_select(outfile, root, lexnames, plurals, freqs):

	raise NotImplementedError("Gram freqs has changed => Need to update this code if you want to use this function again!")

	load_wordnet()
	plurals_map, singulars_map = load_plurals_map()
	inflect_engine = inflect.engine()
	gram_freqs = load_gram_freqs(freqs) if freqs else None

	print(f"Selecting nouns from WordNet hierarchy (without repeats) starting at: {root}")
	if lexnames:
		lexnames = {'noun.' + lexname for lexname in lexnames}
		assert not (lexnames - NOUN_LEXNAMES), format(lexnames - NOUN_LEXNAMES)
		print(f"Allowed lexnames: {', '.join(lexnames)}")
	root_synset = wordnet.synset(root)
	assert root_synset.pos() == wordnet.NOUN
	print("c  = Show a list of child synsets then ask again (cc = depth 2, ccc = depth 3, etc)")
	print("i  = Include this synset but interactively offer options for all child synsets")
	print("e  = Exclude this synset but interactively offer options for all child synsets")
	print("y  = Recursively include this synset and all its descendants")
	print("yy = Do this for all remaining at this level as well")
	print("n  = Recursively exclude this synset and all its descendants")
	print("nn = Do this for all remaining at this level as well")
	visited, selected, _ = select_synsets(root_synset)
	if lexnames:
		selected = {synset for synset in selected if synset.lexname() in lexnames}
	selected_nouns = {noun for synset in selected for noun in synset.lemma_names()}
	print()

	print(f"Visited {len(visited)} unique noun synsets")
	print(f"Selected {len(selected)} unique noun synsets")
	print(f"Selected {len(selected_nouns)} unique noun lemmas")
	print()

	if outfile:
		print(f"Generating and writing output file: {outfile}")
	with contextlib.ExitStack() as stack:

		if outfile in (None, '', 'stdout', 'sys.stdout'):
			file = None
			print('-' * 120, file=file)
		else:
			file = stack.enter_context(open(outfile, 'w'))

		rows = []
		for noun in selected_nouns:
			assert ',' not in noun
			noun = noun.replace('U._S.', 'U.S.')
			spaced_noun = noun.replace('_', ' ').strip()
			row = [0, spaced_noun]
			if re.fullmatch(r'([A-Z]\.){2,}', spaced_noun):
				row.insert(1, spaced_noun.replace('.', ''))
			else:
				row.append('')
			if plurals:
				if noun in plurals_map:
					row.extend(plurals_map[spaced_noun])
				elif spaced_noun.islower() and all((c.isascii() and c.isalpha()) or c in ' -' for c in spaced_noun):
					row.append(inflect_engine.plural_noun(spaced_noun))
			if gram_freqs:
				grams = {gram.lower() for gram in row[1:] if gram}
				row[0] = sum(gram_freqs.get(gram, 0) for gram in grams)
			rows.append(row)

		rows.sort(key=lambda r: (-r[0], r[1].lower()))
		for row in rows:
			print(f"{'0' if row[0] < 1 else format(math.log(row[0]) - 8, '.6f')},{','.join(row[1:])}", file=file)  # Format: FREQ,NOUN,ALTNOUN[,PLURAL[,PLURAL[...]]] where FREQ is 0 if unknown and ALTNOUN may be empty string

		if file is None:
			print('-' * 120, file=file)

	if outfile:
		print("Done")
	print()

# Worker: Recursively allow user to interactively select which synsets to include
def select_synsets(synset, visited=None, selected=None, level=0, response=None, prefix=''):

	if selected is None:
		selected = set()

	if visited is None:
		visited = {synset}
	elif synset in visited:
		return visited, selected, None
	else:
		visited.add(synset)

	hyponyms = synset.hyponyms() + synset.instance_hyponyms()

	while True:

		asked_response = False
		if response is None:
			response = input(f"{prefix}[{level}] SELECT {synset.name()}: {synset.definition().capitalize()} (c/i/e/y/yy/n/nn)? ").lower()
			asked_response = True

		if response and all(c == 'c' for c in response):
			for hypo in hyponyms:
				print(f"    {prefix}[{level + 1}] CHILD {hypo.name()}: {hypo.definition().capitalize()}")
				if len(response) > 1:
					select_synsets(hypo, visited=None, selected=None, level=level + 1, response=response[1:], prefix=prefix + '    ')
			if not asked_response:
				return visited, selected, None
		elif response in ('y', 'yy', 'n', 'nn', 'i', 'e'):
			if response[0] in ('y', 'i'):
				selected.add(synset)
			for hypo in hyponyms:
				_, _, further_response = select_synsets(hypo, visited=visited, selected=selected, level=level + 1, response=None if response in ('i', 'e') else response[0], prefix=prefix)
				if further_response is not None:
					response = further_response
			return visited, selected, response[0] if len(response) > 1 else None

		response = None

#
# Synset spec
#

# Action: Process synset spec to list
def action_synset_spec(spec_file, list_file):

	load_wordnet()

	print(f"Input synset spec:  {spec_file}")
	print(f"Output synset list: {list_file}")
	print()

	print("Loading synset spec...")
	synset_spec = dict(L=set(), E=set(), I=set(), N={ROOT_SYNSET}, Y=set())
	with open(spec_file, 'r') as file:
		for line in file:
			try:
				tags, synset_name = line.split()
				for tag in tags:
					synset_spec[tag.upper()].add(synset_name)
			except (ValueError, KeyError):
				raise ValueError(f"Invalid line in spec: {line}")
	print(f"Loaded {len(synset_spec['L'])} lexnames")
	print(f"Loaded {len(synset_spec['E'])} singular excludes")
	print(f"Loaded {len(synset_spec['I'])} singular includes")
	print(f"Loaded {len(synset_spec['N'])} recursive excludes")
	print(f"Loaded {len(synset_spec['Y'])} recursive includes")
	for synset_name in set.union(*(synset_set for tag, synset_set in synset_spec.items() if tag != 'L')):
		try:
			synset = wordnet.synset(synset_name)
		except (ValueError, WordNetError):
			raise ValueError(f"Invalid synset name: {synset_name}")
		assert synset.name() == synset_name
	print()

	root_synset = wordnet.synset(ROOT_SYNSET)
	assert root_synset.pos() == wordnet.NOUN
	lexnames = {'noun.' + lexname for lexname in synset_spec['L']}
	assert not (lexnames - NOUN_LEXNAMES), format(lexnames - NOUN_LEXNAMES)
	synset_names = collect_synsets(synset_spec, root_synset, lexnames=lexnames)

	print("Saving synset list...")
	with open(list_file, 'w') as file:
		for synset_name in sorted(synset_names):
			file.write(synset_name + '\n')
	print(f"Saved {len(synset_names)} synsets")
	print()

# Worker: Process synset spec to list
def collect_synsets(spec, synset, lexnames, synset_names=None, include=False):

	if synset_names is None:
		synset_names = set()

	synset_name = synset.name()
	synset_lexname = synset.lexname()

	include_synset = include
	if synset_name in spec['N']:
		include_synset = False
		include = False
	if synset_name in spec['Y']:
		include_synset = True
		include = True
	if synset_name in spec['I']:
		include_synset = True
	if synset_name in spec['E']:
		include_synset = False
	if synset_lexname not in lexnames:
		include_synset = False

	if include_synset:
		synset_names.add(synset_name)

	hyponyms = synset.hyponyms() + synset.instance_hyponyms()
	for hypo in hyponyms:
		collect_synsets(spec, hypo, lexnames, synset_names=synset_names, include=include)

	return synset_names

#
# Curate
#

# Sub-canonical data class
@dataclasses.dataclass(eq=False)
class SubCanon:

	canon: str              # Canonical form
	canon_freq: float       # Word frequency of the canonical form
	variants: set[str]      # All seen variants that have this canonical form
	singular: bool          # Whether this canonical form is singular (can be both)
	plural: bool            # Whether this canonical form is plural (can be both)
	count: int = 1          # Number of selected CLemmas this canonical form has appeared in
	lemma_freq: float = -1  # Word frequency contribution when this subcanonical is part of a lemma

	def validate(self):
		assert self.canon in self.variants, f"Variants does not include canon: {self}"
		assert all(char in ALLOWED_CHARS_CANON for char in self.canon), f"Invalid canon chars: {self}"
		assert 0 <= self.canon_freq <= 1, f"Canon frequency out of range: {self.canon_freq}"
		assert all(variant for variant in self.variants), f"Empty variant: {self}"
		assert all(char in ALLOWED_CHARS_CASED for variant in self.variants for char in variant), f"Invalid variant chars: {self}"
		assert all(get_canon(variant) == self.canon for variant in self.variants), f"Variant does not have correct canon: {self}"
		assert self.singular or self.plural, f"Cannot be neither singular nor plural: {self}"
		assert self.count >= 1, f"Invalid count: {self}"
		assert self.lemma_freq < 0 or FREQ_THRESHOLD / self.count <= self.lemma_freq <= 1, f"Canon frequency out of range: {self.canon_freq}"

# Lemma data class
@dataclasses.dataclass(eq=False)
class CLemma:

	name: Optional[str]    # Main lemma (as it should be visualised on bounding boxes to the user)
	canon: str             # Canonical form of main lemma (as it should be output by the object decoder)
	select: bool           # Whether this lemma is selected for inclusion in the curated output
	bases: set[str]        # Set of pre-processed lemma bases this lemma was generated from (cased, always a subset of variants, all have same lemma canon)
	variants: set[str]     # Set of variants belonging to this lemma (cased, all have same lemma canon, always includes name and canon and bases)
	canon_freq: float = 0  # Word frequency of strictly only the canonical form of the main lemma relative to the frequency of the word 'the' (unknown/too rare = 0)
	alternates: set[str] = dataclasses.field(default_factory=set)            # Set of alternates belonging to this lemma (potentially but very rarely cased, must include canon of any included item as well)
	singulars: set[str] = dataclasses.field(default_factory=set)             # Set of singulars belonging to this lemma (cased, must include canon of any included item as well, may be empty if lemma is inherently plural)
	plurals: set[str] = dataclasses.field(default_factory=set)               # Set of plurals belonging to this lemma (lower case, must include canon of any included item as well)
	subcanons: set[SubCanon] = dataclasses.field(default_factory=set)        # Set of sub-canonical forms
	singulars_map: dict[str, int] = dataclasses.field(default_factory=dict)  # Map of all singulars to their integer multiplicities
	plurals_map: dict[str, int] = dataclasses.field(default_factory=dict)    # Map of all plurals to their integer multiplicities
	id: int = dataclasses.field(default_factory=itertools.count().__next__, init=False)  # Automatic unique ID

	def validate(self):
		if self.name is not None:
			assert self.name in self.variants, f"Variants does not include name: {self}"
		assert self.canon in self.variants, f"Variants does not include canon: {self}"
		assert self.bases <= self.variants, f"Variants does not include all bases: {self}"
		assert all(char in ALLOWED_CHARS_CANON for char in self.canon), f"Invalid canon chars: {self}"
		assert all(variant for variant in self.variants), f"Empty variant: {self}"
		assert all(char in ALLOWED_CHARS_CASED for variant in self.variants for char in variant), f"Invalid variant chars: {self}"
		assert all(get_canon(variant) == self.canon for variant in self.variants), f"Variant does not have correct canon: {self}"
		assert 0 <= self.canon_freq <= 1, f"Canon frequency out of range: {self.canon_freq}"
		assert all(alternate for alternate in self.alternates), f"Empty alternate: {self}"
		assert all(char in ALLOWED_CHARS_CASED for alternate in self.alternates for char in alternate), f"Invalid alternate chars: {self}"
		assert all(get_canon(alternate) in self.alternates for alternate in self.alternates), f"Alternates are not closed under canonicalisation: {self}"
		assert all(plural for plural in self.plurals), f"Empty plural: {self}"
		assert all(char in ALLOWED_CHARS_CASED for plural in self.plurals for char in plural), f"Invalid plural chars: {self}"
		assert all(get_canon(plural) in self.plurals for plural in self.plurals), f"Plurals are not closed under canonicalisation: {self}"
		assert all(singular for singular in self.singulars), f"Empty singular: {self}"
		assert all(char in ALLOWED_CHARS_CASED for singular in self.singulars for char in singular), f"Invalid singular chars: {self}"
		assert all(get_canon(singular) in self.singulars for singular in self.singulars), f"Singulars are not closed under canonicalisation: {self}"
		if self.subcanons:
			assert any(subcanon.canon == self.canon for subcanon in self.subcanons), f"Sub-canons does not include canon: {self}"
			assert set.union(self.singulars, self.plurals) <= set.union(*(subcanon.variants for subcanon in self.subcanons)), f"Not all singulars/plurals are in a subcanon: {self}"
			for subcanon in self.subcanons:
				subcanon.validate()
		if self.singulars_map or self.plurals_map:
			assert self.singulars <= set(self.singulars_map), f"Singulars map does not match singulars: {self}"
			assert self.plurals <= set(self.plurals_map), f"Plurals map does not match plurals: {self}"
			assert all(isinstance(mult, int) and mult >= 1 for mult in self.singulars_map.values()), f"Singulars map multiplicities are not all positive integers: {self}"
			assert all(isinstance(mult, int) and mult >= 1 for mult in self.plurals_map.values()), f"Plurals map multiplicities are not all positive integers: {self}"
		assert self.id >= 0, f"Unexpected negative ID: {self}"

# Validate clemmas
def validate_clemmas(iterable):
	for clemma in iterable:
		clemma.validate()

# Validate clemma dict
def validate_clemma_dict(clemma_dict):
	for canon, clemma in clemma_dict.items():
		assert canon == clemma.canon, f"Key is not canonical form: {canon} vs {clemma}"
		clemma.validate()

# Action: Curate nouns from a synset list
def action_curate(outfile, lemmas_file, list_file, alternates, spellings, objects, general, freqs):

	load_wordnet()

	canon_clemma_map = {}
	init_clemma_map(lemmas_file, canon_clemma_map)
	resolve_lemma_names(canon_clemma_map)
	validate_clemma_dict(canon_clemma_map)

	generate_lemma_map(list_file, canon_clemma_map)
	resolve_lemma_names(canon_clemma_map)
	validate_clemma_dict(canon_clemma_map)

	add_object_lists(canon_clemma_map, objects, select=True)
	resolve_lemma_names(canon_clemma_map)
	add_object_lists(canon_clemma_map, general, select=False)
	resolve_lemma_names(canon_clemma_map)
	validate_clemma_dict(canon_clemma_map)

	if spellings:
		add_american_british_alternates(canon_clemma_map, spellings)
	if alternates:
		add_stardict_alternates(canon_clemma_map, alternates)
	canonicalize_alts_plurals(canon_clemma_map)
	validate_clemma_dict(canon_clemma_map)

	lemmatizer = WordNetLemmatizer()
	plurals_map, singulars_map = load_plurals_map()
	populate_singulars_plurals(canon_clemma_map, lemmatizer, plurals_map, singulars_map)
	validate_clemma_dict(canon_clemma_map)

	inflect_engine = inflect.engine()
	add_inflect_plurals(canon_clemma_map, inflect_engine, lemmatizer, plurals_map)
	validate_clemma_dict(canon_clemma_map)

	gram_freqs = load_gram_freqs(freqs)
	populate_canon_freqs(canon_clemma_map, gram_freqs)
	validate_clemma_dict(canon_clemma_map)

	canon_clemma_map = merge_singulars(canon_clemma_map, gram_freqs)
	canon_clemma_map = merge_plurals(canon_clemma_map, gram_freqs)
	validate_clemma_dict(canon_clemma_map)

	canon_clemma_map = select_canons(canon_clemma_map)
	subcanon_map = generate_subcanons(canon_clemma_map, gram_freqs)
	validate_clemma_dict(canon_clemma_map)

	top_freqs = sorted((subcanon.canon_freq for subcanon in subcanon_map.values()), reverse=True)
	adjust_object_freqs(subcanon_map, top_freqs, objects, verbose=True)
	adjust_object_freqs(subcanon_map, top_freqs, general, verbose=False)
	validate_clemma_dict(canon_clemma_map)

	populate_lemma_freqs(subcanon_map)
	validate_clemma_dict(canon_clemma_map)

	calc_per_word_freqs(canon_clemma_map)
	validate_clemma_dict(canon_clemma_map)

	generate_json(canon_clemma_map, outfile)

# Initialise the lemma map
def init_clemma_map(lemmas_file, canon_clemma_map):
	print(f"Initialising lemma map from: {lemmas_file}")
	with open(lemmas_file, 'r') as file:
		for line in file:
			add_lemma_to_map(canon_clemma_map, line, select=False, person=False, allow_new=True, verbose=True)
	print(f"Initialised lemma map with {len(canon_clemma_map)} entries")
	print("Done")
	print()

# Populate a lemma map from WordNet
def generate_lemma_map(list_file, canon_clemma_map):

	print(f"Load list file: {list_file}")
	synset_names_list = []
	with open(list_file, 'r') as file:
		for line in file:
			synset_names_list.append(line.strip())
	synset_counter = collections.Counter(synset_names_list)
	for synset_name, count in synset_counter.items():
		if count > 1:
			print_info(34, f"Synset {synset_name} appeared {count} times")
	synset_names = set(synset_names_list)
	print(f"Loaded {len(synset_names)} selected synset names")
	print("Done")
	print()

	print("Collecting all lemmas from WordNet and marking whether they are selected...")
	collect_lemmas(synset_names, wordnet.synset(ROOT_SYNSET), canon_clemma_map)
	print(f"Collected {len(canon_clemma_map)} canonical lemmas of which {sum(clemma.select for clemma in canon_clemma_map.values())} are selected")
	print(f"Collected {sum(len(clemma.variants) for clemma in canon_clemma_map.values())} non-deduplicated lemma variants in total")
	print("Done")
	print()

# Collect all lemmas from WordNet
def collect_lemmas(synset_names, synset, canon_clemma_map):

	select = synset.name() in synset_names
	person = (synset.lexname() == 'noun.person')
	for lemma in synset.lemma_names():
		lemma = lemma.replace('_', ' ')  # Replace underscores with spaces
		add_lemma_to_map(canon_clemma_map, lemma, select=select, person=person, allow_new=True, verbose=False)

	hyponyms = synset.hyponyms() + synset.instance_hyponyms()
	for hypo in hyponyms:
		collect_lemmas(synset_names, hypo, canon_clemma_map)

# Add a lemma to a map
def add_lemma_to_map(canon_clemma_map, lemma, select, person, allow_new, verbose):

	lemma = ' '.join(lemma.split())  # Replace leading/trailing/multiple inbetween whitespace sequences with a single space
	if set(lemma) - ALLOWED_CHARS_CASED:
		print_warn(f"Invalid name chars: {lemma}")
	lemma = re.sub(r'(^|(?<=\s))([A-Z]\.\s*)+[A-Z]\.((?=\s)|$)', lambda m: re.sub(r'\s', r'', m.group()), (old_lemma := lemma))  # De-space abbreviations: U. S. A. --> U.S.A.
	if lemma != old_lemma and not person:
		print_info(34, f"Despaced abbreviation: {old_lemma} --> {lemma}")

	canon = get_canon(lemma)
	clemma = canon_clemma_map.get(canon, None)
	if clemma:
		if select:
			clemma.select = True
		if lemma in clemma.bases:
			return
		clemma.bases.add(lemma)

	lemma_dotless = lemma.replace('.', '')
	if lemma_dotless != lemma and not person:
		print_info(32, f"Added .-less variant: {lemma} --> {lemma_dotless}")
	lemma_apostless = re.sub(r"(?<!(\s|-)[dloDLO])(?<!^[dloDLO])(?<![sS])'(?=[^A-Z]|$)|(?<=[sS])'(?=[^sA-Z]|$)|'(?=\s|$)", r"", lemma)
	if lemma_apostless != lemma and not re.fullmatch(r"[^']*([a-rt-z]'s|s')(\s|-|$)[^']*", lemma):
		print_info(35, f"Added '-less variant: {lemma} --> {lemma_apostless}")
	lemma_apostless_dotless = lemma_apostless.replace('.', '')
	variants = {lemma, lemma_dotless, lemma_apostless, lemma_apostless_dotless}

	if clemma:
		if new_variants := variants - clemma.variants:
			clemma.variants.update(new_variants)
			if verbose:
				new_variants.discard(clemma.canon)
				if new_variants:
					print_info(37, f"Added new variant(s): {clemma.canon} --> {new_variants}")
	elif allow_new:
		canon_clemma_map[canon] = CLemma(name=None, canon=canon, bases={lemma}, select=select, variants=variants)
		if verbose:
			print_info(36, f"Added new canon: {canon}")

# Resolve lemma names
def resolve_lemma_names(canon_clemma_map):
	print("Resolving any new main lemma names...")
	num_names = 0
	for clemma in canon_clemma_map.values():
		if clemma.name is None:
			num_names += 1
			clemma.name = min(clemma.variants, key=lambda variant: (sum((3 if char == '/' else 2 if char in ".'" else 1 if char == '-' else 0) for char in variant), sum(char.isupper() for char in variant), variant))
			if any(variant.islower() for variant in clemma.variants):
				clemma.name = clemma.name.lower()
			if clemma.name not in clemma.variants:
				print_info(34, f"Added novel lemma name: {sorted(clemma.variants)} --> {clemma.name}")
				clemma.variants.add(clemma.name)
			elif len(clemma.variants) > 1 and len({variant.lower().replace('.', '').replace("'", "") for variant in clemma.variants}) > 1:
				print_info(32, f"Resolved lemma name: {sorted(clemma.variants)} --> {clemma.name}")
			clemma.variants.add(clemma.canon)
	print(f"Resolved {num_names} new main lemma names")
	print("Done")
	print()

# Add object lists
def add_object_lists(canon_clemma_map, objects_dir, select):
	print(f"Loading object lists from: {os.path.join(objects_dir, '*')}")
	print(f"Select mode: {select}")
	object_names = set()
	for filename in sorted(os.listdir(objects_dir)):
		filepath = os.path.join(objects_dir, filename)
		if filename.endswith('.txt') and not filename.startswith('_'):
			with open(filepath, 'r') as file:
				object_names.update(' '.join(entry.split()) for line in file for entry in line.split(','))
		else:
			print_warn(f"Ignoring object list: {filepath}")
	orig_map_size = len(canon_clemma_map)
	orig_lemma_variants = sum(len(clemma.variants) for clemma in canon_clemma_map.values())
	for object_name in object_names:
		add_lemma_to_map(canon_clemma_map, object_name if object_name.isupper() else object_name.lower(), select=select, person=False, allow_new=select, verbose=True)
	print(f"Loaded {len(object_names)} unique object names")
	print(f"Added {len(canon_clemma_map) - orig_map_size} new canonical lemmas")
	print(f"Total {len(canon_clemma_map)} canonical lemmas of which {sum(clemma.select for clemma in canon_clemma_map.values())} are selected")
	new_lemma_variants = sum(len(clemma.variants) for clemma in canon_clemma_map.values())
	print(f"Total {new_lemma_variants} non-deduplicated lemma variants (added {new_lemma_variants - orig_lemma_variants})")
	print("Done")
	print()

# Add American-British alternates
def add_american_british_alternates(canon_clemma_map, spellings):
	print("Loading American-British English mappings...")
	amkey_json_path, brkey_json_path = spellings
	print(f"American-to-British: {amkey_json_path}")
	print(f"British-to-American: {brkey_json_path}")
	with open(amkey_json_path, 'r') as file:
		amkey_map = json.load(file)
	assert isinstance(amkey_map, dict) and all(isinstance(key, str) and isinstance(value, str) for key, value in amkey_map.items())
	print(f"Loaded {len(amkey_map)} American -> British mappings")
	with open(brkey_json_path, 'r') as file:
		brkey_map = json.load(file)
	assert isinstance(brkey_map, dict) and all(isinstance(key, str) and isinstance(value, str) for key, value in brkey_map.items())
	print(f"Loaded {len(brkey_map)} British -> American mappings")
	for clemma in canon_clemma_map.values():
		for variant in clemma.variants:
			variant_parts = variant.split()
			assert variant == ' '.join(variant_parts)
			variant_british = ' '.join(amkey_map.get(part, part) for part in variant_parts)
			if variant_british not in clemma.variants and variant_british not in clemma.alternates:
				clemma.alternates.add(variant_british)
				if len(variant_parts) <= 1 and not simple_am_br_change(variant, variant_british):
					print_info(32, f"Added non-trivial British monogram: {variant} --> {variant_british}")
			variant_american = ' '.join(brkey_map.get(part, part) for part in variant_parts)
			if variant_american not in clemma.variants and variant_american not in clemma.alternates:
				clemma.alternates.add(variant_american)
				if len(variant_parts) <= 1 and not simple_am_br_change(variant_american, variant):
					print_info(34, f"Added non-trivial American monogram: {variant} --> {variant_american}")
	print("Done")
	print()

# Add StarDict dict.org alternates
def add_stardict_alternates(canon_clemma_map, alternates):

	print("Collecting StarDict dict.org noun alternates...")
	print(f"StarDict dict.org TSV: {alternates}")

	singular_map = {}
	plural_map = {}
	ignored_words = set()
	made_substitutions = set()
	num_entries = num_nouns = num_sing = num_pl = num_headword_ambig = num_headword_plural = num_altform_ambig = num_no_headwords = num_headword_exdist = num_sameas_exdist = num_altforms_exdist = 0

	with open(alternates, 'r', newline='') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:

			key, entry = row
			entry = ' '.join(re.sub(r'(?<![^\\]\\)\\n', r'\n', entry).split())
			match = re.fullmatch((
				r"((?:(?:or )?[- a-zA-Z0-9'|]*\\\\[^\\]+\\\\(?: ?\([^)]*\))*[,;.]* ?)+)"              # MATCHES: caddis fly \\caddis fly\\ (Zo["o]l.), caddisfly \\caddisfly\\, --> GROUP 1
				r"((?: ?(?:[a-z]*\.|&))*)"                                                            # MATCHES: n. sing. & pl. --> GROUP 2
				r"(?:[,;.] ?((?:(?:or )?[- a-zA-Z0-9'|]*\\\\[^\\]+\\\\(?: ?\([^)]*\))*[,;.]* ?)+))?"  # MATCHES: , Grouse \\Grouse\\ (grous), --> GROUP 3 (without leading punctuation) or None
				r"([,;.] ?(pl|sing)\.(?: ?(?:(?:[LE]\.|or) ?)?\{[^}]*}(?: ?\([^)]*\))*[,;.]* ?)+)?"   # MATCHES: ; pl. {Lilies} (l[i^]l"[i^]z). --> GROUP 5 or None (GROUP 4 = pl|sing)
				r"((?: ?(?:\([^)]*\)|\[(?:[^\[\]]*\[[^\[\]]*])*[^\[\]]*]))*)"                         # MATCHES: [As. box, L. buxus, fr. Gr. ?. See {Box} a case.] (Bot.) --> GROUP 6
				r"([,;.]* ?[Ss]ame as (?:\{[^}]*}[,;.]* ?)+)?"                                        # MATCHES: Same as {Antihelix}. --> GROUP 7 or None
				r"(.*)"                                                                               # MATCHES: All remaining text --> GROUP 8
			), entry)

			if match:

				num_entries += 1
				word_group, pos_group, extra_word_group, altform_group, altform_type_group, brackets_group, same_as_group, defn_group = match.groups()
				positions = set(re.findall(r'(?:^|[ .&])(n|pl|sing)\.', pos_group))

				if 'n' in positions:

					num_nouns += 1
					num_pl += (headwords_are_plural := 'pl' in positions)
					num_sing += (headwords_are_singular := not headwords_are_plural or 'sing' in positions)

					if headwords_are_plural and headwords_are_singular:
						num_headword_ambig += 1
					elif headwords_are_plural:
						num_headword_plural += 1
					else:

						headwords = {headword.strip().replace('|', '').lower() for headword_group in (word_group, extra_word_group or '') for headword in re.findall(r"(?:or )?([- a-zA-Z0-9'#|]*)\\\\[^\\]+\\\\(?: ?\([^)]*\))*[,;.]* ?", headword_group)}

						if altform_group and altform_type_group:
							altforms = {altform.strip().lower() for altform in re.findall(r'\{([^}]*)}', altform_group)}
							altforms_are_plural = (altform_type_group == 'pl')
						else:
							altforms = set()
							altforms_are_plural = not headwords_are_plural

						if headwords_are_plural == altforms_are_plural:
							num_altform_ambig += 1
						else:

							if same_as_group:
								sameas = {same.strip().lower() for same in re.findall(r'\{([^}]*)}', same_as_group)}
							else:
								sameas = set()
							written_also = {written.strip().lower() for group in (brackets_group, defn_group) for written_section in re.findall(r"\[ ?[Ww]ritten also ((?:[^\[\]]*\[[^\[\]]*])*[^\[\]]*)]", group) for written in re.findall(r'\{([^}]*)}', written_section)}

							headwords, ignored_words, made_substitutions = clean_stardict_alternates(set.union(headwords, written_also), ignored_words, made_substitutions)
							sameas, ignored_words, made_substitutions = clean_stardict_alternates(sameas, ignored_words, made_substitutions)
							altforms, ignored_words, made_substitutions = clean_stardict_alternates(altforms, ignored_words, made_substitutions)

							if not headwords:
								num_no_headwords += 1
							else:

								headword_dist_thres = min(round(2.15 * math.sqrt(max(max(len(headword) for headword in headwords) - 3, 0))), 5)
								max_headword_dist = max((edit_distance(headword_a, headword_b, substitution_cost=2) for headword_a, headword_b in itertools.combinations(headwords, 2)), default=0)
								if max_headword_dist > headword_dist_thres:
									num_headword_exdist += 1
								else:

									if sameas:
										sameas_dist_thres = min(max(round(2.15 * math.sqrt(max(max(len(same) for same in sameas) - 3, 0))), headword_dist_thres), 5)
										num_sameas = len(sameas)
										sameas = {same for same in sameas if len(same) >= 4 and all(edit_distance(same, headword, substitution_cost=2) <= sameas_dist_thres for headword in headwords)}
										num_sameas_exdist += num_sameas - len(sameas)

									if altforms:
										altforms_dist_thres = min(max(round(2.15 * math.sqrt(max(max(len(altform) for altform in altforms) - 3, 0))), headword_dist_thres), 5)
										num_altforms = len(altforms)
										altforms = {altform for altform in altforms if len(altform) >= 4 and any(edit_distance(altform, headword, substitution_cost=2) <= altforms_dist_thres for headword in headwords)}
										num_altforms_exdist += num_altforms - len(altforms)

									singulars = set.union(headwords, sameas)
									plurals = altforms

									for singular in singulars:
										if (alt_singulars := singular_map.get(singular, None)) is None:
											singular_map[singular] = singulars.copy()
										else:
											diff_singulars = singulars - alt_singulars
											diff_alt_singulars = alt_singulars - singulars
											if diff_singulars and diff_alt_singulars:
												print_info(34, f"Merging singulars: {diff_alt_singulars} = {set.intersection(alt_singulars, singulars)} = {diff_singulars}")
											alt_singulars.update(singulars)
										if (alt_plurals := plural_map.get(singular, None)) is None:
											plural_map[singular] = plurals.copy()
										else:
											diff_plurals = plurals - alt_plurals
											diff_alt_plurals = alt_plurals - plurals
											if diff_plurals and diff_alt_plurals:
												print_info(35, f"Merging plurals: {alt_plurals} = {singular} = {plurals}")
											alt_plurals.update(plurals)

			elif not (key.startswith('##') or key.startswith('00-') or key == '0'):
				line = f"{key} => {entry}"
				print_warn(f"Ignoring unparseable line: {line if len(line) <= 120 else line[:117] + '...'}")

	for singular, alt_singulars in singular_map.items():
		alt_singulars.discard(singular)
	singular_map = {singular: alt_singulars for singular, alt_singulars in singular_map.items() if alt_singulars}
	plural_map = {singular: alt_plurals for singular, alt_plurals in plural_map.items() if alt_plurals}
	plural_map['pajama'] = plural_map.get('pajama', set()).union({'PJs'})

	for singular, alt_singulars in singular_map.items():
		dist_thres = min(round(2.15 * math.sqrt(max(max(max(len(alt_singular) for alt_singular in alt_singulars), len(singular)) - 3, 0))), 5)
		max_dist = max(edit_distance(singular, alt_singular, substitution_cost=2) for alt_singular in alt_singulars)
		if max_dist > dist_thres:
			print_warn(f"Accepting singular of edit distance {max_dist} > {dist_thres}: {singular} = {alt_singulars}")
	for singular, alt_plurals in plural_map.items():
		dist_thres = min(round(2.15 * math.sqrt(max(max(max(len(alt_plural) for alt_plural in alt_plurals), len(singular)) - 3, 0))), 5)
		max_dist = max(edit_distance(singular, alt_plural, substitution_cost=2) for alt_plural in alt_plurals)
		if max_dist > dist_thres:
			print_warn(f"Accepting plural of edit distance {max_dist} > {dist_thres}: {singular} = {alt_plurals}")

	num_alternates_added = num_plurals_added = 0
	for clemma in canon_clemma_map.values():
		new_alternates = set()
		new_plurals = set()
		for variant in clemma.variants:
			if (alt_singulars := singular_map.get(variant, None)) is not None:
				new_alternates.update(alt_singulars)
			if (alt_plurals := plural_map.get(variant, None)) is not None:
				new_plurals.update(alt_plurals)
		if new_alternates:
			new_alternates.difference_update(clemma.variants, clemma.alternates)
			if new_alternates:
				clemma.alternates.update(new_alternates)
				print_info(36, f"Added alternates: {clemma.variants} --> {new_alternates}")
				num_alternates_added += len(new_alternates)
		if new_plurals:
			new_plurals.difference_update(clemma.variants, clemma.alternates)
			if new_plurals:
				clemma.plurals.update(new_plurals)
				print_info(37, f"Added plurals: {clemma.variants} --> {new_plurals}")
				num_plurals_added += len(new_plurals)

	print(f"Parsed a total of {num_entries} dictionary entries, of which {num_nouns} were nouns")
	print(f"Found {num_sing} singular nouns and {num_pl} plural nouns")
	print(f"Excluded {num_headword_ambig} nouns that were both singular and plural")
	print(f"Excluded {num_headword_plural} further nouns that were plural already as a headword")
	print(f"Excluded {num_altform_ambig} nouns that had an altform singular/plural mismatch")
	print(f"Ignored {len(ignored_words)} non-word alternates")
	print(f"Made {len(made_substitutions)} alternate substitutions")
	print(f"Skipped {num_no_headwords} nouns due to ignored/missing headwords")
	print(f"Skipped {num_headword_exdist} headword sets that were too different from one another")
	print(f"Skipped {num_sameas_exdist} same-as specifications that were too different to the headwords")
	print(f"Skipped {num_altforms_exdist} plural forms that were too different to the headwords")
	print(f"Collected {sum(len(alt_singulars) for alt_singulars in singular_map.values())} singular alternates for {len(singular_map)} singulars (overcounted)")
	print(f"Collected {sum(len(alt_plurals) for alt_plurals in plural_map.values())} plural alternates for {len(plural_map)} singulars (overcounted)")
	print(f"Added {num_alternates_added} new alternates to the main lemma map")
	print(f"Added {num_plurals_added} new plurals to the main lemma map")
	print("Done")
	print()

# Clean a StarDict alternate
def clean_stardict_alternates(alternates, ignored_words, made_substitutions):
	cleaned_alternates = set()
	for alternate in alternates:
		alternate = re.sub(r"\[['=~\"^`,0-9]?([a-z]*)]", r'\1', orig_alternate := alternate).rstrip('.')
		if any(char not in ALLOWED_CHARS_STARDICT for char in alternate) or alternate[0] in "-'":
			if alternate not in ignored_words:
				print_warn(f"Ignored non-word alternate: {alternate}")
				ignored_words.add(alternate)
		else:
			if alternate != orig_alternate and orig_alternate not in made_substitutions:
				if orig_alternate.replace('[ae]', 'ae') != alternate and orig_alternate.replace('[oe]', 'oe') != alternate:
					print_info(32, f"Made alternate substitution: {orig_alternate} --> {alternate}")
				made_substitutions.add(orig_alternate)
			cleaned_alternates.add(alternate)
	return cleaned_alternates, ignored_words, made_substitutions

# Ensure alternates and plurals are closed under canonicalization
def canonicalize_alts_plurals(canon_clemma_map):
	print("Closing alternate/plural sets under canonicalization...")
	for clemma in canon_clemma_map.values():
		new_alternates = set()
		for alternate in tuple(clemma.alternates):
			canon_alternate = get_canon(alternate)
			if canon_alternate not in clemma.alternates:
				new_alternates.add(canon_alternate)
		if new_alternates:
			print_info(32, f"Added new canon alternates: {clemma.alternates} --> {new_alternates}")
			clemma.alternates.update(new_alternates)
		new_plurals = set()
		for plural in tuple(clemma.plurals):
			canon_plural = get_canon(plural)
			if canon_plural not in clemma.plurals:
				new_plurals.add(canon_plural)
		if new_plurals:
			print_info(34, f"Added new canon plurals: {clemma.plurals} --> {new_plurals}")
			clemma.plurals.update(new_plurals)
	print("Done")
	print()

# Detect plurals and populate singulars/plurals using the inflect library
def populate_singulars_plurals(canon_clemma_map, lemmatizer, plurals_map, singulars_map):
	print("Detecting which variants and alternates are already plural and populating singulars/plurals...")
	for clemma in canon_clemma_map.values():
		new_plural_canons = set()
		variants_alternates = set.union(clemma.variants, clemma.alternates)
		if not any(word.isupper() for word in variants_alternates):
			for word in variants_alternates:
				if is_likely_plural(word, lemmatizer, plurals_map, singulars_map):
					new_plural_canons.add(get_canon(word))
		have_new_plurals = False
		if new_plural_canons:
			for word in variants_alternates:
				if get_canon(word) in new_plural_canons:
					clemma.plurals.add(word)
					have_new_plurals = True
				else:
					clemma.singulars.add(word)
		else:
			clemma.singulars.update(variants_alternates)
		if have_new_plurals:
			print_info(32, f"Detected inherently plural lemma: Singular {clemma.singulars} <-> Plural {clemma.plurals}")
	print("Done")
	print()

# Add plurals using the inflect library
def add_inflect_plurals(canon_clemma_map, inflect_engine, lemmatizer, plurals_map):
	print("Generating plural forms of selected singulars...")
	for clemma in canon_clemma_map.values():
		allow_inflect = allow_inflect_engine(clemma.name, lemmatizer) and all(c in ALLOWED_CHARS_SAFE for singular in clemma.singulars for c in singular)
		for singular in clemma.singulars:
			if plurals := plurals_map.get(singular, None):
				clemma.plurals.update(plurals)
				clemma.plurals.update(get_canon(plural) for plural in plurals)
			elif allow_inflect and allow_inflect_engine(singular, lemmatizer):
				plural = inflect_engine.plural_noun(singular)
				if len(plural) != len(singular) + 1 or all(plural[:m.start()] + plural[m.end():] != singular for m in re.finditer(r'(s)(?=s(\s|-|$))', plural)):
					clemma.plurals.add(plural)
					clemma.plurals.add(get_canon(plural))
					if singular == plural:
						print_info(34, f"Added plural identical to singular: {plural}")
					elif not simple_ending_changed(singular, plural):
						print_info(32, f"Added non-trivial plural: {singular} --> {plural}")
	print("Done")
	print()

# Populate CLemma name word frequencies
def populate_canon_freqs(canon_clemma_map, gram_freqs):
	print("Populating lemma canonical frequencies...")
	num_nonzero = 0
	for clemma in canon_clemma_map.values():
		clemma.canon_freq = gram_freqs.get(clemma.canon, 0)
		if clemma.canon_freq > 0:
			num_nonzero += 1
	print(f"Found gram frequencies for {num_nonzero} out of {len(canon_clemma_map)} canonical keys")
	print("Done")
	print()

# Merge singulars across CLemmas
def merge_singulars(canon_clemma_map, gram_freqs):

	print("Merging lemmas with shared singulars...")

	singular_clemma_map = {}
	for clemma in canon_clemma_map.values():
		for singular in clemma.singulars:
			if (clemmas := singular_clemma_map.get(singular, None)) is None:
				singular_clemma_map[singular] = {clemma}
			else:
				clemmas.add(clemma)

	matched_clemmas_map = {}
	for singular, clemmas in singular_clemma_map.items():
		for clemma_a, clemma_b in itertools.combinations(clemmas, 2):
			if (matched_clemmas := matched_clemmas_map.get(clemma_a, None)) is None:
				matched_clemmas_map[clemma_a] = {clemma_b}
			else:
				matched_clemmas.add(clemma_b)
			if (matched_clemmas := matched_clemmas_map.get(clemma_b, None)) is None:
				matched_clemmas_map[clemma_b] = {clemma_a}
			else:
				matched_clemmas.add(clemma_a)

	canon_clemma_map = merge_matches(canon_clemma_map, gram_freqs, matched_clemmas_map)

	print("Done")
	print()

	return canon_clemma_map

# Merge plurals across CLemmas
def merge_plurals(canon_clemma_map, gram_freqs):

	print("Conditionally merging lemmas with shared plurals...")

	plural_clemma_map = {}
	for clemma in canon_clemma_map.values():
		for plural in clemma.plurals:
			if (clemmas := plural_clemma_map.get(plural, None)) is None:
				plural_clemma_map[plural] = {clemma}
			else:
				clemmas.add(clemma)

	matched_clemmas_map = {}
	for plural, clemmas in plural_clemma_map.items():
		for clemma_a, clemma_b in itertools.combinations(clemmas, 2):
			if allow_plural_match(clemma_a, clemma_b):
				if (matched_clemmas := matched_clemmas_map.get(clemma_a, None)) is None:
					matched_clemmas_map[clemma_a] = {clemma_b}
				else:
					matched_clemmas.add(clemma_b)
				if (matched_clemmas := matched_clemmas_map.get(clemma_b, None)) is None:
					matched_clemmas_map[clemma_b] = {clemma_a}
				else:
					matched_clemmas.add(clemma_a)
			else:
				print_info(36, f"Prevented plural-based merge: {clemma_a.name} <-> {clemma_b.name}")

	canon_clemma_map = merge_matches(canon_clemma_map, gram_freqs, matched_clemmas_map)

	print("Done")
	print()

	return canon_clemma_map

# Merge lemmas given a match map
def merge_matches(canon_clemma_map, gram_freqs, matched_clemmas_map):

	new_canon_clemma_map = {}
	seen_clemmas = set()
	num_collected_clemmas = 0
	merged_clemma_canons = set()
	num_merged_clemmas = 0

	for clemma in canon_clemma_map.values():

		if clemma in seen_clemmas:
			continue

		pending_clemmas = {clemma}
		collected_clemmas = set()
		while pending_clemmas:
			pending_clemma = pending_clemmas.pop()
			collected_clemmas.add(pending_clemma)
			if (matched_clemmas := matched_clemmas_map.get(pending_clemma, None)) is not None:
				pending_clemmas.update(matched_clemmas - collected_clemmas)
		num_collected_clemmas += len(collected_clemmas)
		seen_clemmas.update(collected_clemmas)

		if len(collected_clemmas) <= 1:
			unmatched_lemma = next(iter(collected_clemmas))
			new_canon_clemma_map[unmatched_lemma.canon] = unmatched_lemma
		else:

			best_lemma = max(collected_clemmas, key=lambda cl: (
				cl.canon in cl.singulars,
				cl.canon_freq,
				math.prod(gram_freqs.get(part, 0) for part in cl.canon.split()),
				-sum((3 if char == '/' else 2 if char in ".'" else 1 if char == '-' else 0) for char in cl.name),
				-sum(char.isupper() for char in cl.name),
				-cl.name.count(' '),
				-len(cl.name),
				cl.name,
			))

			merged_lemma = copy.deepcopy(best_lemma)
			for lemma in collected_clemmas:
				merged_lemma.select |= lemma.select
				merged_lemma.singulars.update(lemma.singulars)
				merged_lemma.plurals.update(lemma.plurals)

			if merged_lemma.select != best_lemma.select or merged_lemma.singulars != best_lemma.singulars or merged_lemma.plurals != best_lemma.plurals:
				if any(clemma_a not in matched_clemmas_map[clemma_b] or clemma_b not in matched_clemmas_map[clemma_a] for clemma_a, clemma_b in itertools.combinations(collected_clemmas, 2)):
					print_info(34, f"Merging non-densely matched lemmas: {', '.join(sorted(lemma.name for lemma in collected_clemmas))} --> {merged_lemma.name}")
				else:
					print_info(32, f"Merging densely matched lemmas: {', '.join(sorted(lemma.name for lemma in collected_clemmas))} --> {merged_lemma.name}")

			new_canon_clemma_map[merged_lemma.canon] = merged_lemma
			merged_clemma_canons.update(lemma.canon for lemma in collected_clemmas if lemma.canon != merged_lemma.canon)
			num_merged_clemmas += len(collected_clemmas) - 1

	assert len(seen_clemmas) == num_collected_clemmas == len(canon_clemma_map) == len(new_canon_clemma_map) + num_merged_clemmas
	assert len(merged_clemma_canons) == num_merged_clemmas and set(canon_clemma_map) == set.union(set(new_canon_clemma_map), merged_clemma_canons)
	assert all(isinstance(key, str) and isinstance(value, CLemma) for key, value in new_canon_clemma_map.items())

	return new_canon_clemma_map

# Helper function whether to allow plural merge-matches
def allow_plural_match(clemma_a, clemma_b):
	def disallow_clemma(clemma, clemma_other):
		return (
			clemma.name.endswith('is') or
			(clemma.name.isupper() and clemma_other.name != clemma.name + 's') or
			(len(clemma.name) <= 5 and clemma.name[0].isupper() and clemma.name.endswith('es')) or
			(clemma.name.endswith('f') and not clemma_other.name.endswith('s')) or
			clemma.name in ('Mars', 'Thebes', 'Masters')
		)
	if disallow_clemma(clemma_a, clemma_b) or disallow_clemma(clemma_b, clemma_a):
		return False
	return True

# Filter out any unselected CLemmas from a map
def select_canons(canon_clemma_map):
	print("Reducing lemmas to just the selected ones...")
	canon_clemma_map = {canon: clemma for canon, clemma in canon_clemma_map.items() if clemma.select and len(canon) > 1}
	print(f"Total {len(canon_clemma_map)} canonical lemmas of which {sum(clemma.select for clemma in canon_clemma_map.values())} are selected")
	print(f"Total {sum(len(clemma.variants) for clemma in canon_clemma_map.values())} non-deduplicated lemma variants")
	print("Done")
	print()
	return canon_clemma_map

# Generate sub-canonicals map from selected CLemmas
def generate_subcanons(canon_clemma_map, gram_freqs):
	print("Generating subcanonical map from lemmas...")
	subcanon_map = {}
	for clemma in canon_clemma_map.values():
		clemma_subcanon_map = {}
		for words, is_singular in ((clemma.singulars, True), (clemma.plurals, False)):
			for word in words:
				word_canon = get_canon(word)
				if (subcanon := clemma_subcanon_map.get(word_canon, None)) is not None:
					subcanon.variants.add(word)
					if is_singular:
						subcanon.singular = True
					else:
						subcanon.plural = True
				else:
					clemma_subcanon_map[word_canon] = SubCanon(canon=word_canon, canon_freq=gram_freqs.get(word_canon, 0), variants={word}, singular=is_singular, plural=not is_singular)
		for clemma_subcanon in clemma_subcanon_map.values():
			if (subcanon := subcanon_map.get(clemma_subcanon.canon, None)) is not None:
				subcanon.variants.update(clemma_subcanon.variants)
				if clemma_subcanon.singular:
					subcanon.singular = True
				if clemma_subcanon.plural:
					subcanon.plural = True
				subcanon.count += 1
				print_info(34, f"Subcanonical occurs in multiple lemmas: {subcanon.canon}")
			else:
				subcanon_map[clemma_subcanon.canon] = clemma_subcanon
				subcanon = clemma_subcanon
			clemma.subcanons.add(subcanon)
	assert all(key == value.canon for key, value in subcanon_map.items())
	assert set(subcanon_map.values()) == set.union(*(clemma.subcanons for clemma in canon_clemma_map.values()))
	assert all(sum(1 for clemma in canon_clemma_map.values() if subcanon.canon in clemma.singulars or subcanon.canon in clemma.plurals) == subcanon.count for subcanon in subcanon_map.values() if subcanon.count >= 2)
	print(f"Collected {len(subcanon_map)} subcanonicals into a map")
	print("Done")
	print()
	return subcanon_map

# Adjust object frequencies based on an object list
def adjust_object_freqs(subcanon_map, top_freqs, objects_dir, verbose):

	print(f"Loading object lists from: {os.path.join(objects_dir, '*')}")
	print(f"Verbose mode: {verbose}")

	canon_freq_map = {}
	for filename in sorted(os.listdir(objects_dir)):
		filepath = os.path.join(objects_dir, filename)

		if filename.endswith('.txt') and not filename.startswith('_'):
			with open(filepath, 'r') as file:
				object_sets = tuple({get_canon(entry) for entry in line.split(',')} for line in file)
				num_rows_orig = len(object_sets)
				num_canons_orig = sum(len(object_set) for object_set in object_sets)
				print_info(32, f"Object list {filename}: {num_rows_orig} rows and {num_canons_orig} canonicals")
				if not object_sets:
					continue
		else:
			print_warn(f"Ignoring object list: {filepath}")
			continue

		num_pools = max(len(object_set) for object_set in object_sets)
		pools = tuple({} for _ in range(num_pools + 1))
		for object_set in object_sets:
			if object_set:
				pools[len(object_set)][id(object_set)] = object_set

		seen_objects = set()
		for pool in pools:
			for object_set in pool.values():
				if prune_objects := object_set & seen_objects:
					if verbose:
						print_info(34, f"Pruned {prune_objects} from {object_set}")
					object_set.difference_update(prune_objects)
				seen_objects.update(object_set)

		object_sets = tuple(object_set for object_set in object_sets if object_set)
		num_rows = len(object_sets)
		num_canons = sum(len(object_set) for object_set in object_sets)
		if num_rows != num_rows_orig or num_canons != num_canons_orig:
			print_info(36, f"Object list {filename}: {num_rows} rows and {num_canons} canonicals (deduplicated)")

		assert object_sets
		freq_per_row = top_freqs[min(len(object_sets) - 1, len(top_freqs) - 1)]
		for object_set in object_sets:
			freq_per_obj = freq_per_row / len(object_set)
			for obj in object_set:
				canon_freq_map[obj] = max(canon_freq_map.get(obj, 0), freq_per_obj)

	num_freq_updates = num_found = num_not_found = 0
	for obj, min_freq in canon_freq_map.items():
		if (subcanon := subcanon_map.get(obj, None)) is not None:
			num_found += 1
			if min_freq > subcanon.canon_freq:
				subcanon.canon_freq = min_freq
				num_freq_updates += 1
		else:
			num_not_found += 1
			if verbose:
				print_warn(f"Not found in subcanonical map: {obj}")

	print(f"Collected a total of {len(canon_freq_map)} subcanonical frequency lower bounds")
	print(f"Updated {num_freq_updates}/{num_found} freqs and did not find {num_not_found} freqs")
	print("Done")
	print()

# Populate the lemma frequencies
def populate_lemma_freqs(subcanon_map):
	print("Populating subcanonical lemma frequencies...")
	for subcanon in subcanon_map.values():
		subcanon.lemma_freq = max(subcanon.canon_freq, FREQ_THRESHOLD) / subcanon.count
	print("Done")
	print()

# Calculate per-word frequencies based on subcanonical frequencies
def calc_per_word_freqs(canon_clemma_map):
	print("Assigning per-word frequencies to the lemmas...")
	for clemma in canon_clemma_map.values():
		subcanons = tuple(clemma.subcanons)
		clemma_freq = sum(subcanon.lemma_freq for subcanon in subcanons)
		clemma_mult = calc_mult(clemma_freq)
		word_mults = tuple(calc_mult(subcanon.lemma_freq) for subcanon in subcanons)
		word_mults_sum = sum(word_mults)
		singulars_map = {}
		plurals_map = {}
		for subcanon, word_mult in zip(subcanons, word_mults):
			variant_mult = max(round(clemma_mult * (word_mult / word_mults_sum) / ((subcanon.singular + subcanon.plural) * len(subcanon.variants))), 1)
			for variant in subcanon.variants:
				if subcanon.singular:
					assert variant not in singulars_map
					singulars_map[variant] = variant_mult
				if subcanon.plural:
					assert variant not in plurals_map
					plurals_map[variant] = variant_mult
		clemma.singulars_map = singulars_map
		clemma.plurals_map = plurals_map
		if clemma_freq >= CLEMMA_FREQ_PRINT:
			print_info(32, f"{clemma.canon} has multiplicity {sum(clemma.singulars_map.values()) + sum(clemma.plurals_map.values())}: {tuple(clemma.singulars_map.items())} + {tuple(clemma.plurals_map.items())}")
	print("Done")
	print()

# Calculate the log-scale multiplicity from a word frequency
def calc_mult(freq):
	return 1 + MULT_SCALE * math.log(freq / FREQ_THRESHOLD)

# Generate output JSON file
def generate_json(canon_clemma_map, outfile):
	print(f"Generating output JSON file: {outfile}")
	vocab = []
	for clemma in sorted(canon_clemma_map.values(), key=lambda item: item.canon):
		if clemma.select:
			singulars, singulars_freq = zip(*sorted(clemma.singulars_map.items())) if clemma.singulars_map else ((), ())
			plurals, plurals_freq = zip(*sorted(clemma.plurals_map.items())) if clemma.plurals_map else ((), ())
			vocab.append(dict(
				id=clemma.id,
				target_noun=clemma.canon,
				pretty_noun=clemma.name,
				singulars=singulars,
				plurals=plurals,
				singulars_freq=singulars_freq,
				plurals_freq=plurals_freq,
				hypernyms=[],  # TODO: Implement hypernym tracking (be careful where merging happens as IDs are lost)
			))
	print(f"Vocabulary entries: {len(vocab)}")
	print(f"Input noun chars: \"{''.join(sorted({char for entry in vocab for word in itertools.chain(entry['singulars'], entry['plurals']) for char in word}))}\"")
	print(f"Target noun chars: \"{''.join(sorted({char for entry in vocab for char in entry['target_noun']}))}\"")
	print(f"Pretty noun chars: \"{''.join(sorted({char for entry in vocab for char in entry['pretty_noun']}))}\"")
	num_singulars = sum(len(entry['singulars']) for entry in vocab)
	num_plurals = sum(len(entry['plurals']) for entry in vocab)
	num_freq_singulars = sum(sum(entry['singulars_freq']) for entry in vocab)
	num_freq_plurals = sum(sum(entry['plurals_freq']) for entry in vocab)
	print(f"Num singulars: {num_singulars} ({num_freq_singulars} with freq)")
	print(f"Num plurals: {num_plurals} ({num_freq_plurals} with freq)")
	print(f"Word freq: Min {min(min(min(entry['singulars_freq'], default=math.inf), min(entry['plurals_freq'], default=math.inf)) for entry in vocab)}, Mean {(num_freq_singulars + num_freq_plurals) / (num_singulars + num_plurals):.2f}, Max {max(max(max(entry['singulars_freq'], default=-math.inf), max(entry['plurals_freq'], default=-math.inf)) for entry in vocab)}")
	lemma_freqs = tuple(sum(entry['singulars_freq']) + sum(entry['plurals_freq']) for entry in vocab if len(entry['singulars_freq']) + len(entry['plurals_freq']) <= 8)  # Note: There are a few outliers with huge numbers of variants, which seeing as each variant gets at least a value of 1 results in a 'faulty' maximum
	print(f"Lemma freq: Min {min(lemma_freqs)}, Mean {sum(lemma_freqs) / len(lemma_freqs):.2f}, Max {max(lemma_freqs)}")
	print(f"Num hypernyms: {sum(len(entry['hypernyms']) for entry in vocab)}")
	all_ids = {entry['id'] for entry in vocab}
	assert len(all_ids) == len(vocab)
	assert all(set(entry['hypernyms']) < all_ids for entry in vocab)
	with open(outfile, 'w') as file:
		json.dump(vocab, file, ensure_ascii=False, indent=2, sort_keys=False)
	print("Done")
	print()

# Convert variant to canonical form
def get_canon(variant):
	canon = variant.lower()
	canon = canon.replace("'", "").replace('.', '')
	canon = ' '.join(part for part in re.split(r'[\s/-]+', canon) if part)
	if set(canon) - ALLOWED_CHARS_CANON:
		print_warn(f"Invalid canon chars: {canon}")
	return canon

# Check whether a word is likely already plural
def is_likely_plural(word, lemmatizer, plurals_map, singulars_map):
	if word in plurals_map:
		return False
	elif word in singulars_map:
		return True
	elif len(word) >= 3 and word.endswith('s') and word[:-1].isupper():
		return True
	elif len(word) < 4 or word.endswith('ss') or (word.endswith('as') and not word.endswith('jamas') and not word.endswith(' peas')):
		return False
	word_parts = word.split()
	num_word_parts = len(word_parts)
	if num_word_parts >= 3 and any(joiner in word_parts for joiner in ('of', 'for', 'on', 'in', 'at')):
		return False
	elif num_word_parts == 2 and word_parts[0] in ('class', 'genus', 'subclass', 'order', 'suborder'):
		return False
	elif lemmatizer.lemmatize(word) != word:
		return True
	last_word = word_parts[-1]
	return len(last_word) >= 4 and lemmatizer.lemmatize(last_word) != last_word

# Heuristic whether inflect engine plural noun is expected to work well
def allow_inflect_engine(word, lemmatizer):
	if len(word) <= 2 or not word.islower():
		return False
	hyphen_count = word.count('-')
	if hyphen_count >= 2:
		return False
	word_parts = word.split()
	num_parts = len(word_parts)
	if num_parts >= 3 or (num_parts >= 2 and hyphen_count >= 2):
		return False
	if lemmatizer.lemmatize(word) != word:
		return False
	if len(word_parts) > 1:
		last_word = word_parts[-1]
		if lemmatizer.lemmatize(last_word) != last_word:
			return False
	return True

# Detect whether the difference between an American and British form is simple
TRIVIAL_AM_BR = (('o', 'ou'), ('z', 's'), ('l', 'll'), ('e', 'ae'), ('e', 'oe'), ('er', 're'))
def simple_am_br_change(american, british, am_br=TRIVIAL_AM_BR):
	for am, br in am_br:
		last_br = british.rfind(br)
		if last_br >= 0 and f"{british[:last_br]}{am}{british[last_br + len(br):]}" == american:
			return True
		last_am = american.rfind(am)
		if last_am >= 0 and f"{american[:last_am]}{br}{american[last_am + len(am):]}" == british:
			return True
	return False

# Detect whether simply an ending changed from a singular to a plural
TRIVIAL_ENDINGS = (('', 's'), ('', 'es'), ('y', 'ies'), ('f', 'ves'), ('fe', 'ves'), ('is', 'es'), ('man', 'men'), ('oot', 'eet'), ('ooth', 'eeth'), (' person', ' people'))
def simple_ending_changed(singular, plural, endings=TRIVIAL_ENDINGS):
	for singular_ending, plural_ending in endings:
		if singular_ending:
			if singular.endswith(singular_ending) and singular[:-len(singular_ending)] + plural_ending == plural:
				return True
		elif singular + plural_ending == plural:
			return True
	return False

#
# Utilities
#

# Load WordNet
def load_wordnet():
	wordnet.ensure_loaded()
	print(f"WordNet root:         {wordnet.root}")
	print(f"WordNet version:      {(wordnet_version := wordnet.get_version())}")
	print(f"WordNet languages:    {(wordnet_langs := wordnet.langs())}")
	print(f"WordNet noun synsets: {len(list(wordnet.all_synsets(pos=wordnet.NOUN)))}")
	print(f"WordNet noun lemmas:  {len(list(wordnet.all_lemma_names(pos=wordnet.NOUN)))}")
	print(f"WordNet total words:  {len(list(wordnet.words()))}")
	assert wordnet_version == '3.1' and wordnet_langs == ['eng']
	print()

# Load plural exceptions map
def load_plurals_map():

	print(f"Loading plural noun exceptions file: {wordnet.root.join(PLURAL_NOUNS_EXC)}")

	plurals_map = {}
	singulars_map = {}
	plurals_list = []
	singulars_list = []
	with wordnet.open(PLURAL_NOUNS_EXC) as file:
		while line := file.readline():
			if line:
				nouns = line.split()
				assert all(noun for noun in nouns) and len(nouns) >= 2
				plural_noun, *singular_nouns = nouns
				plural_noun = re.sub(r'_+', ' ', plural_noun)
				plurals_list.append(plural_noun)
				singulars_list.extend(singular_nouns)
				for singular_noun in singular_nouns:
					singular_noun = re.sub(r'_+', ' ', singular_noun)
					if (plural_nouns := plurals_map.get(singular_noun, None)) is None:
						plurals_map[singular_noun] = {plural_noun}
					else:
						plural_nouns.add(plural_noun)
				if (singulars := singulars_map.get(plural_noun, None)) is None:
					singulars_map[plural_noun] = set(singular_nouns)
				else:
					singulars.update(singular_nouns)

	for singular, plurals in plurals_map.items():
		if len(plurals) > 1:
			print_info(32, f"Singular has multiple plurals: {singular} --> {', '.join(sorted(plurals))}")

	for plural, singulars in singulars_map.items():
		if len(singulars) > 1:
			print_info(34, f"Plural has multiple singulars: {plural} --> {', '.join(sorted(singulars))}")

	print(f"Num plurals:   {len(plurals_list)}")
	print(f"Num singulars: {len(singulars_list)}")
	print(f"Num unique plurals:   {(num_unique_plurals := len(set(plurals_list)))}")
	print(f"Num unique singulars: {(num_unique_singulars := len(set(singulars_list)))}")
	assert len(plurals_map) == num_unique_singulars and len(singulars_map) == num_unique_plurals
	print("Done")
	print()

	return plurals_map, singulars_map

# Load canonical gram frequency data (unigram/bigram)
def load_gram_freqs(freqs_file):
	print(f"Gram freqs TSV: {freqs_file}")
	gram_freq = {}
	with open(freqs_file, 'r') as file:
		for line in file:
			cells = [cell.strip() for cell in line.strip().split('\t')]
			assert len(cells) == 2, f"Invalid row: {line}"
			gram = cells[1]
			assert get_canon(gram) == gram, f"Non-canonical gram: {line}"
			freq = float(cells[0])
			assert 0 <= freq <= 1, f"Invalid frequency: {line}"
			assert gram not in gram_freq, f"Gram has already been seen: {line}"
			gram_freq[gram] = freq
	print(f"Loaded {len(gram_freq)} canonical gram frequencies")
	print()
	return gram_freq

# Print info
def print_info(color, msg, **kwargs):
	print(f" \033[{color}m[INFO] {msg}\033[0m", **kwargs)

# Print warning
def print_warn(msg, **kwargs):
	print(f" \033[33m[WARN] {msg}\033[0m", **kwargs)

#
# Main
#

# Run the main function
if __name__ == "__main__":
	sys.exit(1 if main(sys.argv[1:]) else 0)
# EOF
