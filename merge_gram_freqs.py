#!/usr/bin/env python3

# Imports
import sys
import operator
import unidecode

# Main function
def main(path1w, path2w, outpath):

	print(f"Monogram path: {path1w}")
	print(f"Bigram path: {path2w}")
	print()

	gram_freq = {}

	print("Loading monograms...")
	raw_chars, chars = load_grams(path1w, gram_freq)
	print(f"Monogram chars raw: \"{''.join(sorted(raw_chars))}\"")
	print(f"Monogram chars:     \"{''.join(sorted(chars))}\"")
	print("Done")
	print()

	print("Loading bigrams...")
	raw_chars, chars = load_grams(path2w, gram_freq)
	print(f"Bigram chars raw: \"{''.join(sorted(raw_chars))}\"")
	print(f"Bigram chars:     \"{''.join(sorted(chars))}\"")
	print("Done")
	print()

	print(f"Saving merged normalised gram freqs: {outpath}")
	the_freq = gram_freq['the']
	with open(outpath, 'w') as file:
		for gram, freq in sorted(gram_freq.items(), key=operator.itemgetter(1), reverse=True):
			print(f"{freq / the_freq:.12f}\t{gram}", file=file)
	print("Done")
	print()

# Load grams
def load_grams(path, gram_freq):
	chars = set()
	raw_chars = set()
	num_stags = 0
	with open(path, 'r') as file:
		for line in file:
			cells = [cell.strip() for cell in line.strip().split('\t')]
			assert len(cells) == 2, f"Invalid row: {line}"
			raw_gram = cells[0]
			if '<S>' in raw_gram.split():
				num_stags += 1
				continue
			raw_count = int(cells[1])
			raw_chars.update(set(raw_gram))
			gram = ' '.join(unidecode.unidecode(raw_gram).lower().split())
			if gram != raw_gram.lower():
				print_info(34, f"Special replace: {raw_gram} --> {gram}")
			if any(not ord('a') <= ord(char) <= ord('z') and char != ' ' for char in gram):
				print_info(35, f"Non-[ a-z] character: {gram}")
			chars.update(set(gram))
			gram_freq[gram] = gram_freq.get(gram, 0) + raw_count
	print(f"Ignored {num_stags} <S> tags")
	return raw_chars, chars

# Print info
def print_info(color, msg, **kwargs):
	print(f" \033[{color}m[INFO] {msg}\033[0m", **kwargs)

# Run the main function
if __name__ == "__main__":
	sys.exit(1 if main(*sys.argv[1:]) else 0)
# EOF
