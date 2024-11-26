from multiprocessing import Pool, cpu_count


def _generate_ngrams(tokens, n):
    """
    Generate n-grams for a specific n
    """
    return [tokens[i : i + n] for i in range(len(tokens) - n + 1)]


def get_ngrams_parallel(tokens, ngram_range):
    """
    Fully parallelize the generation of n-grams
    """
    # Parallelize the outer loop (over n values)
    with Pool(cpu_count()) as pool:
        # Generate n-grams for each n in parallel
        ngram_results = pool.starmap(
            _generate_ngrams,
            [(tokens, n) for n in range(ngram_range[0], ngram_range[1] + 1)],
        )

    # Flatten the list of lists into a single list
    return [tuple(ngram) for ngram_list in ngram_results for ngram in ngram_list]
