import pandas as pd

def count_sets_overlap(sequences, names=None, exclusive=False):
    """Generate a NxN table of overlap counts, where `N=len(sequences)`.

    Args:
        sequences (iterable, list): A list of lists, sets, or pd.Series that contain
            items that need to be counted.
        names (_type_, optional): If provided, the corresponding element in the output
            count table will be named as such. Otherwise, sequences are indexed from `0` to
            `N`. If a given sequence has a `name` attribute, then that name will be 
            prioritized. Defaults to None.
        exclusive (bool, optional): Whether exclusive overlaps should be counted. 
            Defaults to False.

    Returns:
        Pandas.DataFrame: A dataframe of overlap counts, where columns and indices are
            named according to the given sequences.
    """
    n_seqs = len(sequences)
    sets = [set(seq) for seq in sequences]
    if names is None:
        names = [seq.name if hasattr(seq, 'name') else i for (i, seq) in enumerate(sequences)]

    overlaps = pd.DataFrame(columns=names, index=names)
    for i in range(n_seqs):
        for j in range(n_seqs):
            if exclusive:
                overlaps.iat[i, j] = len(sets[i] - sets[j])
            else:
                overlaps.iat[i, j] = len(sets[i] & sets[j])
    return overlaps
