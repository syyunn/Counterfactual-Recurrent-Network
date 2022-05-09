import pickle

with open('../plot/training_processed.pickle', "rb") as handle:
    training_processed = pickle.load(handle)
    sequence_lengths = list(training_processed['sequence_lengths'])

with open('./predictions.pickle', "rb") as handle:
    ones = pickle.load(handle)
    ones = ones.reshape((-1, 2, 1))
    pass
    # ones = ones[:,:,0]

seqlen_total = 0
results = []
for seqlen in sequence_lengths:
    results.append(list(ones[int(seqlen_total): int(seqlen_total+seqlen)-1]))
    seqlen_total += len(list(ones[int(seqlen_total): int(seqlen_total+seqlen)-1]))


if __name__ == "__main__":
    pass
