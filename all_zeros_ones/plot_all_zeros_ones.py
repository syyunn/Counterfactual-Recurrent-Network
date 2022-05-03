import pickle

with open('./all_ones.pickle_pre', "rb") as handle:
    ones = pickle.load(handle)
    ones = ones[:,:,0]

with open('./all_ones.pickle_pre', "rb") as handle:
    zeros = pickle.load(handle)
    zeros = zeros[:,:,0]

if __name__ == "__main__":
    pass
