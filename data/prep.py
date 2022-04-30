import pandas as pd
import pickle
import numpy as np
import random

pickle_map_lv = dict()

pickle_map = pickle.load(open("../results/new_cancer_sim_2_2.p", "rb"))

lv = pd.read_csv("./lv.csv")

qtr = lv["qtr"]

gvkeys = list(set(lv["gvkey"]))
gvkeys.sort()  # ascending
num_gvkeys = len(gvkeys)
num_time_steps = int((max(lv["qtr"]) - min(lv["qtr"])) * 4 + 1)

# y
niq_adj_vol = np.zeros((num_gvkeys, num_time_steps))

# a
amount = np.zeros((num_gvkeys, num_time_steps))
amount_bool = np.zeros((num_gvkeys, num_time_steps))

# x
atq_adj = np.zeros((num_gvkeys, num_time_steps))
niq_adj = np.zeros((num_gvkeys, num_time_steps))
revtq_adj = np.zeros((num_gvkeys, num_time_steps))
emp = np.zeros((num_gvkeys, num_time_steps))
mkvaltq_adj = np.zeros((num_gvkeys, num_time_steps))
PRisk = np.zeros((num_gvkeys, num_time_steps))
timecode = np.zeros((num_gvkeys, num_time_steps))

# v
sequence_lengths = np.zeros((num_gvkeys,))
naics = np.zeros((num_gvkeys,))

min_qtr = min(lv["qtr"])

rev_dict = dict()
for gvkey_idx, gvkey in enumerate(gvkeys):
    rev_dict[gvkey] = gvkey_idx
    subset = lv[lv["gvkey"] == gvkey]
    subset = subset.reset_index()
    subset_len = int(len(subset))
    print(gvkey_idx)
    sequence_lengths[gvkey_idx] = subset_len  # this is used for finding active values.
    for index, row in subset.iterrows():
        ts = int((row["qtr"] - min_qtr) * 4)
        # y
        niq_adj_vol[gvkey_idx, ts] = row["niq_adj_vol"]
        # niq_adj_vol[gvkey_idx, ts + 1] = subset.loc[index + 1, "niq_adj_vol"]
        # v
        naics[gvkey_idx] = int(row["naics"])
        # x
        atq_adj[gvkey_idx, ts] = row["atq_adj"]
        niq_adj[gvkey_idx, ts] = row["niq_adj"]
        revtq_adj[gvkey_idx, ts] = row["revtq_adj"]
        emp[gvkey_idx, ts] = row["emp"]
        mkvaltq_adj[gvkey_idx, ts] = row["mkvaltq_adj"]
        PRisk[gvkey_idx, ts] = row["PRisk"]
        timecode[gvkey_idx, ts] = row["qtr"]
        # a
        amount[gvkey_idx, ts] = row["amount"]
        if amount[gvkey_idx, ts] > 0:
            amount_bool[gvkey_idx, ts] = 1
        pass

random.seed(17800)
random.shuffle(gvkeys)
num_train = int(np.floor(len(gvkeys) * 0.8))
num_valid = int(np.floor(len(gvkeys) * 0.1))
num_test = int(len(gvkeys) - (num_train + num_valid))

train_gvkeys = gvkeys[:num_train]
valid_gvkeys = gvkeys[num_train : num_train + num_valid]
test_gvkeys = gvkeys[num_train + num_valid :]

training_data = dict()
validation_data = dict()
test_data = dict()

# --- training data
mask = [rev_dict[gvkey] for gvkey in train_gvkeys]
# y
training_data["niq_adj_vol"] = niq_adj_vol[mask, :]
# v
training_data["naics"] = naics[mask]
# x
training_data["atq_adj"] = atq_adj[mask, :]
training_data["niq_adj"] = niq_adj[mask, :]
training_data["revtq_adj"] = revtq_adj[mask, :]
training_data["mkvaltq_adj"] = mkvaltq_adj[mask, :]
training_data["emp"] = emp[mask, :]
training_data["PRisk"] = PRisk[mask, :]
training_data["timecode"] = timecode[mask, :]
# a
training_data["sequence_lengths"] = sequence_lengths[mask]
training_data["amount"] = amount[mask, :]
training_data["amount_bool"] = amount_bool[mask, :]

# --- validation data
mask = [rev_dict[gvkey] for gvkey in valid_gvkeys]
# y
validation_data["niq_adj_vol"] = niq_adj_vol[mask, :]
# v
validation_data["naics"] = naics[mask]
# x
validation_data["atq_adj"] = atq_adj[mask, :]
validation_data["niq_adj"] = niq_adj[mask, :]
validation_data["revtq_adj"] = revtq_adj[mask, :]
validation_data["mkvaltq_adj"] = mkvaltq_adj[mask, :]
validation_data["emp"] = emp[mask, :]
validation_data["PRisk"] = PRisk[mask, :]
validation_data["timecode"] = timecode[mask, :]
# a
validation_data["sequence_lengths"] = sequence_lengths[mask]
validation_data["amount"] = amount[mask, :]
validation_data["amount_bool"] = amount_bool[mask, :]

# --- test data
mask = [rev_dict[gvkey] for gvkey in test_gvkeys]
# y
test_data["niq_adj_vol"] = niq_adj_vol[mask, :]
# v
test_data["naics"] = naics[mask]
# x
test_data["atq_adj"] = atq_adj[mask, :]
test_data["niq_adj"] = niq_adj[mask, :]
test_data["revtq_adj"] = revtq_adj[mask, :]
test_data["mkvaltq_adj"] = mkvaltq_adj[mask, :]
test_data["emp"] = emp[mask, :]
test_data["PRisk"] = PRisk[mask, :]
test_data["timecode"] = timecode[mask, :]
# a
test_data["sequence_lengths"] = sequence_lengths[mask]
test_data["amount"] = amount[mask, :]
test_data["amount_bool"] = amount_bool[mask, :]

pickle_map_lv["training_data"] = training_data
pickle_map_lv["validation_data"] = validation_data
pickle_map_lv["test_data"] = test_data

with open("./pickle_map_lv", "wb") as handle:
    pickle.dump(pickle_map_lv, handle, protocol=2)

if __name__ == "__main__":
    pass
