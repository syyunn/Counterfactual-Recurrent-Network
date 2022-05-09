# Copyright (c) 2020, Ioana Bica

import numpy as np
import pandas as pd

from CRN_model import CRN_Model

import pickle

from scipy.ndimage.interpolation import shift



def write_results_to_file(filename, data):
    with open(filename, "wb") as handle:
        pickle.dump(data, handle, protocol=2)


def append_results_to_file(filename, data):
    with open(filename, "a+b") as handle:
        pickle.dump(data, handle, protocol=2)


def load_trained_model(
    dataset_test, hyperparams_file, model_name, model_folder, b_decoder_model=False
):
    _, length, num_covariates = dataset_test["current_covariates"].shape
    num_treatments = dataset_test["current_treatments"].shape[-1]
    num_outputs = dataset_test["outputs"].shape[-1]

    params = {
        "num_treatments": num_treatments,
        "num_covariates": num_covariates,
        "num_outputs": num_outputs,
        "max_sequence_length": length,
        "num_epochs": 100,

    }

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, "rb") as handle:
        best_hyperparams = pickle.load(handle)

    model = CRN_Model(params, best_hyperparams)
    if b_decoder_model:
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)

    model.load_model(model_name=model_name, model_folder=model_folder)
    return model


def get_processed_data(raw_sim_data, scaling_params):
    """
    Create formatted data to train both encoder and seq2seq atchitecture.
    """
    mean, std = scaling_params

    horizon = 1  # for the encoder, we perform "one-step ahead prediction"
    offset = 1  # to do one-step ahead, remove the final one

    # horizon = 8  # for the encoder, we perform "one-step ahead prediction"
    # offset = 8  # to do one-step ahead, remove the final one

    # ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    input_means = mean[
        ["niq_adj_vol", "atq_adj", "niq_adj", "revtq_adj", "mkvaltq_adj", "emp", "PRisk", "timecode", "naics", "amount_bool"] #, "amount"]
    ].values.flatten()  # with time code, you can perfectly predict treatment

    input_stds = std[
        ["niq_adj_vol", "atq_adj", "niq_adj", "revtq_adj", "mkvaltq_adj", "emp", "PRisk", "timecode", "naics", "amount_bool"] #, "amount"]
    ].values.flatten()  # with time code, you can perfectly predict treatment

    # Continuous values
    niq_adj_vol = (raw_sim_data["niq_adj_vol"] - mean["niq_adj_vol"]) / std[
        "niq_adj_vol"
    ]

    atq_adj = (raw_sim_data["atq_adj"] - mean["atq_adj"]) / std[
        "atq_adj"
    ]

    niq_adj = (raw_sim_data["niq_adj"] - mean["niq_adj"]) / std[
        "niq_adj"
    ]

    revtq_adj = (raw_sim_data["revtq_adj"] - mean["revtq_adj"]) / std[
        "revtq_adj"
    ]

    mkvaltq_adj = (raw_sim_data["mkvaltq_adj"] - mean["mkvaltq_adj"]) / std[
        "mkvaltq_adj"
    ]

    emp = (raw_sim_data["emp"] - mean["emp"]) / std[
        "emp"
    ]

    PRisk = (raw_sim_data["PRisk"] - mean["PRisk"]) / std[
        "PRisk"
    ]

    timecode = (raw_sim_data["timecode"] - mean["timecode"]) / std[
        "timecode"
    ]

    # amount = (raw_sim_data["amount"] - mean["amount"]) / std[
    #     "amount"
    # ]

    # Static values
    naics = (raw_sim_data["naics"] - mean["naics"]) / std[
        "naics"
    ]

    naics = np.stack(
        [naics for t in range(niq_adj_vol.shape[1])], axis=1
    )

    # Binary application
    amount_bool = raw_sim_data["amount_bool"]
    sequence_lengths = raw_sim_data["sequence_lengths"]

    # Convert treatments to one-hot encoding
    # treatments = np.concatenate(
    #     [
    #         amount[:, :-offset, np.newaxis],  # transpose.
    #     ],
    #     axis=-1,
    # )

    treatments = np.concatenate(
        [
            amount_bool[:, :-offset, np.newaxis],  # transpose.
        ],
        axis=-1,
    )

    one_hot_treatments = np.zeros(
        shape=(treatments.shape[0], treatments.shape[1], 1)
    )  # this 4 means dimension of one-hot vector.

    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (
                treatments[patient_id][timestep][0] == 0
            ):
                one_hot_treatments[patient_id][timestep] = [0]
            elif (
                treatments[patient_id][timestep][0] == 1
            ):
                one_hot_treatments[patient_id][timestep] = [1]

    # for patient_id in range(treatments.shape[0]):
    #     for timestep in range(treatments.shape[1]):
    #         one_hot_treatments[patient_id][timestep] = [treatments[patient_id][timestep][0]]
    one_hot_previous_treatments = np.roll(one_hot_treatments, 1, axis=1)
    one_hot_previous_treatments[:, 0, :] = 0
    one_hot_previous_treatments = one_hot_previous_treatments[:, :one_hot_treatments.shape[1]-1, :]

    # one_hot_treatments[np.isnan(one_hot_treatments)] = 0 # just make sure

    # one_hot_previous_treatments = one_hot_treatments[:, 1:, :]

    # covariates are only volume and type (two) in this case

    current_covariates = np.concatenate(
        [
            niq_adj_vol[:, :-offset, np.newaxis],
            atq_adj[:, :-offset, np.newaxis],
            niq_adj[:, :-offset, np.newaxis],
            revtq_adj[:, :-offset, np.newaxis],
            mkvaltq_adj[:, :-offset, np.newaxis],
            emp[:, :-offset, np.newaxis],
            PRisk[:, :-offset, np.newaxis],
            # timecode[:, :-offset, np.newaxis],
            naics[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )
    outputs = niq_adj_vol[:, horizon:, np.newaxis]  # volume is y in this case.

    output_means = mean[["niq_adj_vol"]].values.flatten()[
        0
    ]  # because we only need scalars here
    output_stds = std[["niq_adj_vol"]].values.flatten()[0]

    print(outputs.shape)

    # let's make this from prep.py
    active_entries = raw_sim_data["active_entries"]
    active_entries = active_entries[:, horizon:, np.newaxis]  # adjust as same as output - because we use this for loss computation, which copares with output.

    # Add active entries
    # active_entries = np.zeros(outputs.shape)
    # for i in range(sequence_lengths.shape[0]):  # number of different patient ids
    #     sequence_length = int(sequence_lengths[i])
    #     active_entries[
    #         i, :sequence_length, :
    #     ] = 1  # in this way, we manage the varying length of each patients.

    raw_sim_data["current_covariates"] = current_covariates
    raw_sim_data["previous_treatments"] = one_hot_previous_treatments
    raw_sim_data["current_treatments"] = one_hot_treatments
    raw_sim_data["outputs"] = outputs
    raw_sim_data["active_entries"] = active_entries

    raw_sim_data["unscaled_outputs"] = (
        outputs * std["niq_adj_vol"] + mean["niq_adj_vol"]
    )
    raw_sim_data["input_means"] = input_means
    raw_sim_data["inputs_stds"] = input_stds
    raw_sim_data["output_means"] = output_means
    raw_sim_data["output_stds"] = output_stds

    return raw_sim_data


def get_mse_at_follow_up_time(mean, output, active_entires):
    mses = np.sum(
        np.sum((mean - output) ** 2 * active_entires, axis=-1), axis=0
    ) / active_entires.sum(axis=0).sum(axis=-1)

    return pd.Series(mses, index=[idx for idx in range(len(mses))])


def train_BR_optimal_model(
    dataset_train,
    dataset_val,
    hyperparams_file,
    model_name,
    model_folder,
    b_decoder_model=False,
):
    _, length, num_covariates = dataset_train["current_covariates"].shape
    num_treatments = dataset_train["current_treatments"].shape[-1]
    num_outputs = dataset_train["outputs"].shape[-1]

    params = {
        "num_treatments": num_treatments,
        "num_covariates": num_covariates,
        "num_outputs": num_outputs,
        "max_sequence_length": length,
        "num_epochs": 100,
    }

    print("Loading best hyperparameters for model")
    with open(hyperparams_file, "rb") as handle:
        best_hyperparams = pickle.load(handle)

    print("Best Hyperparameters")
    print(best_hyperparams)

    if b_decoder_model:
        print(best_hyperparams)
        model = CRN_Model(params, best_hyperparams, b_train_decoder=True)
    else:
        model = CRN_Model(params, best_hyperparams)
    model.train(
        dataset_train, dataset_val, model_name=model_name, model_folder=model_folder
    )
