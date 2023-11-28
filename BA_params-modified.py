# ==============================================================================
#  Copyright (c) 2022. Imen Ben Amor
# ==============================================================================
import warnings

warnings.filterwarnings("ignore")
import gc
import numpy as np
# import var_env as env
import itertools
import argparse
import logging

import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import seaborn as sns
from itertools import product
import scipy.stats as stats
import re
import itertools
from scipy.stats import norm
from Step2.preprocessing.BA_params import *
from Step2.preprocessing.load_train_param import *
from Step2.preprocessing.filters import *
from Step2.LR.lr import *
from Step2.LR.trials import *
from Step2.LR.llr_prediction import *
from Step2.LR.performance import cllr, min_cllr
from sklearn.metrics.pairwise import cosine_similarity
import shap
import pickle

utt_correspondance = {}


def number_utterances(utt):
    '''
    This function calculates for each speaker the number of utterances
    :param utt: list of utterances
    :return: dictionary of speakers and corresponding number of utterances
    '''
    speaker_dict = {}
    loc_list = []

    x = 0

    for u in utt:
        first_element_after_split = u.split("-")[0]
        if speaker_dict.get(first_element_after_split) is not None:
            speaker_dict[first_element_after_split] += 1
        else:
            speaker_dict[first_element_after_split] = 1

        x += 1

        utt_correspondance[u] = "utt" + str(x)
        loc_list.append(first_element_after_split)

    print("speaker_dict", speaker_dict)
    print("loc_list", loc_list)
    return speaker_dict, loc_list


def todelete1(xvectors):
    """
    This function deletes zero columns for all rows in array "xvectors"
    :param xvectors: array of binary xvectors
    :return: filtered array, index of deleted column
    """
    res = np.all(xvectors[..., :] == 0, axis=0)
    idx = []
    for i in range(len(res)):
        if res[i]:
            idx.append(i)
    v = np.delete(xvectors, idx, axis=1)
    return v, idx


def todelete(df, f):
    todelete = []
    for c in df[f]:
        if len(df[df[c] != 0]) == 0:
            todelete.append(c)
    df = df.drop(todelete, axis=1)
    return df


def profil_spk(xvectors, utt_per_spk, BA):
    """
    This function calculate the profile for each speaker
    :param xvectors: array of xvectors
    :param utt_per_spk: dict spk: nb_utterances
    :param BA: list of BAs
    :return:
    """
    profil = {}
    j = 0
    for spk in list(utt_per_spk.keys()):
        BA_dict = {}
        df_spk = xvectors[j:utt_per_spk[spk] + j]
        for c, ba in zip(df_spk.T, BA):
            if 1 in c:
                BA_dict[ba] = 1
            else:
                BA_dict[ba] = 0
        profil[spk] = BA_dict
        j += utt_per_spk[spk]
        # print(j)
    return profil


def compute_typicality(b, couples, profil):
    # if not couples:
    #     return 0.0
    """
    This function calculates typicality
    :param b: BAi
    :param couples: combination of all speakers in couples
    :param profil: dictionary of speakers profiles
    :return: dictionary of BAi:typ_value
    """
    nb = 0
    for (spk1, spk2) in couples:
        if spk1 != spk2 and profil[spk1][b] == 1 and profil[spk2][b] == 1:
            nb += 1
    # stat_BA[b] = nb
    typ_BA = nb / len(couples)
    # print(f"b: {b}, nb: {nb}, len(couples): {len(couples)}, typ_BA: {typ_BA}")
    return typ_BA, nb


def compute_dropout(b, profil, utt_spk, matrix_utterances, index_of_b):
    """

    :param b: BAi
    :param profil: dictionary spk: BAi:0 or 1
    :param utt_spk: dictionary spk:utt_"index_utt"
    :param matrix_utterances:
    :param index_of_b:
    :return:dropout per BAi, {spkj:x} for BAi,{spkj:dout} for BAi,number of speakers having b active
    """
    BA_spk = 0
    nb_BA_spk_b = {}
    spk_has_b_atleast_once = 0
    dropout_per_spk = {}
    for spk in utt_spk.keys():
        nb_BA = 0
        nb_present_BA = 0
        if profil[spk][b] != 0:
            spk_has_b_atleast_once += 1
            for u in utt_spk[spk]:
                index_utt = int(u[3:])
                if matrix_utterances[index_utt][index_of_b] == 0:
                    nb_BA += 1
                else:
                    nb_present_BA += 1
        nb_BA_spk_b[spk] = nb_present_BA
        BA_spk += nb_BA / len(utt_spk[spk])
        dropout_per_spk[spk] = nb_BA / len(utt_spk[spk])

    # print(spk_has_b_atleast_once)
    out = BA_spk / spk_has_b_atleast_once
    return out, nb_BA_spk_b, dropout_per_spk, spk_has_b_atleast_once


def utterance_spk(nb_utt_spk):
    """
    This function provides a dictionary of the utterance for spki
    :param nb_utt_spk: dictionary of spk:nbr of utterances
    :return: spk1:["utt0","utt1"],spk2:["utt3","utt4"]
    """
    utt_spk = {}
    j = 0
    for spk in nb_utt_spk.keys():
        nb = nb_utt_spk[spk]
        utt_spk[spk] = ["utt" + str(i) for i in range(j, j + nb)]
        j += nb
    return utt_spk


def utterance_dictionary(binary_vectors, utterances, BA):
    """
    This function gives the binary vector (using BAs) for each utterance
    :param binary_vectors: array of all binary vectors files
    :param utterances: list of utterances ids
    :param BA: list of BAs
    :return: {"id001-9fddfetl-001":{"BA0":1,"BA2":0, "BA3":1..},...}
    """
    utt = {}
    for (u, row) in zip(utterances, binary_vectors):
        utt[u] = {b: i for i, b in zip(row, BA)}
    return utt


#
# it calculates the typicality and dropout for each BA wich is the number of speakers having BA active in their utterances
# the dropout is the average of the dropout per speaker, dropout means the number of BA inactive in the utterances of a speaker
# the typicality is the number of couples having BA active in their utterances
def typicality_and_dropout(profil, couples, utt_spk, BA, vectors, typ_path, dout_path):
    """
    This function calculate the typicality and Dropout for all BAs
    :param profil: dictionary of speakers profiles
    :param couples: combination of all speakers in couples
    :param utt_spk: dictionary spk: list of utterances"index"
    :param BA:
    :param vectors: Train data binary array
    :param typ_path: path of typicality file
    :param dout_path: path of dropout file
    :return: 2 files
    """
    with open(typ_path, "w+") as file1:
        with open(dout_path, "w+") as file2:
            last_percent = -1
            nb_couples_b = {}
            typicalities = {}
            dropouts = {}
            nb_utt_spk_b = {}
            dropout_spk = {}
            nb_spk_has_BA = {}
            for index, b in enumerate(BA):
                typ, couples_active_b = compute_typicality(b, couples, profil)
                # print("\n\n_ncouples_active_b : ", couples_active_b)
                # print("\n\n\n typ : ", couples_active_b)
                nb_couples_b[b] = couples_active_b
                typicalities[b] = typ
                # typ_BA = compute_typicality2(b, utt_spk, utt, profil)
                dropout, nb_BA_spk_b, dropout_per_spk, spk_has_b = compute_dropout(b, profil, utt_spk, vectors,
                                                                                   index)
                # print("\n\n\n dropout : ", dropout)
                # print("\n\n\n nb_BA_spk_b : ", nb_BA_spk_b)
                # print("\n\n\n dropout_per_spk : ", dropout_per_spk)
                # print("\n\n\n spk_has_b : ", spk_has_b)
                # print("\n\n\n b : ", b)
                nb_spk_has_BA[b] = spk_has_b  # number of speakers per BA
                nb_utt_spk_b[b] = nb_BA_spk_b  # dict(spki:nb_BA active in utterances}
                dropout_spk[b] = dropout_per_spk  # dict(spki:dout)
                dropouts[b] = dropout

                file1.write("%s : %f " % (b, typ))
                file1.write("\n")

                file2.write("%s:%f" % (b, dropout))
                file2.write("\n")

                percent = round((index / len(BA)) * 100, 0)
                if percent % 10 == 0 and last_percent != percent:
                    logging.info(f"{percent}%")
                    last_percent = percent

        file2.close()
    file1.close()
    # print(typicalities)
    return nb_couples_b, typicalities, dropouts, nb_spk_has_BA, nb_utt_spk_b, dropout_spk


def stringToList(string):
    listRes = list(string.split(" "))
    return listRes


import re


def readVectors_test(filePath):
    vectors = []
    utt = []
    with open(filePath, "r") as f:
        lignes = f.readlines()
        line_idx = 0
        last_printed_percent = -1
        number_of_lines = len(lignes)

        for ligne in lignes:
            if ligne:
                print("ligne", ligne)
                match = re.match(r'^(\S+)\s+\[([\d\s.]+)]$', ligne)
                if match:
                    identifiant, elements_str = match.group(1), match.group(2)
                    elements = np.array([float(e) for e in elements_str.split()])
                    utt.append(identifiant)
                    vectors.append(elements)

                    # Afficher la progression
                    line_idx += 1
                    percent = int((line_idx / number_of_lines) * 100)
                    if percent % 10 == 0 and percent != last_printed_percent:
                        print(f"Progression : {percent}%")
                        last_printed_percent = percent
                else:
                    print(f"Erreur à la ligne {line_idx} : {ligne}")

    return utt, np.array(vectors)


def readVectors(filePath):
    vectors = []
    utt = []
    with open(filePath, "r") as f:
        line_idx = 0
        last_printed_percent = -1
        number_of_lines = 5105875
        for line in f:
            line_idx += 1
            elems = line.split("  ")
            vec = []
            utt.append(elems[0])
            for elem in stringToList(elems[1][2:-2].rstrip()):
                value_to_append = 1 if (round(float(elem), 4) != 0) else 0
                vec.append(value_to_append)
            vectors.append(vec)
            percent = round(line_idx / number_of_lines * 100, 0)
            if percent % 10 == 0 and percent != last_printed_percent:
                print(f"{percent}%")
                last_printed_percent = percent
    return utt, np.array(vectors)


def find_id_by_utt_in_utt_correspondance(utt):
    for key, value in utt_correspondance.items():
        if value == utt:
            return key


# def predict if 2 utterances utt1 and utt2 are from the same speaker with the LLR score, return label and probability
def predict(utt1, utt2):
    couple = (utt1, utt2)
    couple_index = tar.index(couple)
    llr_target_for_couple = LLR_target[couple_index]
    llr_non_for_couple = LLR_non[couple_index]
    # print("llr_target_for_couple", llr_target_for_couple)
    # print("llr_non_for_couple", llr_non_for_couple)
    if llr_target_for_couple > llr_non_for_couple:
        return 1
    else:
        return 0


import random


def do_edit(vector):
    indices_to_modify = random.sample(range(0, len(vector)), 3)
    for i in indices_to_modify:
        if vector[i] == 0:
            vector[i] = 1
        else:
            vector[i] = 0
    return vector


def generate_counterfactual(utt, couple, dout, typ, tar, non):
    llr_target_for_couple = 100000
    llr_non_for_couple = 0
    while llr_target_for_couple > llr_non_for_couple:
        vector1 = np.array(list(utt[couple[0]].values()))
        vector2 = np.array(list(utt[couple[1]].values()))

        new_vector1 = do_edit(vector1)
        new_vector2 = do_edit(vector2)

        for key, value in utt.items():
            test = np.array(list(value.values()))
            if key == couple[0]:
                print(utt[key])
                utt[key] = dict(zip(utt[key].keys(), new_vector1))
                print(utt[key])
            if key == couple[1]:
                utt[key] = dict(zip(utt[key].keys(), new_vector2))
        LLR_target, LLR_non, list_eer, list_cllr_min, list_cllr_act, list_Din = LR_framework(dout, typ, utt, tar, non,
                                                                                             [0.12])
        couple_index = tar.index(couple)
        llr_target_for_couple = LLR_target[couple_index]
        llr_non_for_couple = LLR_non[couple_index]

        print("llr_target_for_couple_counterfactual", llr_target_for_couple)
        print("llr_non_for_couple_counterfactual", llr_non_for_couple)

        print("new predict", predict(couple[0], couple[1]))


if __name__ == "__main__":
    # Arguments
    # env.logging_config(env.PATH_LOGS + "/logFile")
    parse = argparse.ArgumentParser()
    parse.add_argument("--path", default="./vox1-transformed.txt", type=str)
    parse.add_argument("--typ_path", default="./data/typ_j.txt", type=str)
    parse.add_argument("--dout_path", default="./data/dout_j.txt", type=str)
    args = parse.parse_args()
    logging.info("read xvectors")
    utterances, binary = readVectors_test(args.path)

    logging.info("finish reading xvectors")
    logging.info("xvectors array ready")
    utt_per_spk, loc_list = number_utterances(utterances)
    logging.info("delete zero columns...")

    binary_vectors, idx = todelete1(binary)

    logging.info(f"number of deleted columns: {len(idx)}")
    BA = ['BA' + str(i) for i in range(binary.shape[1]) if np.array([i]) not in idx]
    # liberate memory
    # del binary
    # del loc_list
    # del idx
    gc.collect()
    logging.info("utterance_spk...")
    utt_spk = utterance_spk(utt_per_spk)
    logging.info("profil_spk...")
    profil = profil_spk(binary_vectors, utt_per_spk, BA)
    # speakers couples
    logging.info("computing combinations...")
    couples = list(itertools.combinations(utt_per_spk.keys(), 2))
    # print(couples)
    typicality_and_dropout(profil, couples, utt_spk, BA, binary_vectors, args.typ_path, args.dout_path)

    # BA = ['BA' + str(i) for i in range(binary.shape[1])]
    df = pd.DataFrame(binary_vectors, columns=BA)
    df = todelete(df, BA)
    df

    typ, dout = load_filter_soft(args.typ_path, args.dout_path)
    df = df[list(typ.keys())]
    BA_test = list(typ.keys())

    utt = {}
    for (idx, row) in df.iterrows():
        utt[f"utt{idx}"] = dict(row)

    # print(utt_correspondance)

    ## Write target and non files 

    with open("./trials_vox1.txt", "r") as file:
        lines = file.readlines()

    with open("./target.txt", "w+") as target:
        with open("./non.txt", "w+") as non:

            for line in lines:
                first_utt = utt_correspondance[line.split()[1]]
                second_utt = utt_correspondance[line.split()[2]]

                if line.split()[0] == '1':
                    target.write("('" + first_utt + "', '" + second_utt + "')\n")
                elif line.split()[0] == '0':
                    non.write("('" + first_utt + "', '" + second_utt + "')\n")
                else:
                    print("PROBLEM!")

    non, tar = load_trials()
    # print("dout : ", dout)
    LLR_target, LLR_non, list_eer, list_cllr_min, list_cllr_act, list_Din = LR_framework(dout, typ, utt, tar, non,
                                                                                         [0.12])
    # plt.show()
    # Assuming you have a specific couple
    # couple_to_print = ('utt124', 'utt126')
    couple_to_print = ('utt17', 'utt20')
    couple_of_id = (
        find_id_by_utt_in_utt_correspondance(couple_to_print[0]),
        find_id_by_utt_in_utt_correspondance(couple_to_print[1]))
    print("couple_of_id", couple_of_id)

    print("BA " + couple_to_print[0], utt[couple_to_print[0]])
    print("BA " + couple_to_print[1], np.array(utt[couple_to_print[1]].values()))

    # couple_to_print_non = ('utt264', 'utt226')

    # Find the index of the couple in the 'tar' list
    couple_index = tar.index(couple_to_print)

    # couple_index_non = non.index(couple_to_print_non)

    # Access the LLR scores for the specified couple in 'LLR_target'
    llr_target_for_couple = LLR_target[couple_index]
    llr_non_for_couple = LLR_non[couple_index]

    print(predict(couple_to_print[0], couple_to_print[1]))

    # llr_target_for_couple_non = LLR_target[couple_index_non]
    # llr_non_for_couple_non = LLR_non[couple_index_non]

    # define below function to generate the counterfactual method by minimizing the distance between the couple and the counterfactual
    # it takes utt, couple, and random modifies the utterance of the couple to generate a counterfactual and pass it to LR framework

    generate_counterfactual(utt, couple_to_print, dout, typ, tar, non)

    # Print the LLR scores
    print(f"LLR target for couple {couple_to_print}: {llr_target_for_couple}")
    print(f"LLR non for couple {couple_to_print}: {llr_non_for_couple}")
