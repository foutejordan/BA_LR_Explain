# Description: This file contains the implementation of the counterfactual method 2

import warnings

warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")
from Step2.preprocessing.BA_params import *
from Step2.preprocessing.load_train_param import *
from Step2.LR.trials import *
from Step2.LR.llr_prediction import *
import copy

from itertools import combinations

def visualize_llr_framework(llr_framework):
    LLR_target = llr_framework["LLR_target"]
    LLR_non = llr_framework["LLR_non"]
    list_eer = llr_framework["list_eer"]
    list_cllr_min = llr_framework["list_cllr_min"]
    list_cllr_act = llr_framework["list_cllr_act"]
    list_Din = llr_framework["list_Din"]

    # Plot LLR target and non-target scores
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.plot(LLR_target)
    plt.title("LLR Target")

    plt.subplot(2, 3, 2)
    plt.plot(LLR_non)
    plt.title("LLR Non-Target")

    plt.subplot(2, 3, 3)
    plt.plot(list_eer)
    plt.title("EER")

    plt.subplot(2, 3, 4)
    plt.plot(list_cllr_min)
    plt.title("Cllr Min")

    plt.subplot(2, 3, 5)
    plt.plot(list_cllr_act)
    plt.title("Cllr Actual")

    plt.subplot(2, 3, 6)
    plt.plot(list_Din)
    plt.title("D-in")

    plt.tight_layout()
    plt.show()


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
            
        x+=1
            
        utt_correspondance[u] = "utt"+str(x)
        loc_list.append(first_element_after_split)
        
    # print("speaker_dict", speaker_dict)
    # print("loc_list", loc_list)
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
    #print(f"b: {b}, nb: {nb}, len(couples): {len(couples)}, typ_BA: {typ_BA}")
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

    #print(spk_has_b_atleast_once)
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
    #print(typicalities)
    return nb_couples_b, typicalities, dropouts, nb_spk_has_BA, nb_utt_spk_b, dropout_spk


def stringToList(string):
    listRes = list(string.split(" "))
    return listRes


import re

vectors_by_id = {}

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
                #print("ligne", ligne)
                match = re.match(r'^(\S+)\s+\[([\d\s.]+)]$', ligne)
                if match:
                    identifiant, elements_str = match.group(1), match.group(2)
                    elements = np.array([float(e) for e in elements_str.split()])
                    utt.append(identifiant)
                    vectors.append(elements)
                    vectors_by_id[identifiant] = elements

                    # Afficher la progression
                    line_idx += 1
                    percent = int((line_idx / number_of_lines) * 100)
                    if percent % 10 == 0 and percent != last_printed_percent:
                        print(f"Progression : {percent}%")
                        last_printed_percent = percent
                else:
                    print(f"Erreur à la ligne {line_idx} : {ligne}")

    return utt, np.array(vectors)

def hamming_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")

    distance = 0
    for bit1, bit2 in zip(vector1, vector2):
        if bit1 != bit2:
            distance += 1

    return distance

def generate_all_modifications(binary_vector, n):
    """
    Generate all binary vectors with exactly n bits flipped from the input binary vector.
    """
    length = len(binary_vector)
    
    # Generate all combinations of indices to flip
    indices_to_flip_combinations = combinations(range(length), n)

    # Initialize a list to store all modified vectors
    modified_vectors = []

    # Iterate through each combination of indices
    for indices_to_flip in indices_to_flip_combinations:
        # Create a copy of the original binary vector
        modified_vector = binary_vector.copy()

        # Flip the bits at the specified indices
        for index in indices_to_flip:
            modified_vector[index] = 1 - modified_vector[index]

        # Append the modified vector to the list
        modified_vectors.append(modified_vector)

    return modified_vectors


def modify_vector(modification_vector, vector_to_modify):
    if len(modification_vector) != len(vector_to_modify):
        print("ERROR!")
    else:
        modified_vector = copy.deepcopy(vector_to_modify)
        for i in range(len(modification_vector)):
            if modification_vector[i] == 1:
                if modified_vector[i] == 0:
                    modified_vector[i] = 1
                else:
                    modified_vector[i] = 0
    return modified_vector


def random_modification(vector, n, vector_size):
    for i in range(1, n+1):
        random = np.random.randint(vector_size)
        if vector[i] == 0:
            vector[i] = 1
        else:
            vector[i] = 0
    return vector

def find_keys_by_value(dictionary, target_value):
    """
    Find key(s) in the dictionary with the specified value.
    """
    keys_found = []
    for key, value in dictionary.items():
        if value == target_value:
            keys_found.append(key)
    return keys_found

if __name__ == "__main__":
    
    best_llr_mean = 1000
    VECTORS_SIZE = 206
    last_round = False
    zero_vector = [0] * VECTORS_SIZE
    
    utt_correspondance = {}
    target_column_1 = []
    non_utts = []
    target_utts = []
    # Arguments
    # env.logging_config(env.PATH_LOGS + "/logFile")
    parse = argparse.ArgumentParser()
    parse.add_argument("--path", default="./vox1-transformed.txt", type=str)
    parse.add_argument("--typ_path", default="data/typ_j.txt", type=str)
    parse.add_argument("--dout_path", default="data/dout_j.txt", type=str)
    args = parse.parse_args()
    
    utterances, binary = readVectors_test(args.path)
    
    utt_per_spk, loc_list = number_utterances(utterances)
    
    with open("./trials_vox1.txt", "r") as file:
                    lines = file.readlines()       
                    
    with open("./target.txt", "w+") as target:
        with open("./non.txt", "w+") as non:
        
            for line in lines:
                first_utt = utt_correspondance[line.split()[1]]
                second_utt = utt_correspondance[line.split()[2]]                
                
                if line.split()[0] == '1':
                    target.write("('"+first_utt+"', '"+second_utt+"')\n")
                    target_column_1.append(first_utt)
                    target_utts.append(first_utt)
                    target_utts.append(second_utt)
                elif line.split()[0] == '0':
                    non.write("('"+first_utt+"', '"+second_utt+"')\n")
                    non_utts.append(first_utt)
                    non_utts.append(second_utt)
                else:
                    print("PROBLEM!")
                
    utt_spk = utterance_spk(utt_per_spk)
    typ,dout=load_filter_soft(args.typ_path, args.dout_path)
    
    non,tar= load_trials()
    vectors = []

    for utt in target_utts:
        vector = vectors_by_id[find_keys_by_value(utt_correspondance, utt)[0]]
        vectors.append(vector)

    for utt in non_utts:
        vector = vectors_by_id[find_keys_by_value(utt_correspondance, utt)[0]]
        vectors.append(vector)
             
    vectors_array = np.array(binary)
    
    first_iteration = True
                    
    for modification_complexity in range(1, VECTORS_SIZE):
                
        if last_round == False : 
            
            modification_vectors = generate_all_modifications(zero_vector, modification_complexity)
        
            for modification_vector in modification_vectors:
                
                for i in range(len(target_utts)):
                    if i % 2 == 0:
                        for j in range(206):
                            if modification_vector[j] == 1:
                                vectors_array[i, j] = 1 - vectors_array[i, j]

                binary_vectors, idx = todelete1(vectors_array)

                logging.info(f"number of deleted columns: {len(idx)}")
                BA = ['BA' + str(i) for i in range(vectors_array.shape[1]) if np.array([i]) not in idx]
                
                profil = profil_spk(binary_vectors, utt_per_spk, BA)
                # speakers couples
                
                couples = list(itertools.combinations(utt_per_spk.keys(), 2))
                typicality_and_dropout(profil, couples, utt_spk, BA, binary_vectors, args.typ_path, args.dout_path)
                
                del couples

                #BA = ['BA' + str(i) for i in range(binary.shape[1])]
                df = pd.DataFrame(binary_vectors, columns=BA)
                df=todelete(df,BA)
                df
                
                df=df[list(typ.keys())]
                BA_test=list(typ.keys())

                utt={}
                for (idx,row) in df.iterrows():
                    utt[f"utt{idx}"]=dict(row)
                
                if first_iteration:
                    LLR_target, original_LLR_non, *_ =LR_framework(dout,typ,utt,tar,non,[0.12])
                else : 
                    LLR_target, *_ =LR_framework(dout,typ,utt,tar,non,[0.12])
                    
                first_iteration = False

                if np.mean(LLR_target) < best_llr_mean:
                    print("Modification complexity : ", modification_complexity)
                   
                    best_llr_mean = np.mean(LLR_target)
                    print(best_llr_mean)
                    best_llr_framework = {
                            "LLR_target": LLR_target,
                            "LLR_non": original_LLR_non
                        }
                    
                    best_modification_vector = modification_vectors[i]
                    best_llr_framework_figure_number = plt.gcf().number
                    
                    LLR_target_array = np.array(LLR_target)
                    #if np.mean(LLR_target_array < 180) >= 0.75:
                    if (sum(value < 180 for value in LLR_target) / len(LLR_target)) >= 0.7:
                        last_round = True
                        
                    del LLR_target_array
                
                # liberate memory
                del binary_vectors, df, utt, idx, profil, LLR_target
                gc.collect()
                
                for i in range(len(target_utts)):
                    if i % 2 == 0:
                        for j in range(206):
                            if modification_vector[j] == 1:
                                vectors_array[i, j] = 1 - vectors_array[i, j]
                
            if last_round == True:
                print("Last Round")
                    
            del modification_vectors
                        
            gc.collect()   
            
            for num in plt.get_fignums():
                if num != best_llr_framework_figure_number:
                    plt.close(num)
                    
        print("Best modification vector : ", best_modification_vector)
        print("Indexs des bits modifiés : ")
        for index in range(len(best_modification_vector)):
            bit = best_modification_vector[index]
            if bit == 1:
                print("Index : ", index)
        print("Best LLR_Mean : ", best_llr_mean)
        
    plt.show()
    
    