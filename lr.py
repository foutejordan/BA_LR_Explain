
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
path="/home/maax/Documents/Mega Sync/Cours M2/Explicabilité/BA_LR_Explained/vox1-transformed.txt"

typ_path="/home/maax/Documents/Mega Sync/Cours M2/Explicabilité/BA_LR_Explained/data/typ.txt"
dout_path="/home/maax/Documents/Mega Sync/Cours M2/Explicabilité/BA_LR_Explained/data/dout.txt"

utterances, vectors = readVectors_test(path)
xvectors = np.array(vectors).astype('float64')
#print(xvectors.shape)
BA = ['BA' + str(i) for i in range(xvectors.shape[1])]
df = pd.DataFrame(xvectors, columns=BA)
df=todelete(df,BA)
df

utt_per_spk, loc_list = number_utterances(utterances)
utt_per_spk

utt_spk = utterance_spk(utt_per_spk)
utt_spk

profil = profil_spk(xvectors, utt_per_spk, BA)
profil

logging.info("computing combinations...")
couples = list(itertools.combinations(utt_per_spk.keys(), 2))
couples

typicality_and_dropout(profil, couples, utt_spk, BA, xvectors, typ_path, dout_path)


# typ,dout=load_filter_soft(typ_path,dout_path)
# df=df[list(typ.keys())]
# BA_test=list(typ.keys())

# utt={}
# for (idx,row) in df.iterrows():
#     utt[f"utt{idx}"]=dict(row)
    
# non,tar= load_trials()
# print("dout : ", dout)
# LLR_target,LLR_non,list_eer,list_cllr_min,list_cllr_act,list_Din=LR_framework(dout,typ,utt,tar,non,[0.12])
# plt.show()