# ==============================================================================
#  Copyright (c) 2023. Imen Ben Amor
# ==============================================================================

def load_trials():
    with open("/home/maax/Documents/Mega Sync/Cours M2/Explicabilité/BA_LR_Explained/target.txt","r") as f:
        text=f.readlines()
        target=[]
        for couple in text:
            target.append(eval(couple.strip()))
    with open("/home/maax/Documents/Mega Sync/Cours M2/Explicabilité/BA_LR_Explained/non.txt","r") as f:
        text=f.readlines()
        non=[]
        for couple in text:
            non.append(eval(couple.strip()))
    return non, target