import io
import time
import base64
import pandas as pd
from SparseIMUFunctions import train, test
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_confusion_matrix, f1_score
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager as fm, rc
from itertools import combinations
from multiprocessing import Pool
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
plt.switch_backend("Agg")


def read_data(pathToData):
    print("Loading dataset, this might take a while. Go, grab a coffee...")
    Xtracted = pd.read_hdf(pathToData + 'merged_Xtracted.h5', key="data")
    y = pd.read_csv(pathToData + 'merged_9_Gestures.csv')

    # rename "Class" to "Class_true"
    y.rename(columns={"Class": "Class_true"}, inplace=True)

    # add another col with finger+gesture
    y['Class_true_w_finger'] = y[['Class_true', 'Finger']].apply(lambda x: '_'.join(x), axis=1)

    assert len(Xtracted) == len(y), 'Lengths of X and y differ'
    print("Finished Loading Data")

    ## Keep only subset of sensors data
    # sensors to keep
    sensors_to_keep = sorted(["_acc_", "_gyr_", "_mag_"])
    sensors_to_keep_string = "|".join(sensors_to_keep)
    filename_string = "_".join([sensor.strip("_") for sensor in sensors_to_keep])

    # remove length and sum features
    feat_to_remove = ['length', 'sum']
    feat_to_remove_string = "|".join(feat_to_remove) # length|sum

    Xtracted_sub_cols = \
    Xtracted.columns[Xtracted.columns.str.contains(sensors_to_keep_string) \
                     & ~(Xtracted.columns.str.contains(feat_to_remove_string))].to_list()

    # sanity check
    assert len(Xtracted_sub_cols) == 6 * 17 * 3 * 3, "the feature input count is inconsistent"  # 6 features * 17 IMUs * acc, gyr, mag * x, y, z 

    Xtracted_sub_cols = ["TrialNo"] + Xtracted_sub_cols + ["PiD"] 
    X = Xtracted[Xtracted_sub_cols]

    return X, y

def train_test_split(X, y, actions_to_keep, gestures_to_keep, imus_to_keep):
    # choose the test participants
    np.random.seed(4)
    participants = [*range(1, 13)]

    # test participants
    test_participants = np.random.choice(participants, 2)
    print(f"The test participants are: {test_participants}")

    # train participants
    train_participants = list(set(participants) - set(test_participants))
    print(f"The train participants are: {train_participants}")

    # calculate columns to drop
    filter_string = "|".join(sorted(imus_to_keep))
    columns_to_drop = ["TrialNo", "PiD"]

    comboName, cols_to_keep = get_combo(sorted(imus_to_keep), X.columns.to_list())
    
    # create train test split
    ## Rearranges the order of columns!
    # X_train = X[X["PiD"].isin(train_participants)].drop(columns=columns_to_drop).filter(regex=filter_string)
    # X_test = X[X["PiD"].isin(test_participants)].drop(columns=columns_to_drop).filter(regex=filter_string)
    
    X_train = X[X["PiD"].isin(train_participants)].drop(columns=columns_to_drop)[cols_to_keep]
    X_test = X[X["PiD"].isin(test_participants)].drop(columns=columns_to_drop)[cols_to_keep]
    y_train = y[y["PiD"].isin(train_participants)]
    y_test = y[y["PiD"].isin(test_participants)]

    # get the correct rows
    rows_train = (y_train["Action"].isin(actions_to_keep)) & (y_train["Class_true_w_finger"].isin(gestures_to_keep))
    rows_test = (y_test["Action"].isin(actions_to_keep)) & (y_test["Class_true_w_finger"].isin(gestures_to_keep))
    
    X_train = X_train[rows_train]
    y_train = y_train[rows_train]
    
    X_test = X_test[rows_test]
    y_test = y_test[rows_test]

    assert len(y_test) == len(X_test), "test sets are not the same length"
    assert len(y_train) == len(X_train), "train sets are not the same length"

    return X_train, X_test, y_train, y_test

def rms(array):
    # Apply RMS to combine the features from same IMU
    return np.sqrt(np.mean(array ** 2))

def get_feature_importances(model, imus_to_keep, feature_names, isFreeHand, feature_cutoff=918):
    
    # importances = model.feature_importances_
    importances = pd.DataFrame(model.feature_importances_, columns=["Importance"], index=feature_names)
    
    if isFreeHand:
        # if FreeHand is part of the actions set
        # sort the importances
        importances_sorted = importances.sort_values(by="Importance", ascending=False)

        # get the top (feature_cutoff) features:
        top_importances = importances_sorted.iloc[:feature_cutoff]

        # this gets the features for each imu, stored in a dict
        fi_dict = defaultdict(list)
        for imu in imus_to_keep:
            for feature in top_importances.index:
                if imu in feature:
                    fi_dict[imu].append(importances.loc[feature]["Importance"])

        # return importances
        # calculate sum over features of same imu
        imu_fi_dict = {}
        for imu in imus_to_keep:
            imu_fi_dict[imu] = sum(fi_dict[imu])
    else:
        # this gets the features for each imu, stored in a dict
        fi_dict = defaultdict(list)
        for imu in imus_to_keep:
            for feature in feature_names:
                if imu in feature:
                    fi_dict[imu].append(importances.loc[feature]["Importance"])

        # calculate rms over features of same imu
        imu_fi_dict = {}
        for imu in imus_to_keep:
            # imu_fi_dict[imu] = rms(np.array(fi_dict[imu]))
            imu_fi_dict[imu] = sum(np.array(fi_dict[imu]))

    df_imu_fi= pd.DataFrame.from_dict(imu_fi_dict, orient="index", columns=["Feature importance"])

    # Normalizing
    df_imu_fi["Feature importance"] /= df_imu_fi["Feature importance"].sum()

    return df_imu_fi
    
def get_finger_gestures_set(X, y, actions_to_keep):
    # # this is the union
    # # get the gesture set after a set of actions is selected
    # gestures_set = y[y["Action"].isin(actions_to_keep)]['Class_true_w_finger'].unique()

    # get all gesture sets
    gestures_set = set(y[y["Action"].isin(actions_to_keep)]['Class_true_w_finger'].unique())
    print(gestures_set)

    for action in actions_to_keep:
        print(action)
        gestures_set_action = set(y[y["Action"] == action]['Class_true_w_finger'].unique())
        gestures_set = gestures_set.intersection(gestures_set_action)

    return list(gestures_set)


def get_cm(model, X_test, y_test, best_imus=None, acc=None, f1=None):
    labels = sorted(list(set(y_test)))
    fig, ax = plt.subplots(dpi=500)
    cm = confusion_matrix(y_test, model.predict(X_test), labels=labels)

    # make pretty labels
    labels_pretty = sorted([x.replace("_", " ") if x != "Static_Static" else "Static" for x in labels])
    # print(labels_pretty)

    cm_normal = cm/np.sum(cm, axis=1)
    cm_normal_df = pd.DataFrame(cm_normal, index=labels, columns=labels)

    print(pd.DataFrame(cm_normal, index=labels, columns=labels))
    ax = sns.heatmap(cm_normal, linecolor="white", linewidths=1, fmt=".3", xticklabels=labels_pretty, yticklabels=labels_pretty, ax=ax, cmap="Greens", annot=True, square=True, cbar=False, annot_kws={"size": 25 / np.sqrt(len(cm))})

    # ax.set_title(f"Confusion Matrix\nEstimated Accuracy: {acc:.3f}\n{' '.join(best_imus)}", fontsize=20)

    # old
    # ax.set_title("Confusion Matrix\nEstimated Accuracy: " + r"$\bf{:.2f}\%$".format(acc*100) + "\n", fontsize=18)
    ax.set_title("Confusion Matrix\nEstimated F1 score: " + r"$\bf{:.2f}\%$".format(f1*100) + "\n", fontsize=18)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    fig.tight_layout()

    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    return_dict = {
            "img": pngImageB64String,
            "df": cm_normal_df
            }

    # return pngImageB64String
    return return_dict

def train_subset(X, y, actions_to_keep, gestures_to_keep, imus_to_keep, imu_count_form):
    # get cutoffs 
    ks = get_cutoffs()
    fi_cutoff = ks[imu_count_form -1]

    # get train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, actions_to_keep, gestures_to_keep, imus_to_keep)

    model = train(X_train, y_train["Class_true_w_finger"])
    y_pred = model.predict(X_test)
    
    model_accuracy = accuracy_score(y_test["Class_true_w_finger"], y_pred)
    model_report = classification_report(y_test["Class_true_w_finger"], y_pred)

    print(f"Model Accuracy: {model_accuracy}")
    print(f"Model Report: {model_report}")
    
    info = {"Accuracy": model_accuracy,
           "Report": model_report}

    isFreeHand = True
    df_imu_fi = get_feature_importances(model, sorted(imus_to_keep), X_train.columns.to_list(),
                                        feature_cutoff=918, isFreeHand=isFreeHand)

    accuracy_dict = train_all_combos_top_n(X, y, df_imu_fi, fi_cutoff, imu_count_form,
                                          actions_to_keep, gestures_to_keep)
    
    best_imus_name = max(accuracy_dict, key=accuracy_dict.get)
    best_imus_fi = best_imus_name.split("-")

        
    # train a model with the top n imus and report estimates for accuracy and cm
    comboName, fi_cols = get_combo(sorted(best_imus_fi), X_train.columns.to_list())
    
    X_train_min = X_train[fi_cols]
    X_test_min = X_test[fi_cols]

    min_model = train(X_train_min, y_train["Class_true_w_finger"])
    y_pred = min_model.predict(X_test_min)
    
    min_model_accuracy = accuracy_score(y_test["Class_true_w_finger"], y_pred)
    min_model_f1 = f1_score(y_test["Class_true_w_finger"], y_pred, average="macro")
    min_model_report = classification_report(y_test["Class_true_w_finger"], y_pred)

    print(f"Min Model Accuracy: {min_model_accuracy}")
    print(f"Min Model f1: {min_model_f1}")
    print(f"Min Model Report: {min_model_report}")

    # old
    # return_dict = get_cm(min_model, X_test_min, y_test["Class_true_w_finger"], best_imus_fi, min_model_accuracy)

    return_dict = get_cm(min_model, X_test_min, y_test["Class_true_w_finger"], best_imus_fi, f1=min_model_f1)
    image = return_dict["img"]
    cm_df = return_dict["df"]

    return_val = {
            "image": image,
            "accuracy": min_model_accuracy,
            "best_imus": best_imus_fi,
            "cm_df": cm_df,
            "f1": min_model_f1
            }

    return return_val

def get_combo(Combo, cols):
    '''
    args: list of sensors to keep (combo)
    
    returns:
        combo_name: string of the combo
        cols_to_keep: columns that contain the combo
    '''
    combo_name = ''
    cols_to_keep = []

    for element in sorted(Combo):
        # create Combo name
        if element == Combo[-1]:
            combo_name += element
        else:
            combo_name += element + '_'

        # find all columns that involve the Combo
        for col in cols:
            if (element in col):
                cols_to_keep.append(col)

    return combo_name, cols_to_keep

def train_all_combos_top_n(X, y, fi, top_n, n_sensors_to_keep, actions_set, gestures_set):
    # get the top n imus
    top_n_imus = sorted(fi.sort_values(by="Feature importance", ascending=False).iloc[:top_n].index.to_list())
    # print(top_n_imus)
    
    # get all combos of n_sensors_to_keep chosen from top_n sensors
    # assuming top_n >= n_sensors_to_keep
    combos = list(map(list, combinations(top_n_imus, n_sensors_to_keep)))
    params = [{
        "Combo": combo,
        "data": X,
        "labels": y,
        "actions_set": actions_set, 
        "gestures_set": gestures_set
    } for combo in combos]
    
    accuracy_dict = {}
    for i in range(0, len(combos), 5):
        with Pool(40) as p:
            for res in p.imap(multitrain_func, params[i:i+5]):
                accuracy = res["accuracy"]
                name = res["name"]
                accuracy_dict[name] = accuracy
                
    return accuracy_dict 

def get_cutoffs():
    ks = []

    for n in range(1, 18):
        for k in range(1,18):
            kCn = len(list(combinations(range(k), n)))

            # we want the k for which kCn >= 1% of 17Cn
            if n <= 3:
                RHS = len(list(combinations(range(17), n))) * 10/100 # for count <=3, we use 10% of layouts
            else: 
                RHS = len(list(combinations(range(17), n))) * 1/100 # for count =4+, we use 1% of layouts
            

            if kCn >= RHS:
                # print(n, k, kCn, RHS)
                ks.append(k)
                break
    return ks

def multitrain_func(params):
       
    imus_to_keep = params["Combo"]
    X = params["data"]
    y = params["labels"]
    actions_set = params["actions_set"]
    gestures_set = params["gestures_set"]
    
    # create train, test split with only those imus
    X_train, X_test, y_train, y_test = train_test_split(X, y, actions_to_keep=actions_set,
                                                        gestures_to_keep=gestures_set, 
                                                        imus_to_keep=imus_to_keep)

    # train a model
    model = train(X_train, y_train["Class_true_w_finger"], n_jobs=-1)

    # get the accuracy
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test["Class_true_w_finger"], y_pred)
    name = "-".join(imus_to_keep)
    
    return_dict = {
        "accuracy": accuracy,
        "name": name
    }
    
    return return_dict
