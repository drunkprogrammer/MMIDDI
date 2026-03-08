import sqlite3
import datetime
import numpy as np
import pandas as pd
import os
import torch
import json
import pickle
import random
from time import strftime
from pandas import DataFrame
from sklearn.decomposition import PCA
from torch.cuda import is_available as cuda_is_available
from sklearn.model_selection import KFold
from test_ddi import save_result
from train_ddi import cross_validation, cross_validation_task2_3, cross_validation_special


def prepare(df_drug, feature_list, mechanism, action, drugA, drugB, task='task1', PCA_components=572, save_path=None):
    d_label = {}
    d_feature = {}
    d_event=[]
    for i in range(len(mechanism)):
        d_event.append(mechanism[i]+" "+action[i])
    count={}
    for i in d_event:
        if i in count:
            count[i]+=1
        else:
            count[i]=1

    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]]=i

    if save_path is not None:
        filename = os.path.join(save_path, 'label_mapping.txt')
        if not os.path.exists(filename):
            print(f"Saving label mapping to {filename}")
            with open(filename, 'w') as f:
                sorted_labels = sorted(d_label.items(), key=lambda item: item[1])
                for event_name, index in sorted_labels:
                    f.write(f"{index} : {event_name}\n")

    vector = np.zeros((len(np.array(df_drug['drug_name']).tolist()), 0), dtype=float)
    for i in feature_list:
        v1, pca_obj = feature_vector(i, df_drug, PCA_components)

        if save_path is not None:
            pkl_name = os.path.join(save_path, f'pca_{i}.pkl')
            if not os.path.exists(pkl_name):  # Avoid overwriting if calling prepare multiple times
                with open(pkl_name, 'wb') as f:
                    pickle.dump(pca_obj, f)
                print(f"Saved PCA model for {i} to {pkl_name}")

        vector = np.hstack((vector, v1))

    # Transfrom the drug ID to feature vector
    for i in range(len(np.array(df_drug['drug_name']).tolist())):
        d_feature[np.array(df_drug['drug_name']).tolist()[i]] = vector[i]

    # Use the dictionary to obtain feature vector and label
    if task == 'task1':
        new_feature = []
        new_label = []
        for i in range(len(d_event)):
            new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))) # new_feature[0].shape (1144,)
            new_label.append(d_label[d_event[i]])
        new_feature = np.array(new_feature) # new_feature size: list(37264)
        new_label = np.array(new_label)
        return (new_feature, new_label)

    elif task == 'task2_3':
        map = {}
        for index, row in df_drug.iterrows():
            map[row['drug_name']] = index
        new_feature = []
        new_label = []
        record_label = []
        for i in range(len(d_event)):
            new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]]))) # new_feature[0].shape (1144,)
            record_label.append([map[drugA[i]], map[drugB[i]]])
            new_label.append(d_label[d_event[i]])
        new_feature = np.array(new_feature) # new_feature size: list(37264)
        new_label = np.array(new_label)
        return (new_feature, new_label, record_label)


def prepare_special(df_drug, feature_list, mechanism, action, drugA, drugB, task='task1', PCA_components=572, save_path=None):
    d_label = {}
    d_feature = {}
    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1

    list1 = sorted(count.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(list1)):
        d_label[list1[i][0]] = i

    if save_path is not None:
        filename = os.path.join(save_path, 'label_mapping.txt')
        if not os.path.exists(filename):
            print(f"Saving label mapping to {filename}")
            with open(filename, 'w') as f:
                sorted_labels = sorted(d_label.items(), key=lambda item: item[1])
                for event_name, index in sorted_labels:
                    f.write(f"{index} : {event_name}\n")

    vector = np.zeros((len(np.array(df_drug['drug_name']).tolist()), 0), dtype=float)
    for i in feature_list:
        v1, pca_obj = feature_vector(i, df_drug, PCA_components)

        if save_path is not None:
            pkl_name = os.path.join(save_path, f'pca_{i}.pkl')
            if not os.path.exists(pkl_name):  # Avoid overwriting if calling prepare multiple times
                with open(pkl_name, 'wb') as f:
                    pickle.dump(pca_obj, f)
                print(f"Saved PCA model for {i} to {pkl_name}")

        vector = np.hstack((vector, v1))

    for i in range(len(np.array(df_drug['drug_name']).tolist())):
        d_feature[np.array(df_drug['drug_name']).tolist()[i]] = vector[i]

    new_feature = []
    new_label = []
    pair_map = {}  # Key: (DrugName1, DrugName2), Value: Matrix Index

    drug_map_indices = {}
    if task != 'task1':
        for index, row in df_drug.iterrows():
            drug_map_indices[row['drug_name']] = index
    record_label = []

    for i in range(len(d_event)):
        name_a = str(drugA[i]).strip()
        name_b = str(drugB[i]).strip()

        pair_map[(name_a, name_b)] = i

        new_feature.append(np.hstack((d_feature[drugA[i]], d_feature[drugB[i]])))
        new_label.append(d_label[d_event[i]])

        if task != 'task1':
            record_label.append([drug_map_indices[drugA[i]], drug_map_indices[drugB[i]]])

    new_feature = np.array(new_feature)
    new_label = np.array(new_label)

    if task == 'task1':
        return (new_feature, new_label, pair_map)
    else:
        return (new_feature, new_label, record_label, pair_map)

def feature_vector(feature_name, df, PCA_components):
    # df are the 572 kinds of drugs
    # Jaccard Similarity
    def Jaccard(matrix):
        matrix = np.asmatrix(matrix)
        numerator = matrix * matrix.T
        denominator = np.ones(np.shape(matrix)) * matrix.T + matrix * np.ones(np.shape(matrix.T)) - matrix * matrix.T
        return numerator / denominator

    all_feature = []
    drug_list = np.array(df[feature_name]).tolist()
    # Features for each drug, for example, when feature_name is target, drug_list=["P30556|P05412","P28223|P46098|â€¦â€¦"]
    for i in drug_list:
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)  # obtain all the features
    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = DataFrame(feature_matrix, columns=all_feature)  # Consrtuct feature matrices with key of dataframe
    for i in range(len(drug_list)):
        for each_feature in df[feature_name].iloc[i].split('|'):
            df_feature.loc[i, each_feature] = 1

    sim_matrix = Jaccard(np.array(df_feature))
    sim_matrix1 = np.asarray(sim_matrix)
    pca = PCA(n_components=PCA_components)
    pca.fit(sim_matrix1)
    sim_matrix = pca.transform(sim_matrix1)
    return sim_matrix, pca


def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
    device = torch.device("cuda" if cuda_is_available() else "cpu")

    parameters = {}
    parameters['learning_rate'] = float(args['learningRate'])
    parameters['weight_decay'] = float(args['weightDecay'])
    parameters['smooth_eps'] = float(args['smooth'])
    parameters['drop_rate'] = float(args['dropout'])
    parameters['batch_size'] = int(args['batchSize'])
    parameters['epochs'] = int(args['epochs'])
    parameters['sigma'] = float(args['sigma'])
    parameters['epsilon'] = float(args['epsilon'])
    parameters['mode'] = args['mode']
    parameters['oversampling'] = args['oversampling']
    parameters['seed'] = 0
    parameters['CV'] = 5
    parameters['num_layers'] = 1
    parameters['device'] = device

    random.seed(parameters['seed'])
    os.environ['PYTHONHASHSEED'] = str(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])
    torch.cuda.manual_seed(parameters['seed'])
    torch.cuda.manual_seed_all(parameters['seed'])
    torch.backends.cudnn.deterministic = True

    for db_name in args['database']:
        if db_name == 'event_small':
            nowtime = strftime('%Y_%m_%d-%H_%M_%S')
            saved_path = './results/DDIMDL_improved/event_small/' + nowtime + '/'
            parameters['event_num'] = 65
            parameters['vector_size'] = 572
            parameters['PCA_components'] = 572
            parameters['num_heads'] = 4
        elif db_name == 'event_medium':
            nowtime = strftime('%Y_%m_%d-%H_%M_%S')
            saved_path = './results/DDIMDL_improved/event_medium/' + nowtime + '/'
            parameters['event_num'] = 64
            parameters['vector_size'] = 643
            parameters['PCA_components'] = 642
            parameters['num_heads'] = 6
        elif db_name == 'event_large':
            nowtime = strftime('%Y_%m_%d-%H_%M_%S')
            saved_path = './results/DDIMDL_improved/event_large/' + nowtime + '/'
            parameters['event_num'] = 100
            parameters['vector_size'] = 1258
            parameters['PCA_components'] = 1258
            parameters['num_heads'] = 17
        elif db_name == 'event_special':
            nowtime = strftime('%Y_%m_%d-%H_%M_%S')
            saved_path = './results/DDIMDL_improved/event_special/' + nowtime + '/'
            parameters['event_num'] = 64
            parameters['vector_size'] = 643
            parameters['PCA_components'] = 642
            parameters['num_heads'] = 6
        else:
            raise ValueError("Invalid database choice")

        if not os.path.exists(saved_path):
            os.makedirs(saved_path)
            print("Directory ", saved_path, " Created ")
        else:
            print("Directory ", saved_path, " already exists")

        # ==========================================================
        # ### SAVE HYPERPARAMETERS ###
        # ==========================================================

        config_to_save = parameters.copy()

        config_to_save.update(args)

        if 'device' in config_to_save:
            config_to_save['device'] = str(config_to_save['device'])

        config_path = os.path.join(saved_path, 'hyperparameters.json')
        with open(config_path, 'w') as f:
            json.dump(config_to_save, f, indent=4)

        print(f"Configuration saved to: {config_path}")

        # ==========================================================
        # ### END ###
        # ==========================================================

        conn = sqlite3.connect(f"../../Dataset1/version10/{db_name}.db")
        df_drug = pd.read_sql('select * from drug;', conn)

        feature_list = args['featureList']
        featureName="+".join(feature_list)
        clf_list = args['classifier']
        task_list = args['task']

        result_all = {}
        result_eve = {}
        result_keep = {}

        extraction = pd.read_sql('select * from extraction;', conn)
        mechanism = extraction['mechanism']
        action = extraction['action']

        drugA = np.array(extraction['drugA'])
        drugB = np.array(extraction['drugB'])

        d_event = [f"{m} {a}" for m, a in zip(mechanism, action)]
        count = {}
        for i in d_event:
            count[i] = count.get(i, 0) + 1

        sorted_events = sorted(count.items(), key=lambda x: x[1], reverse=True)
        # d_label = {name: index}
        d_label = {item[0]: i for i, item in enumerate(sorted_events)}

        # Create the reverse list: class_names[0] = "Metabolism increased"
        class_names = [None] * len(d_label)
        for name, idx in d_label.items():
            class_names[idx] = name
        # -----------------------------------------------

        start = datetime.datetime.now()

        for clf in clf_list:
            print(clf)
            for task in task_list:
                print(task)
                if task == 'task1':
                    if db_name != 'event_special':
                        all_matrix = []
                        for feature in feature_list:
                            print(feature)
                            new_feature, new_label = prepare(df_drug, [feature], mechanism, action, drugA, drugB, task='task1', PCA_components=parameters['PCA_components'], save_path=saved_path)
                            all_matrix.append(new_feature)
                        keep_result, all_result, each_result = cross_validation(all_matrix, new_label, clf, parameters, saved_path, device)
                    else:
                        all_matrix = []
                        for feature in feature_list:
                            print(feature)
                            new_feature, new_label, pair_map = prepare_special(df_drug, [feature], mechanism, action, drugA, drugB, task='task1', PCA_components=parameters['PCA_components'], save_path=saved_path)
                            all_matrix.append(new_feature)
                        df_event = pd.read_sql('select * from event_tasks;', conn)
                        keep_result, all_result, each_result = cross_validation_special(all_matrix, new_label, df_event, pair_map, task, clf, parameters, saved_path, device)

                    save_result(featureName, 'keep', clf, keep_result, saved_path)
                    save_result(featureName, 'all', clf, all_result, saved_path)
                    save_result(featureName, 'each', clf, each_result, saved_path)
                    result_all[clf] = all_result
                    result_eve[clf] = each_result
                    result_keep[clf] = keep_result

                elif task == 'task2_3':
                    if db_name != 'event_special':
                        all_matrix = []
                        kf = KFold(n_splits=parameters['CV'], shuffle=True, random_state=parameters['seed'])

                        keep_2 = np.zeros((6, 1))
                        all_2 = np.zeros((11, 1))
                        each_2 = np.zeros((parameters['event_num'], 6))

                        keep_3 = np.zeros((6, 1))
                        all_3 = np.zeros((11, 1))
                        each_3 = np.zeros((parameters['event_num'], 6))

                        for feature in feature_list:
                            print(feature)
                            new_feature, new_label, record_label = prepare(df_drug, [feature], mechanism, action, drugA, drugB, task='task2_3', PCA_components=parameters['PCA_components'], save_path=saved_path)
                            all_matrix.append(new_feature)

                        count = 0
                        for train_drug, test_drug in kf.split(range(parameters['vector_size'])):
                            count = count + 1
                            print(f"count: {count}")
                            keep_result2, all_result2, each_result2, keep_result3, all_result3, each_result3 = cross_validation_task2_3(all_matrix, new_label, clf, parameters, train_drug, record_label, count, saved_path, device)

                            keep_2 += (keep_result2) / 5
                            all_2 += (all_result2) / 5
                            each_2 += (each_result2) / 5
                            keep_3 += (keep_result3) / 5
                            all_3 += (all_result3) / 5
                            each_3 += (each_result3) / 5

                        save_result(featureName, 'keep2', clf, keep_result2, saved_path)
                        save_result(featureName, 'all2', clf, all_2, saved_path)
                        save_result(featureName, 'each2', clf, each_2, saved_path)
                        save_result(featureName, 'keep3', clf, keep_result3, saved_path)
                        save_result(featureName, 'all3', clf, all_3, saved_path)
                        save_result(featureName, 'each3', clf, each_3, saved_path)

                        print("-----------------Task2 results------------------------")
                        print("accuracy_score:", all_2[0])
                        print("roc_aupr_score_micro:", all_2[1])
                        print("roc_auc_score_micro:", all_2[3])
                        print("f1_score_macro:", all_2[6])
                        print("precision_score_macro:", all_2[8])
                        print("recall_score_macro:", all_2[10])

                        print("-----------------Task3 results------------------------")
                        print("accuracy_score:", all_3[0])
                        print("roc_aupr_score_micro:", all_3[1])
                        print("roc_auc_score_micro:", all_3[3])
                        print("f1_score_macro:", all_3[6])
                        print("precision_score_macro:", all_3[8])
                        print("recall_score_macro:", all_3[10])

                    else:
                        df_event = pd.read_sql('select * from event_tasks;', conn)

                        all_matrix = []
                        for feature in feature_list:
                            print(feature)
                            new_feature, new_label, record_label, pair_map = prepare_special(df_drug, [feature], mechanism, action,
                                                                               drugA, drugB, task='task2',
                                                                               PCA_components=parameters[
                                                                                   'PCA_components'],
                                                                               save_path=saved_path)

                            all_matrix.append(new_feature)
                        keep_result, all_result, each_result = cross_validation_special(all_matrix, new_label, df_event, pair_map, 'task2', clf, parameters, saved_path, device)
                        save_result(featureName, 'task2_keep', clf, keep_result, saved_path)
                        save_result(featureName, 'task2_all', clf, all_result, saved_path)
                        save_result(featureName, 'task2_each', clf, each_result, saved_path)

                        all_matrix = []
                        for feature in feature_list:
                            print(feature)
                            new_feature, new_label, record_label, pair_map = prepare_special(df_drug, [feature], mechanism, action,
                                                                               drugA, drugB, task='task3',
                                                                               PCA_components=parameters[
                                                                                   'PCA_components'],
                                                                               save_path=saved_path)
                            all_matrix.append(new_feature)
                        keep_result, all_result, each_result = cross_validation_special(all_matrix, new_label, df_event, pair_map,  'task3', clf, parameters, saved_path, device)
                        save_result(featureName, 'task3_keep', clf, keep_result, saved_path)
                        save_result(featureName, 'task3_all', clf, all_result, saved_path)
                        save_result(featureName, 'task3_each', clf, each_result, saved_path)

        end = datetime.datetime.now()
        print("time used:", end - start)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--database", choices=["event_small", "event_medium", "event_large", "event_special"], default=["event_large"], help="database to use", nargs="+")
    parser.add_argument("-f", "--featureList", default=["smile", "target", "enzyme"], help="features to use", nargs="+")
    parser.add_argument("-c", "--classifier", choices=["MMIDDI"], default=["MMIDDI"], help="classifiers to use", nargs="+")
    parser.add_argument("-t", "--task", choices=["task1", "task2_3"], default=["task1"], help="classify task", nargs="+")
    parser.add_argument("-m", "--mode",  choices=["train", "test", "train_test"], default=["train_test"], help="mode to run")
    parser.add_argument("-g", "--gpu", default=0, help="GPU number")
    parser.add_argument("-l", "--learningRate", default=0.0001, help="Learning Rate")
    parser.add_argument("-w", "--weightDecay", default=0.001, help="Weight Decay")
    parser.add_argument("-s", "--smooth", default=0.1, help="Label Smoothing")
    parser.add_argument("-o", "--dropout", default=0.2, help="Dropout Rate")
    parser.add_argument("-b", "--batchSize", default=128, help="Batch Size")
    parser.add_argument("-e", "--epochs", default=300, help="Training Epochs")
    parser.add_argument("-a", "--sigma", default=0.7, help="Sigma for Adversarial Training")
    parser.add_argument("-p", "--epsilon", default=0.01, help="Epsilon for Adversarial Training")
    parser.add_argument("-v", "--oversampling", action='store_true', help="Whether to use oversampling")
    args = vars(parser.parse_args())
    print(args)
    main(args)