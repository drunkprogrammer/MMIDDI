# Problem: the long-tail distribution of the interaction data and unbalanced drug occurrence data
# Solution: use the grouping drug data and hierarchical sampling to balance the data distribution
# Input: drug interaction data
# Output: balanced drug data
import itertools

import pandas as pd
import numpy as np
import random
import os
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit



def get_drug_data(conn, table_name):
    # read the drug data from the database
    query = 'SELECT * FROM ' + table_name
    df = pd.read_sql(query, conn)
    return df


def define_pair_frequency_split(df_ddi, random_seed):
    def get_pair_frequency(row):
        if row['frequency_a'] == 'low' or row['frequency_b'] == 'low':
            return 'low'
        elif row['frequency_a'] == 'high' or row['frequency_b'] == 'high':
            return 'high'
        else:
            return 'mid'

    df_ddi['pair_frequency'] = df_ddi.apply(get_pair_frequency, axis=1)

    high_ddi = df_ddi[df_ddi['pair_frequency'] == 'high']
    mid_ddi = df_ddi[df_ddi['pair_frequency'] == 'mid']
    low_ddi = df_ddi[df_ddi['pair_frequency'] == 'low']

    df_ddi['stratify_key'] = df_ddi['label'].astype(str) + '_' + df_ddi['pair_frequency'].astype(str)

    def stratified_split(df, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        folds = []
        for _, test_idx in skf.split(df, df['label']):
            folds.append(df.iloc[test_idx])
        return folds

    high_folds = stratified_split(high_ddi)
    mid_folds = stratified_split(mid_ddi)
    low_folds = stratified_split(low_ddi)

    final_folds = []
    for i in range(5):
        # 合并高频、中频、低频的第i部分
        fold = pd.concat([high_folds[i], mid_folds[i], low_folds[i]])
        final_folds.append(fold)

    for fold_idx, fold in enumerate(final_folds):
        print(f"Fold {fold_idx + 1}:")

        # 验证频率分布
        freq_dist = fold['pair_frequency'].value_counts(normalize=True)
        print("Frequency Distribution:")
        print(freq_dist)

        # 验证标签覆盖
        labels_present = fold['label'].unique()
        print(f"Unique Labels: {len(labels_present)}/65")


def assign_train_labels(train_drug_ids, df_event, name):
    for index, row in df_event.iterrows():
        if row['drug_id1'] in train_drug_ids and row['drug_id2'] in train_drug_ids:
            df_event.at[index, name] = 1
        else:
            df_event.at[index, name] = 0
    return df_event

def assign_task2_labels(train_drug_ids, test_drug_ids, df_event, name):
    for index, row in df_event.iterrows():
        if row['drug_id1'] in test_drug_ids and row['drug_id2'] in train_drug_ids:
            df_event.at[index, name] = 1
        elif row['drug_id1'] in train_drug_ids and row['drug_id2'] in test_drug_ids:
            df_event.at[index, name] = 1
        else:
            df_event.at[index, name] = 0
    return df_event

def assign_task3_labels(test_drug_ids, df_event, name):
    for index, row in df_event.iterrows():
        if row['drug_id1'] in test_drug_ids and row['drug_id2'] in test_drug_ids:
            df_event.at[index, name] = 1
        else:
            df_event.at[index, name] = 0
    return df_event

# Verify current label coverage in Task 2 and Task 3
def verify_label_coverage(df_event, task_column, all_labels):
    task_labels = set(df_event[df_event[task_column] == 1]['label'])
    missing_labels = all_labels - task_labels
    return missing_labels

def verify_drug_pair_coverage(df_event, drug_ids, label):
    # Verify drug pair coverage in the dataset
    if len(drug_ids) < 2:
        return False

    drug_pairs = list(itertools.permutations(drug_ids, 2))
    df_event_label = df_event[df_event['label'] == label]
    event_drug_pairs = list(zip(df_event_label['drug_id1'], df_event_label['drug_id2']))

    # Generate all non-repeating pairwise combinations
    for drug_pair in drug_pairs:
        if drug_pair in event_drug_pairs:
            return True

    return False

def collect_ids(x):
    ## Group by 'label' and collect unique IDs from both 'id1' and 'id2'
    ids1 = set(x['drug_id1'])  # Get unique IDs from id1
    ids2 = set(x['drug_id2'])  # Get unique IDs from id2
    return list(ids1.union(ids2))  # Union the sets to remove duplicates and convert to list

def connect_to_db(filename):
    # create a database connection
    import sqlite3
    conn = sqlite3.connect(filename)
    cur = conn.cursor()
    return cur, conn

def summary_drug_label_information(db_name):
    cur, conn = connect_to_db(db_name)
    df_drug = get_drug_data(conn, 'drug')
    df_event = get_drug_data(conn, 'event')
    df_event = df_event.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # get the name and id of drugs
    drug_list = defaultdict(str)
    for index, row in df_drug.iterrows():
        drug_list[row['drug_id']] = row['drug_name']

    # get each row drug pairs from the event data
    drug_counts = defaultdict(int)
    for index, row in df_event.iterrows():
        drug_counts[row['drug_id1']] += 1
        drug_counts[row['drug_id2']] += 1

    df_drug_counts = pd.DataFrame({
        'id': list(drug_counts.keys()),
        'count': list(drug_counts.values())
    })
    df_drug_counts = df_drug_counts.sort_values(by='count', ascending=False)

    q_high = df_drug_counts["count"].quantile(0.6)  # 前40%高频：0% ~ 40% -> 对应分位数 60%
    q_mid = df_drug_counts["count"].quantile(0.2)   # 中间40%中频：40% ~ 80% -> 对应分位数 20%
    q_low = 0                                      # 后20%低频：80% ~ 100%

    df_drug_counts["frequency"] = pd.cut(
        df_drug_counts["count"],
        bins=[q_low, q_mid, q_high, np.inf],  # 分箱边界
        labels=["low", "mid", "high"],        # 标签顺序对应 bins 区间
        include_lowest=True,                  # 包含最小值
        right=False                           # 左闭右开区间
    )

    event_number = df_event['interaction'].value_counts()
    event_number = event_number.index.tolist()
    event_number = {event_number[i]: i for i in range(len(event_number))}
    df_event['label'] = df_event['interaction'].map(event_number)

    drugA_label_numbers = df_event.groupby('drug_id1')['label'].apply(set)
    drugB_label_numbers = df_event.groupby('drug_id2')['label'].apply(set)

    # Merge label sets based on the same drug_id, ensuring all unique labels are included
    df_label_sets = pd.concat([drugA_label_numbers, drugB_label_numbers]).groupby(level=0).agg(lambda x: set().union(*x)).reset_index()
    df_label_sets.columns = ['id', 'label_set']

    # Merge the Drug information, frequency information, Count information, Label Sets information
    df_drug_info = pd.merge(
        df_drug,
        df_drug_counts[['id', 'frequency', 'count']],
        how='left',
        left_on='drug_id',
        right_on='id',
    ).drop('id', axis=1)

    df_drug_new = pd.merge(
        df_drug_info,
        df_label_sets,
        how='left',
        left_on='drug_id',
        right_on='id',
    ).drop('id', axis=1)

    # Summary each label corresponding to the drug ids
    # drug_ids_by_label = df_event.groupby('label', group_keys=False).apply(collect_ids)
    drug_ids_by_label = df_event.groupby('label').apply(collect_ids)
    df_label_new = pd.DataFrame({
        'label': drug_ids_by_label.index,
        'interaction': event_number.keys(),
        'drug_ids': drug_ids_by_label.values,
        'count': drug_ids_by_label.apply(len)
    })

    print(df_drug_new.head())
    print(df_label_new.head())

    # Remove rows where the 'count' column contains NaN values
    df_drug_new = df_drug_new.dropna(subset=['count'])

    return df_drug_new, df_label_new, df_event, cur, conn

def sort_label_count(df_drug, df_label):

    frequency_order = {"high": 3, "mid": 2, "low": 1}
    df_drug = df_drug.set_index("drug_id")
    drug_freq_map = df_drug["frequency"].map(frequency_order).to_dict()

    label_counts = defaultdict(int)
    all_drug_ids = set(drug_id for sublist in df_label["drug_ids"] for drug_id in sublist)
    for drug_id in all_drug_ids:
        label_counts[drug_id] = len(df_drug.loc[drug_id, "label_set"])

    def combined_sort(drug_ids):
        """
           Sort the drug ids first by their frequency (high > mid > low) and then by label counts.
           If they are the same, use the drug_id as a secondary sorting criterion to avoid random ordering.
        """
        sorted_ids = sorted(
            drug_ids,
            key=lambda x: (-drug_freq_map.get(x, 0), -label_counts[x], x),
            reverse=False
        )
        high_freq = [x for x in sorted_ids if drug_freq_map.get(x, 0) == 3]
        mid_freq = [x for x in sorted_ids if drug_freq_map.get(x, 0) == 2]
        low_freq = [x for x in sorted_ids if drug_freq_map.get(x, 0) == 1]
        return [high_freq, mid_freq, low_freq]

    df_label["drug_ids"] = df_label["drug_ids"].apply(combined_sort)
    df_label = df_label.sort_values(by="count", ascending=True).reset_index(drop=True)
    print(df_label.head())
    return df_label

def plot_label_drug_count_distribution(df_label):
    # Plot the distribution of drug labels
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_label["label"], df_label["count"], color="skyblue")
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel("Label", fontsize=12)
    ax.set_title("Distribution of Drug Labels", fontsize=14)
    plt.show()

def should_drop_drug(drop_id, df_drug, labels, drug_ids):
    """
    Determines if dropping a drug from a dataset affects the completeness of the label set.

    Args:
        drop_id (str): The ID of the drug to potentially drop.
        df_drug (DataFrame): DataFrame containing drug data with 'label_set'.
        labels (set): The complete set of labels that must be maintained (e.g., train_drug_labels or test_drug_labels).
        drug_ids (set): The set of drug IDs to consider (e.g., train_drug_ids or test_drug_ids).

    Returns:
        bool: True if dropping the drug does not affect the label coverage; False otherwise.
    """
    drop_labels = set()
    for drug_id in drug_ids:
        if drug_id != drop_id:
            label_set = df_drug.loc[drug_id, "label_set"]
            drop_labels.update(label_set)

    if drop_labels == labels:
        return True
    else:
        return False

def balance_train_test_drugs(train_ids, test_ids, label, df_event, mode='train'):
    # mode='train' represents the training dataset labels are not coveraged
    # mode='test' means the test dataset labels are not coveraged
    if mode == 'train':
        # adjust test drug to train drug
        for test in test_ids:
            train_supply_ids = train_ids + [test]
            test_drop_ids = [x for x in test_ids if x != test]
            if verify_drug_pair_coverage(df_event, train_supply_ids, label) and verify_drug_pair_coverage(df_event, test_drop_ids, label):
                return train_supply_ids, test_drop_ids
        print(f"Warning: Failed to balance label {label} in training set using one drug!")

        permutated_test_ids = list(itertools.permutations(test_ids, 2))
        for permutated_test in permutated_test_ids:
            train_supply_ids = train_ids + list(permutated_test)
            test_drop_ids = [x for x in test_ids if x not in permutated_test]
            if verify_drug_pair_coverage(df_event, train_supply_ids, label) and verify_drug_pair_coverage(df_event, test_drop_ids, label):
                return train_supply_ids, test_drop_ids
        print(f"Warning: Failed to balance label {label} in training set using two drugs")
        return train_ids, test_ids

    elif mode == 'test':
        # adjust train drug to test drug
        for train in train_ids:
            test_supply_ids = test_ids + [train]
            train_drop_ids = [x for x in train_ids if x != train]
            if verify_drug_pair_coverage(df_event, test_supply_ids, label) and verify_drug_pair_coverage(df_event, train_drop_ids, label):
                return train_drop_ids, test_supply_ids
        print(f"Warning: Failed to balance label {label} in test set using one drug")

        permutated_train_ids = list(itertools.permutations(train_ids, 2))
        for permutated_train in permutated_train_ids:
            test_supply_ids = test_ids + list(permutated_train)
            train_drop_ids = [x for x in train_ids if x not in permutated_train]
            if verify_drug_pair_coverage(df_event, test_supply_ids, label) and verify_drug_pair_coverage(df_event, train_drop_ids, label):
                return train_drop_ids, test_supply_ids
        print(f"Warning: Failed to balance label {label} in test set using two drugs")
        return train_ids, test_ids
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'test'")


def label_grouping(df_label_sorted, df_drug_new, df_event):
    # train and test drug partitioned
    train_drug_ids = set()
    test_drug_ids = set()

    train_labels = set()
    test_labels = set()

    high_train_nums = 0
    mid_train_nums = 0
    low_train_nums = 0

    high_test_nums = 0
    mid_test_nums = 0
    low_test_nums = 0

    df_drug_new = df_drug_new.set_index("drug_id")
    high_train_ids, mid_train_ids, low_train_ids = set(), set(), set()
    high_test_ids, mid_test_ids, low_test_ids = set(), set(), set()

    for index, row in df_label_sorted.iterrows():
        train_id = []
        test_id = []
        # select 80% of high frequency, mid frequency, and low frequency drugs for training, down rounded
        high_freq, mid_freq, low_freq = row["drug_ids"]
        high_train_count = math.floor(ratio * len(high_freq))
        mid_train_count = math.floor(ratio * len(mid_freq))
        low_train_count = math.floor(ratio * len(low_freq))
        high_test_count = len(high_freq) - high_train_count
        mid_test_count = len(mid_freq) - mid_train_count
        low_test_count = len(low_freq) - low_train_count

        train_count = high_train_count + mid_train_count + low_train_count
        test_count = high_test_count + mid_test_count + low_test_count

        label = row["label"]

        if len(train_labels) == 0:
            train_id = high_freq[:high_train_count] + mid_freq[:mid_train_count] + low_freq[:low_train_count]
            test_id = high_freq[high_train_count:] + mid_freq[mid_train_count:] + low_freq[low_train_count:]
            high_train_nums += high_train_count
            mid_train_nums += mid_train_count
            low_train_nums += low_train_count
            high_test_nums += high_test_count
            mid_test_nums += mid_test_count
            low_test_nums += low_test_count

            high_train_ids.update(high_freq[:high_train_count])
            mid_train_ids.update(mid_freq[:mid_train_count])
            low_train_ids.update(low_freq[:low_train_count])
            high_test_ids.update(high_freq[high_train_count:])
            mid_test_ids.update(mid_freq[mid_train_count:])
            low_test_ids.update(low_freq[low_train_count:])

            train_drug_ids.update(train_id)
            test_drug_ids.update(test_id)

        else:
            # judge train and test drug ids are in the high_freq, mid_freq, low_freq, and select no repeated ids
            high_train_duplicates = train_drug_ids.intersection(high_freq)
            mid_train_duplicates = train_drug_ids.intersection(mid_freq)
            low_train_duplicates = train_drug_ids.intersection(low_freq)

            high_test_duplicates = test_drug_ids.intersection(high_freq)
            mid_test_duplicates = test_drug_ids.intersection(mid_freq)
            low_test_duplicates = test_drug_ids.intersection(low_freq)

            train_duplicates = len(high_train_duplicates) + len(mid_train_duplicates) + len(low_train_duplicates)
            test_duplicates = len(high_test_duplicates) + len(mid_test_duplicates) + len(low_test_duplicates)

            if train_count == 0 and train_duplicates == 0:
                print(f"label: {label} training samples are lack")

            if test_count == 0 and test_duplicates == 0:
                print(f"label {label} test samples are lack")

            if train_count < 2:
                print(f"label: {label} training samples are lack of training conditions")

            if test_count < 2:
                print(f"label: {label} test samples are lack of double drug cold-start")

            if high_train_count >= len(high_train_duplicates):
                high_supply_train_nums = high_train_count - len(high_train_duplicates)
                k = 0
                for i in range(len(high_freq)):
                    if high_freq[i] not in train_drug_ids and high_freq[i] not in test_drug_ids:
                        train_id.append(high_freq[i])
                        k = k + 1
                    if k == high_supply_train_nums or i == len(high_freq) - 1:
                        break
                high_train_ids.update(train_id)
                train_drug_ids.update(train_id)
                high_train_nums += k
            else:
                # from high_train_duplicates to drop the no effect ids from train_drug_ids, if no drug id can remove, then keep the same
                high_drop_train_nums = len(high_train_duplicates) - high_train_count
                i = 0
                k = 0
                for drug_id in high_train_duplicates:
                    if should_drop_drug(drug_id, df_drug_new, train_labels, train_drug_ids):
                        high_train_ids.remove(drug_id)
                        train_drug_ids.remove(drug_id)
                        high_train_duplicates.remove(drug_id)
                        high_drop_train_nums -= 1
                        k = k + 1
                    if high_drop_train_nums == 0 or i == len(high_train_duplicates) - 1:
                        break
                    i += 1
                high_train_nums -= k

            if mid_train_count >= len(mid_train_duplicates):
                mid_supply_train_nums = mid_train_count - len(mid_train_duplicates)
                k = 0
                for i in range(len(mid_freq)):
                    if mid_freq[i] not in train_drug_ids and mid_freq[i] not in test_drug_ids:
                        train_id.append(mid_freq[i])
                        k = k + 1
                    if k == mid_supply_train_nums or i == len(mid_freq) - 1:
                        break
                mid_train_ids.update(train_id)
                train_drug_ids.update(train_id)
                mid_train_nums += k
            else:
                mid_drop_train_nums = len(mid_train_duplicates) - mid_train_count
                i = 0
                k = 0
                for drug_id in mid_train_duplicates:
                    if should_drop_drug(drug_id, df_drug_new, train_labels, train_drug_ids):
                        mid_train_ids.remove(drug_id)
                        train_drug_ids.remove(drug_id)
                        mid_train_duplicates.remove(drug_id)
                        mid_drop_train_nums -= 1
                        k = k + 1
                    if mid_drop_train_nums == 0 or i == len(mid_train_duplicates) - 1:
                        break
                    i += 1
                mid_train_nums -= k

            if low_train_count >= len(low_train_duplicates):
                low_supply_train_nums = low_train_count - len(low_train_duplicates)
                k = 0
                for i in range(len(low_freq)):
                    if low_freq[i] not in train_drug_ids and low_freq[i] not in test_drug_ids:
                        train_id.append(low_freq[i])
                        k = k + 1
                    if k == low_supply_train_nums or i == len(low_freq) - 1:
                        break
                low_train_ids.update(train_id)
                train_drug_ids.update(train_id)
                low_train_nums += k
            else:
                low_drop_train_nums = len(low_train_duplicates) - low_train_count
                i = 0
                k = 0
                for drug_id in low_train_duplicates:
                    if should_drop_drug(drug_id, df_drug_new, train_labels, train_drug_ids):
                        low_train_ids.remove(drug_id)
                        train_drug_ids.remove(drug_id)
                        low_train_duplicates.remove(drug_id)
                        low_drop_train_nums -= 1
                        k = k + 1
                    if low_drop_train_nums == 0 or i == len(low_train_duplicates) - 1:
                        break
                    i += 1
                low_train_nums -= k

            if high_test_count >= len(high_test_duplicates):
                high_supply_test_nums = high_test_count - len(high_test_duplicates)
                k = 0
                for i in range(len(high_freq)):
                    if high_freq[i] not in train_drug_ids and high_freq[i] not in test_drug_ids and high_freq[i] not in train_id:
                        test_id.append(high_freq[i])
                        k = k + 1
                    if k == high_supply_test_nums or i == len(high_freq) - 1:
                        break
                high_test_ids.update(test_id)
                test_drug_ids.update(test_id)
                high_test_nums += k
            else:
                high_drop_test_nums = len(high_test_duplicates) - high_test_count
                i = 0
                k = 0
                for drug_id in high_test_duplicates:
                    if should_drop_drug(drug_id, df_drug_new, test_labels, test_drug_ids):
                        high_test_ids.remove(drug_id)
                        test_drug_ids.remove(drug_id)
                        high_test_duplicates.remove(drug_id)
                        high_drop_test_nums -= 1
                        k = k + 1
                    if high_drop_test_nums == 0 or i == len(high_test_duplicates) - 1:
                        break
                    i += 1
                high_test_nums -= k

            if mid_test_count >= len(mid_test_duplicates):
                mid_supply_test_nums = mid_test_count - len(mid_test_duplicates)
                k = 0
                for i in range(len(mid_freq)):
                    if mid_freq[i] not in train_drug_ids and mid_freq[i] not in test_drug_ids and mid_freq[
                        i] not in train_id:
                        test_id.append(mid_freq[i])
                        k = k + 1
                    if k == mid_supply_test_nums or i == len(mid_freq) - 1:
                        break
                mid_test_ids.update(test_id)
                test_drug_ids.update(test_id)
                mid_test_nums += k
            else:
                mid_drop_test_nums = len(mid_test_duplicates) - mid_test_count
                i = 0
                k = 0
                for drug_id in mid_test_duplicates:
                    if should_drop_drug(drug_id, df_drug_new, test_labels, test_drug_ids):
                        mid_test_ids.remove(drug_id)
                        test_drug_ids.remove(drug_id)
                        mid_test_duplicates.remove(drug_id)
                        mid_drop_test_nums -= 1
                        k = k + 1
                    if mid_drop_test_nums == 0 or i == len(mid_test_duplicates) - 1:
                        break
                    i += 1
                mid_test_nums -= k

            if low_test_count >= len(low_test_duplicates):
                low_supply_test_nums = low_test_count - len(low_test_duplicates)
                k = 0
                for i in range(len(low_freq)):
                    if low_freq[i] not in train_drug_ids and low_freq[i] not in test_drug_ids and low_freq[i] not in train_id:
                        test_id.append(low_freq[i])
                        k = k + 1
                    if k == low_supply_test_nums or i == len(low_freq) - 1:
                        break
                low_test_ids.update(test_id)
                test_drug_ids.update(test_id)
                low_test_nums += k
            else:
                low_drop_test_nums = len(low_test_duplicates) - low_test_count
                i = 0
                k = 0
                for drug_id in low_test_duplicates:
                    if should_drop_drug(drug_id, df_drug_new, test_labels, test_drug_ids):
                        low_test_ids.remove(drug_id)
                        test_drug_ids.remove(drug_id)
                        low_test_duplicates.remove(drug_id)
                        low_drop_test_nums -= 1
                        k = k + 1
                    if low_drop_test_nums == 0 or i == len(low_test_duplicates) - 1:
                        break
                    i += 1
                low_test_nums -= k

            if high_train_duplicates:
                train_id.extend(high_train_duplicates)
            if mid_train_duplicates:
                train_id.extend(mid_train_duplicates)
            if low_train_duplicates:
                train_id.extend(low_train_duplicates)
            if high_test_duplicates:
                test_id.extend(high_test_duplicates)
            if mid_test_duplicates:
                test_id.extend(mid_test_duplicates)
            if low_test_duplicates:
                test_id.extend(low_test_duplicates)

        if verify_drug_pair_coverage(df_event, train_id, label):
            train_labels.add(label)
        else:
            train_ids, test_ids = balance_train_test_drugs(train_id, test_id, label, df_event, mode='train')
            if train_id == train_ids or test_ids == test_id:
                print(f"Label {label} is not covered in the training set")
            else:
                train_supply_id = list(set(train_ids) - set(train_id))
                test_drop_id = list(set(test_id) - set(test_ids))
                train_supply_frequency = df_drug_new.loc[train_supply_id, 'frequency'].iloc[0]
                test_drop_frequency = df_drug_new.loc[test_drop_id, 'frequency'].iloc[0]
                if train_supply_frequency == 'high':
                    high_train_nums += 1
                    high_train_count += 1
                elif train_supply_frequency == 'mid':
                    mid_train_nums += 1
                    mid_train_count += 1
                    mid_train_count += 1
                elif train_supply_frequency == 'low':
                    low_train_nums += 1
                    low_train_count += 1

                if test_drop_frequency == 'high':
                    high_test_nums -= 1
                    high_test_count -= 1
                elif test_drop_frequency == 'mid':
                    mid_test_nums -= 1
                    mid_test_count -= 1
                elif test_drop_frequency == 'low':
                    low_test_nums -= 1
                    low_test_count -= 1

                train_labels.add(label)
                train_drug_ids |= set(train_supply_id)
                test_drug_ids -= set(test_drop_id)


        if verify_drug_pair_coverage(df_event, test_id, label):
            test_labels.add(label)
        else:
            train_ids, test_ids = balance_train_test_drugs(train_id, test_id, label, df_event, mode='test')
            if train_id == train_ids or test_ids == test_id:
                print(f"Label {label} is not covered in the test set for double drug cold-start")
            else:
                test_supply_id = list(set(test_ids) - set(test_id))
                train_drop_id = list(set(train_id) - set(train_ids))
                test_supply_frequency = df_drug_new.loc[test_supply_id, 'frequency'].iloc[0]
                train_drop_frequency = df_drug_new.loc[train_drop_id, 'frequency'].iloc[0]
                if train_drop_frequency == 'high':
                    high_train_nums -= 1
                    high_train_count -= 1
                elif train_drop_frequency == 'mid':
                    mid_train_nums -= 1
                    mid_train_count -= 1
                elif train_drop_frequency == 'low':
                    low_train_nums -= 1
                    low_train_count -= 1

                if test_supply_frequency == 'high':
                    high_test_nums += 1
                    high_test_count += 1
                elif test_supply_frequency == 'mid':
                    mid_test_nums += 1
                    mid_test_count += 1
                elif test_supply_frequency == 'low':
                    low_test_nums += 1
                    low_test_count += 1

                test_labels.add(label)
                train_drug_ids -= set(train_drop_id)
                test_drug_ids |= set(test_supply_id)


    print("Train Drug Numbers: ", len(train_drug_ids))
    print("Test Drug Numbers: ", len(test_drug_ids))

    print("High Train Drug Numbers: ", high_train_nums)
    print("Mid Train Drug Numbers: ", mid_train_nums)
    print("Low Train Drug Numbers: ", low_train_nums)

    print("High Test Drug Numbers: ", high_test_nums)
    print("Mid Test Drug Numbers: ", mid_test_nums)
    print("Low Test Drug Numbers: ", low_test_nums)

    return train_drug_ids, test_drug_ids

def validate_labels(db_name="./Dataset1/version9/new_event.db", table_columns=['ddi_mechanism', 'ddi_action', 'drugA', 'drugB']):
    import sqlite3

    conn = sqlite3.connect(db_name)
    extraction = pd.read_sql('select * from extraction;', conn)
    mechanism = extraction[table_columns[0]]
    action = extraction[table_columns[1]]

    d_event = []
    for i in range(len(mechanism)):
        d_event.append(mechanism[i] + " " + action[i])
    count = {}
    for i in d_event:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1

    return count

# count_1 = validate_labels(db_name="./Dataset1/version9/new_event.db", table_columns=['ddi_mechanism', 'ddi_action', 'drugA', 'drugB'])
# print(len(count_1.keys()))
#
# count_2 = validate_labels(db_name="./Dataset1/version6/new_event.db", table_columns=['mechanism', 'action', 'drugA', 'drugB'])
# print(len(count_2.keys()))
#
# print(count_2.keys() - count_1.keys())

def get_binary_feature_matrix(feature_name, df_drug):
    """Creates a binary matrix indicating feature presence for each drug."""
    all_feature = []
    drug_list = np.array(df_drug[feature_name]).tolist()
    for i in drug_list:
        if not isinstance(i, str):
            continue
        for each_feature in i.split('|'):
            if each_feature not in all_feature:
                all_feature.append(each_feature)

    feature_matrix = np.zeros((len(drug_list), len(all_feature)), dtype=float)
    df_feature = pd.DataFrame(feature_matrix, columns=all_feature)
    for i in range(len(drug_list)):
        drug_features_str = df_drug[feature_name].iloc[i]
        if not isinstance(drug_features_str, str):
            continue
        for each_feature in drug_features_str.split('|'):
            if each_feature in df_feature.columns:
                df_feature.loc[i, each_feature] = 1
    return np.array(df_feature)

def split_task1(db_name, event_num, seed=0):
    # In test set, both drug_id1 and drug_id2 must already exist (appear) in the training set.
    # In train set, ratio is 80%, and the rest 20% is used for test set.
    cur, conn = connect_to_db(db_name)
    df_event = get_drug_data(conn, 'event')
    df_event = df_event.sample(frac=1, random_state=seed).reset_index(drop=True)

    df_event_number = get_drug_data(conn, 'event_number')
    event_to_index = {event: idx for idx, event in enumerate(df_event_number['event'].unique())}
    print("event number: "+ str(len(event_to_index)))

    train_events_size = int(len(df_event) * 0.6)
    train_drug_ids = set()
    train_event_labels = set()
    train_events = []
    test_events = []
    test_event_labels = set()
    for idx, row in df_event.iterrows():
        drug1 = row['drug_id1']
        drug2 = row['drug_id2']

        if len(train_events) < train_events_size:
            train_events.append(row)
            train_drug_ids.update([drug1, drug2])
            train_event_labels.add(event_to_index[row['interaction']])
            df_event.at[idx, 'train_task1'] = 1
            df_event.at[idx, 'test_task1'] = 0
        else:
            if drug1 in train_drug_ids and drug2 in train_drug_ids:
                test_events.append(row)
                test_event_labels.add(event_to_index[row['interaction']])
                df_event.at[idx, 'train_task1'] = 0
                df_event.at[idx, 'test_task1'] = 1
            else:
                train_events.append(row)
                train_drug_ids.update([drug1, drug2])
                train_event_labels.add(event_to_index[row['interaction']])
                df_event.at[idx, 'train_task1'] = 1
                df_event.at[idx, 'test_task1'] = 0

    # Convert lists to DataFrames
    df_train = pd.DataFrame(train_events)
    df_test = pd.DataFrame(test_events)
    print(len(df_train))
    print(len(df_test))

    # Check for missing labels in train and test sets
    all_labels = set(range(0, event_num))
    missing_train_labels = all_labels - train_event_labels
    missing_test_labels = all_labels - test_event_labels
    print("---------------- Task 1 -----------------")
    print("Missing Train Labels:", missing_train_labels)
    print("Missing Test Labels:", missing_test_labels)

    return df_event

def split_task2(db_name, event_num):

    df_drug_new, df_label_new, df_event, cur, conn = summary_drug_label_information(db_name)
    df_label_sorted = sort_label_count(df_drug_new, df_label_new)
    train_drug_ids, test_drug_ids = label_grouping(df_label_sorted, df_drug_new, df_event)

    df_event = assign_train_labels(train_drug_ids, df_event, name='train_task2')
    df_event = assign_task2_labels(train_drug_ids, test_drug_ids, df_event, name='test_task2')
    print(df_event['train_task2'].value_counts())
    print(df_event['test_task2'].value_counts())

    # Check for missing labels after reclassification
    all_labels = set(range(0, event_num))
    missing_train_labels = verify_label_coverage(df_event, 'train_task2', all_labels)
    missing_task2_labels = verify_label_coverage(df_event, 'test_task2', all_labels)

    print("---------------- Task 2 -----------------")
    print("Missing Train Labels:", missing_train_labels)
    print("Missing Task 2 Labels:", missing_task2_labels)

    return df_event

def split_task3(db_name, event_num):
    df_drug_new, df_label_new, df_event, cur, conn = summary_drug_label_information(db_name)
    df_label_sorted = sort_label_count(df_drug_new, df_label_new)

    train_drug_ids, test_drug_ids = label_grouping(df_label_sorted, df_drug_new, df_event)

    df_event = assign_train_labels(train_drug_ids, df_event, name='train_task3')
    df_event = assign_task3_labels(test_drug_ids, df_event, name='test_task3')
    print(df_event['train_task3'].value_counts())
    print(df_event['test_task3'].value_counts())

    # Check for missing labels after reclassification
    all_labels = set(range(0, event_num))
    missing_train_labels = verify_label_coverage(df_event, 'train_task3', all_labels)
    missing_task3_labels = verify_label_coverage(df_event, 'test_task3', all_labels)

    print("---------------- Task 3 -----------------")
    print("Missing Train Labels:", missing_train_labels)
    print("Missing Task 3 Labels:", missing_task3_labels)

    return df_event


if __name__ == "__main__":

    small_event_num = 65
    medium_event_num = 64
    special_event_num = 64
    large_event_num = 100
    ratio = 0.95  # train, test drug ratio
    random_seed = 42  # split dataset

    df_small_event1 = split_task1(db_name='../../Dataset1/version10/event_small.db', event_num=small_event_num, seed=random_seed)
    df_small_event2 = split_task2(db_name='../../Dataset1/version10/event_small.db', event_num=small_event_num)
    df_small_event3 = split_task3(db_name='../../Dataset1/version10/event_small.db', event_num=small_event_num)

    df_medium_event1 = split_task1(db_name='../../Dataset1/version10/event_medium.db', event_num=medium_event_num, seed=random_seed)
    df_medium_event2 = split_task2(db_name='../../Dataset1/version10/event_medium.db', event_num=medium_event_num)
    df_medium_event3 = split_task3(db_name='../../Dataset1/version10/event_medium.db', event_num=medium_event_num)

    df_large_event1 = split_task1(db_name='../../Dataset1/version10/event_large.db', event_num=large_event_num, seed=random_seed)
    df_large_event2 = split_task2(db_name='../../Dataset1/version10/event_large.db', event_num=large_event_num)
    df_large_event3 = split_task3(db_name='../../Dataset1/version10/event_large.db', event_num=large_event_num)

    df_special_event1 = split_task1(db_name='../../Dataset1/version10/event_special.db', event_num=special_event_num, seed=random_seed)
    df_special_event2 = split_task2(db_name='../../Dataset1/version10/event_special.db', event_num=special_event_num)
    df_special_event3 = split_task3(db_name='../../Dataset1/version10/event_special.db', event_num=special_event_num)