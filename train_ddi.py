import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from model import MMIDDI
from test_ddi import test_model, evaluate, save_raw_data
from figure import draw_loss_curve, draw_sub_class_epoch_accuracy
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

min_w = 0.05
max_w = 5.0

def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(0, event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1
    return index_all_class

def oversample_training_data(train_index, label_matrix, event_num, target_samples_per_class=None):
    """
    Performs random oversampling on the training data indices to balance classes.

    Args:
        train_index (np.array): The original indices for the training set.
        label_matrix (np.array): The full label array for all samples.
        event_num (int): The total number of event classes.
        target_samples_per_class (int, optional): The desired number of samples for each class.
                                                   If None, it defaults to the size of the largest class.

    Returns:
        np.array: The new, oversampled array of training indices.
    """
    print("Performing oversampling on training data...")
    original_train_labels = label_matrix[train_index]
    class_counts = np.bincount(original_train_labels, minlength=event_num)

    if target_samples_per_class is None:
        target_samples_per_class = np.max(class_counts)
        print(f"Target samples per class set to the majority class size: {target_samples_per_class}")

    oversampled_indices = [train_index]

    for class_idx in range(event_num):
        current_count = class_counts[class_idx]

        # Check if the class needs oversampling
        if 0 < current_count < target_samples_per_class:
            num_to_add = target_samples_per_class - current_count

            # Find the indices of the samples belonging to this class within the original full dataset
            class_sample_indices = train_index[np.where(original_train_labels == class_idx)[0]]

            # Randomly choose from these existing samples (with replacement) to add more
            if len(class_sample_indices) > 0:
                new_indices = np.random.choice(class_sample_indices, size=num_to_add, replace=True)
                oversampled_indices.append(new_indices)

    # Concatenate all original and new indices and shuffle them
    final_train_index = np.concatenate(oversampled_indices)
    np.random.shuffle(final_train_index)

    print(f"Original training size: {len(train_index)}, Oversampled training size: {len(final_train_index)}")
    return final_train_index

def train_model(model, train_loader, test_loader, class_weights, parameters, k, saved_path, device):

    event_num = parameters['event_num']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    w_decay = parameters['weight_decay']
    smooth_eps = parameters['smooth_eps']

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    if class_weights is not None:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smooth_eps)
    else:
        ce_loss = nn.CrossEntropyLoss(label_smoothing=smooth_eps)

    best_loss = float('inf')

    train_losses = []
    valid_losses = []
    valid_class_acc_history = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()

        for *modalities, labels in tqdm(train_loader):

            if labels.ndim == 2:
                labels_idx = labels.argmax(dim=1).long().to(device)
            else:
                labels_idx = labels.long().to(device)

            optimizer.zero_grad()
            modalities = [m.to(device) for m in modalities]
            outputs, attention_maps = model(*modalities)
            loss1 = ce_loss(outputs, labels_idx)
            # loss2 = fc_loss(outputs, labels.to(device))
            # loss = loss1 + loss2
            loss = loss1
            # loss = loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, class_accs = validate_model(model, test_loader, ce_loss, event_num, device)
        validate_loss = val_loss / len(test_loader)
        valid_losses.append(validate_loss)
        valid_class_acc_history.append(class_accs)

        scheduler.step()

        print(f'    Epoch {epoch + 1}   training loss: {train_loss:.6f}     validation loss: {validate_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), saved_path + 'ddi_best_model' + str(k) + '.pth')
            patience = 30
        else:
            patience -= 1
            if patience == 0:
                break

    return train_losses, valid_losses, valid_class_acc_history


def cross_validation(feature_matrix, label_matrix, clf_type, parameters, saved_path, device):

    event_num = parameters['event_num']
    num_heads = parameters['num_heads']
    batch_size = parameters['batch_size']
    PCA_components = parameters['PCA_components']
    num_layers = parameters['num_layers']
    drop_rate = parameters['drop_rate']
    seed = parameters['seed']
    CV = parameters['CV']
    oversampling = parameters['oversampling']

    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label_matrix, event_num, seed, CV) # Get the 5 Fold index in all events
    matrix = []

    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        feature_matrix = matrix

    for k in range(CV):
        train_index = np.where(index_all_class != k)
        print("train_index", len(train_index[0]))

        if oversampling and len(train_index[0]) > 0:
            train_index = oversample_training_data(train_index[0], label_matrix, event_num)
            print("After oversampling, train_index: ", len(train_index[0]))

        test_index = np.where(index_all_class == k)
        print("test_index: ", len(test_index[0]))

        pred = np.zeros((len(test_index[0]), event_num), dtype=float)

        y_train = label_matrix[train_index]
        y_train_one_hot = (np.arange(event_num) == y_train[:, None]).astype(dtype='float32')
        y_train_one_hot = torch.tensor(y_train_one_hot)

        class_counts = np.bincount(y_train, minlength=event_num)
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum()  # normalize
        class_weights = np.clip(class_weights, min_w, max_w)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

        y_test = label_matrix[test_index]
        y_test_one_hot = (np.arange(event_num) == y_test[:, None]).astype(dtype='float32')
        y_test_one_hot = torch.tensor(y_test_one_hot)

        num_modalites = len(feature_matrix)

        if num_modalites == 3:
            # [smiles1, smiles2, target1, target2, enzyme1, enzyme2]
            x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
            x_train_1 = torch.tensor(feature_matrix[1][train_index], dtype=torch.float32)
            x_train_2 = torch.tensor(feature_matrix[2][train_index], dtype=torch.float32)
            train_dataset = TensorDataset(x_train_0, x_train_1, x_train_2, y_train_one_hot)

            x_test_0 = torch.tensor(feature_matrix[0][test_index], dtype=torch.float32)
            x_test_1 = torch.tensor(feature_matrix[1][test_index], dtype=torch.float32)
            x_test_2 = torch.tensor(feature_matrix[2][test_index], dtype=torch.float32)
            test_dataset = TensorDataset(x_test_0, x_test_1, x_test_2, y_test_one_hot)

        elif num_modalites == 2:
            # [smiles1, smiles2, target1, target2]
            x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
            x_train_1 = torch.tensor(feature_matrix[1][train_index], dtype=torch.float32)
            train_dataset = TensorDataset(x_train_0, x_train_1, y_train_one_hot)

            x_test_0 = torch.tensor(feature_matrix[0][test_index], dtype=torch.float32)
            x_test_1 = torch.tensor(feature_matrix[1][test_index], dtype=torch.float32)
            test_dataset = TensorDataset(x_test_0, x_test_1, y_test_one_hot)

        elif num_modalites == 4:
            x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
            x_train_1 = torch.tensor(feature_matrix[1][train_index], dtype=torch.float32)
            x_train_2 = torch.tensor(feature_matrix[2][train_index], dtype=torch.float32)
            x_train_3 = torch.tensor(feature_matrix[3][train_index], dtype=torch.float32)
            train_dataset = TensorDataset(x_train_0, x_train_1, x_train_2, x_train_3, y_train_one_hot)

            x_test_0 = torch.tensor(feature_matrix[0][test_index], dtype=torch.float32)
            x_test_1 = torch.tensor(feature_matrix[1][test_index], dtype=torch.float32)
            x_test_2 = torch.tensor(feature_matrix[2][test_index], dtype=torch.float32)
            x_test_3 = torch.tensor(feature_matrix[3][test_index], dtype=torch.float32)
            test_dataset = TensorDataset(x_test_0, x_test_1, x_test_2, x_test_3, y_test_one_hot)

        else:
            # [smiles1, smiles2]
            x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
            train_dataset = TensorDataset(x_train_0, y_train_one_hot)

            x_test_0 = torch.tensor(feature_matrix[0][test_index], dtype=torch.float32)
            test_dataset = TensorDataset(x_test_0, y_test_one_hot)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        if clf_type == 'MMIDDI':
            model = MMIDDI(drug_channels=PCA_components, num_heads=num_heads, num_layers=num_layers, drop_rate=drop_rate, num_modalities=len(feature_matrix), output_dim=event_num)
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(device)
            train_losses, valid_losses, valid_class_acc_history = train_model(model, train_loader, test_loader, class_weights_tensor, parameters, k, saved_path, device)
            pred += test_model(model, test_loader, event_num, device)

        save_raw_data(
            saved_path,
            f'task1_history_fold{k}',
            train_losses=train_losses,
            valid_losses=valid_losses,
            valid_class_acc_history=valid_class_acc_history,
            epochs=np.arange(1, len(train_losses) + 1)
        )

        draw_loss_curve(train_losses, valid_losses, save_path=saved_path, k=k, task='task1')
        draw_sub_class_epoch_accuracy(valid_class_acc_history, event_num, saved_path, task='task1', k=k)

        pred_type = np.argmax(pred, axis=1)
        y_true = np.hstack((y_true, y_test))
        y_pred = np.hstack((y_pred, pred_type))
        y_score = np.vstack((y_score, pred))

    result_keep, result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, task='task1', saved_path=saved_path)
    return result_keep, result_all, result_eve


def train_model_task2(model, train_loader, test_loader, class_weights, parameters, saved_path, device):

    event_num = parameters['event_num']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    w_decay = parameters['weight_decay']
    smooth_eps = parameters['smooth_eps']
    sigma = parameters['sigma']
    epsilon = parameters['epsilon']

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    if class_weights is not None:
        ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=smooth_eps)
    else:
        ce_loss = nn.CrossEntropyLoss(label_smoothing=smooth_eps)

    best_loss = float('inf')

    train_losses = []
    valid_losses = []
    valid_class_acc_history = []

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        model.train()

        for *modalities, labels in tqdm(train_loader):

            if labels.ndim == 2:
                labels_idx = labels.argmax(dim=1).long().to(device)
            else:
                labels_idx = labels.long().to(device)

            modalities = [m.to(device) for m in modalities]
            for m in modalities:
                if m.dtype == torch.float32:
                    m.requires_grad = True

            outputs, attention_maps = model(*modalities)
            loss = ce_loss(outputs, labels_idx)

            model.zero_grad()
            loss.backward(retain_graph=True)

            # Generate adversarial inputs (FGSM)
            adv_modalities = []
            for m in modalities:
                if m.dtype == torch.float32:
                    grad_sign = m.grad.sign()
                    adv_modalities.append(m + epsilon * grad_sign)
                else:
                    adv_modalities.append(m)

            adv_outputs, attention_maps = model(*adv_modalities)
            adv_loss = ce_loss(adv_outputs, labels_idx)

            total_loss = (1 - sigma) * loss + sigma * adv_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        val_loss, class_accs = validate_model(model, test_loader, ce_loss, event_num, device)
        validate_loss = val_loss / len(test_loader)
        valid_losses.append(validate_loss)
        valid_class_acc_history.append(class_accs)

        scheduler.step()

        print(f'    Epoch {epoch + 1}   training loss: {train_loss:.6f}     validation loss: {validate_loss:.6f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), saved_path + 'ddi_best_model.pth')
            patience = 20
        else:
            patience -= 1
            if patience == 0:
                break

    return train_losses, valid_losses, valid_class_acc_history


def cross_validation_task2_3(feature_matrix, label_matrix, clf_type, parameters, train_drug, record_label, fold, saved_path, device):

    event_num = parameters['event_num']
    num_heads = parameters['num_heads']
    batch_size = parameters['batch_size']
    PCA_components = parameters['PCA_components']
    num_layers = parameters['num_layers']
    drop_rate = parameters['drop_rate']
    vector_size = parameters['vector_size']
    oversampling = parameters['oversampling']

    y_true2 = np.array([])
    y_pred2 = np.array([])
    y_score2 = np.zeros((0, event_num), dtype=float)

    y_true3 = np.array([])
    y_pred3 = np.array([])
    y_score3 = np.zeros((0, event_num), dtype=float)

    matrix = []
    if type(feature_matrix) != list:
        matrix.append(feature_matrix)
        feature_matrix = matrix

    is_train_drug = [False] * vector_size
    for i in train_drug:
        is_train_drug[i] = True
    train_index = []
    test2_index = []
    test3_index = []
    count = 0
    for i in record_label:
        if (is_train_drug[i[0]] and is_train_drug[i[1]]):
            train_index.append(count)
        elif (is_train_drug[i[0]]) or (is_train_drug[i[1]]):
            test2_index.append(count)
        else:
            test3_index.append(count)
        count = count + 1

    train_index = np.array(train_index)
    test2_index = np.array(test2_index)
    test3_index = np.array(test3_index)

    print("train_index: ", len(train_index))
    print("test2_index: ", len(test2_index))
    print("test3_index: ", len(test3_index))

    if oversampling and len(train_index) > 0:
        train_index = oversample_training_data(train_index, label_matrix, event_num)
        print("After oversampling, train_index: ", len(train_index))

    pred2 = np.zeros((test2_index.shape[0], event_num), dtype=float)
    pred3 = np.zeros((test3_index.shape[0], event_num), dtype=float)

    y_train = label_matrix[train_index]
    y_train_one_hot = (np.arange(event_num) == y_train[:, None]).astype(dtype='float32')
    y_train_one_hot = torch.tensor(y_train_one_hot)

    y_test2 = label_matrix[test2_index]
    y_test_2_one_hot = (np.arange(event_num) == y_test2[:, None]).astype(dtype='float32')
    y_test_2_one_hot = torch.tensor(y_test_2_one_hot)

    y_test3 = label_matrix[test3_index]
    y_test_3_one_hot = (np.arange(event_num) == y_test3[:, None]).astype(dtype='float32')
    y_test_3_one_hot = torch.tensor(y_test_3_one_hot)

    num_modalites = len(feature_matrix)

    if num_modalites == 3:
        # [smiles1, smiles2, target1, target2, enzyme1, enzyme2]
        x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
        x_train_1 = torch.tensor(feature_matrix[1][train_index], dtype=torch.float32)
        x_train_2 = torch.tensor(feature_matrix[2][train_index], dtype=torch.float32)
        train_dataset = TensorDataset(x_train_0, x_train_1, x_train_2, y_train_one_hot)

        x_test_2_0 = torch.tensor(feature_matrix[0][test2_index], dtype=torch.float32)
        x_test_2_1 = torch.tensor(feature_matrix[1][test2_index], dtype=torch.float32)
        x_test_2_2 = torch.tensor(feature_matrix[2][test2_index], dtype=torch.float32)
        test2_dataset = TensorDataset(x_test_2_0, x_test_2_1, x_test_2_2, y_test_2_one_hot)

        x_test_3_0 = torch.tensor(feature_matrix[0][test3_index], dtype=torch.float32)
        x_test_3_1 = torch.tensor(feature_matrix[1][test3_index], dtype=torch.float32)
        x_test_3_2 = torch.tensor(feature_matrix[2][test3_index], dtype=torch.float32)
        test3_dataset = TensorDataset(x_test_3_0, x_test_3_1, x_test_3_2, y_test_3_one_hot)

    elif num_modalites == 2:
        x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
        x_train_1 = torch.tensor(feature_matrix[1][train_index], dtype=torch.float32)
        train_dataset = TensorDataset(x_train_0, x_train_1, y_train_one_hot)

        x_test_2_0 = torch.tensor(feature_matrix[0][test2_index], dtype=torch.float32)
        x_test_2_1 = torch.tensor(feature_matrix[1][test2_index], dtype=torch.float32)
        test2_dataset = TensorDataset(x_test_2_0, x_test_2_1, y_test_2_one_hot)

        x_test_3_0 = torch.tensor(feature_matrix[0][test3_index], dtype=torch.float32)
        x_test_3_1 = torch.tensor(feature_matrix[1][test3_index], dtype=torch.float32)
        test3_dataset = TensorDataset(x_test_3_0, x_test_3_1, y_test_3_one_hot)

    elif num_modalites == 4:
        # [smiles1, smiles2, target1, target2, enzyme1, enzyme2, pathway1, pathway2]
        x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
        x_train_1 = torch.tensor(feature_matrix[1][train_index], dtype=torch.float32)
        x_train_2 = torch.tensor(feature_matrix[2][train_index], dtype=torch.float32)
        x_train_3 = torch.tensor(feature_matrix[3][train_index], dtype=torch.float32)
        train_dataset = TensorDataset(x_train_0, x_train_1, x_train_2, x_train_3, y_train_one_hot)

        x_test_2_0 = torch.tensor(feature_matrix[0][test2_index], dtype=torch.float32)
        x_test_2_1 = torch.tensor(feature_matrix[1][test2_index], dtype=torch.float32)
        x_test_2_2 = torch.tensor(feature_matrix[2][test2_index], dtype=torch.float32)
        x_test_2_3 = torch.tensor(feature_matrix[3][test2_index], dtype=torch.float32)
        test2_dataset = TensorDataset(x_test_2_0, x_test_2_1, x_test_2_2, x_test_2_3, y_test_2_one_hot)

        x_test_3_0 = torch.tensor(feature_matrix[0][test3_index], dtype=torch.float32)
        x_test_3_1 = torch.tensor(feature_matrix[1][test3_index], dtype=torch.float32)
        x_test_3_2 = torch.tensor(feature_matrix[2][test3_index], dtype=torch.float32)
        x_test_3_3 = torch.tensor(feature_matrix[3][test3_index], dtype=torch.float32)
        test3_dataset = TensorDataset(x_test_3_0, x_test_3_1, x_test_3_2, x_test_3_3, y_test_3_one_hot)

    else:
        x_train_0 = torch.tensor(feature_matrix[0][train_index], dtype=torch.float32)
        train_dataset = TensorDataset(x_train_0, y_train_one_hot)

        x_test_2_0 = torch.tensor(feature_matrix[0][test2_index], dtype=torch.float32)
        test2_dataset = TensorDataset(x_test_2_0, y_test_2_one_hot)

        x_test_3_0 = torch.tensor(feature_matrix[0][test3_index], dtype=torch.float32)
        test3_dataset = TensorDataset(x_test_3_0, y_test_3_one_hot)

    class_counts = np.bincount(y_train, minlength=event_num)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()  # normalize
    class_weights = np.clip(class_weights, min_w, max_w)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test2_loader = DataLoader(test2_dataset, batch_size=batch_size, shuffle=False)
    test3_loader = DataLoader(test3_dataset, batch_size=batch_size, shuffle=False)

    if clf_type == 'MMIDDI':
        model = MMIDDI(drug_channels=PCA_components, num_heads=num_heads, num_layers=num_layers, drop_rate=drop_rate, num_modalities=len(feature_matrix), output_dim=event_num)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
        train2_losses, valid2_losses, valid2_class_acc_history = train_model_task2(model, train_loader, test2_loader, class_weights_tensor, parameters, saved_path, device)
        draw_loss_curve(train2_losses, valid2_losses, save_path=saved_path, k=fold, task='task2')
        pred2 += test_model(model, test2_loader, event_num, device)
        pred3 += test_model(model, test3_loader, event_num, device)

    save_raw_data(
        saved_path,
        f'task2_history_fold{fold}',
        train_losses=train2_losses,
        valid_losses=valid2_losses,
        valid_class_acc_history=valid2_class_acc_history,
        epochs=np.arange(1, len(train2_losses) + 1)
    )

    draw_sub_class_epoch_accuracy(valid2_class_acc_history, event_num, saved_path, task='task2', k=fold)

    pred_type2 = np.argmax(pred2, axis=1)
    y_true2 = np.hstack((y_true2, y_test2))
    y_pred2 = np.hstack((y_pred2, pred_type2))
    y_score2 = np.vstack((y_score2, pred2))

    pred_type3 = np.argmax(pred3, axis=1)
    y_true3 = np.hstack((y_true3, y_test3))
    y_pred3 = np.hstack((y_pred3, pred_type3))
    y_score3 = np.vstack((y_score3, pred3))

    result_keep2, result_all2, result_eve2 = evaluate(y_pred2, y_score2, y_true2, event_num, task='task2', saved_path=saved_path)
    result_keep3, result_all3, result_eve3 = evaluate(y_pred3, y_score3, y_true3, event_num, task='task3', saved_path=saved_path)
    return result_keep2, result_all2, result_eve2, result_keep3, result_all3, result_eve3

def cross_validation_special(feature_matrix, label_matrix, df_event, pair_map, task, clf_type, parameters, saved_path, device):

    event_num = parameters['event_num']
    num_heads = parameters['num_heads']
    batch_size = parameters['batch_size']
    PCA_components = parameters['PCA_components']
    num_layers = parameters['num_layers']
    drop_rate = parameters['drop_rate']
    oversampling = parameters['oversampling']

    if not isinstance(feature_matrix, list):
        feature_matrix = [feature_matrix]

    train_indices = []
    test_indices = []

    col_train = f'train_{task}'
    col_test  = f'test_{task}'

    required_cols = ['drug_name1', 'drug_name2', col_train, col_test]
    for col in required_cols:
        if col not in df_event.columns:
            raise ValueError(
                f"Column '{col}' not found in event_tasks table. Columns present: {df_event.columns.tolist()}")

    matched_count = 0
    for idx, row in df_event.iterrows():
        d1 = str(row['drug_name1']).strip()
        d2 = str(row['drug_name2']).strip()

        current_key = (d1, d2)
        reversed_key = (d2, d1)

        matrix_index = None

        if current_key in pair_map:
            matrix_index = pair_map[current_key]
        elif reversed_key in pair_map:
            matrix_index = pair_map[reversed_key]

        if matrix_index is not None:
            matched_count += 1
            if row[col_train] == 1:
                train_indices.append(matrix_index)
            elif row[col_test] == 1:
                test_indices.append(matrix_index)

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    if oversampling and len(train_indices) > 0:
        train_indices = oversample_training_data(train_indices, label_matrix, event_num)
        print("After oversampling, train_index: ", len(train_indices))

    print(f"Total rows in event_tasks: {len(df_event)}")
    print(f"Successfully matched: {matched_count}")
    print(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")

    if len(train_indices) == 0 and len(test_indices) == 0:
        raise ValueError("0 matches found. Please check if drug names in 'extraction' match 'event_tasks' exactly.")

    y_train = label_matrix[train_indices]
    y_test = label_matrix[test_indices]

    print("train size:", len(y_train), "test size:", len(y_test))

    x_train_tensors = [torch.tensor(fm[train_indices], dtype=torch.float32) for fm in feature_matrix]
    y_train_one_hot = torch.tensor((np.arange(event_num) == y_train[:, None]).astype(np.float32))

    x_test_tensors = [torch.tensor(fm[test_indices], dtype=torch.float32) for fm in feature_matrix]
    y_test_one_hot = torch.tensor((np.arange(event_num) == y_test[:, None]).astype(np.float32))

    class_counts = np.bincount(y_train, minlength=event_num)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum()  # normalize
    class_weights = np.clip(class_weights, min_w, max_w)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(*x_train_tensors, y_train_one_hot)
    test_dataset = TensorDataset(*x_test_tensors, y_test_one_hot)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pred_accumulator = np.zeros((len(y_test), event_num), dtype=float)

    if clf_type == 'MMIDDI':
        model = MMIDDI(drug_channels=PCA_components, num_heads=num_heads, num_layers=num_layers, drop_rate=drop_rate, num_modalities=len(feature_matrix), output_dim=event_num)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

        if task == 'task1':
            train_losses, valid_losses, valid_class_acc_history = train_model(model, train_loader, test_loader, class_weights_tensor, parameters, k=0, saved_path=saved_path, device=device)
        else:
            train_losses, valid_losses, valid_class_acc_history = train_model_task2(model, train_loader, test_loader, class_weights_tensor, parameters, saved_path=saved_path, device=device)

        pred_accumulator += test_model(model, test_loader, event_num, device)

    save_raw_data(
        saved_path,
        f'{task}_special_history',
        train_losses=train_losses,
        valid_losses=valid_losses,
        valid_class_acc_history=valid_class_acc_history,
        epochs=np.arange(1, len(train_losses) + 1)
    )

    draw_loss_curve(train_losses, valid_losses, save_path=saved_path, k=0, task=task)
    draw_sub_class_epoch_accuracy(valid_class_acc_history, event_num, saved_path, task=task, k=0)

    y_pred = np.argmax(pred_accumulator, axis=1)
    y_score = pred_accumulator
    y_true = y_test

    result_keep, result_all, result_eve = evaluate(y_pred, y_score, y_true, event_num, task=task, saved_path=saved_path)
    return result_keep, result_all, result_eve


def validate_model(model, validate_loader, criterion, event_num, device):
    model.eval()
    pred_list, true_list = [], []
    running_loss = 0.0

    with torch.no_grad():
        for *modalities, labels in validate_loader:
            if labels.ndim == 2:
                labels_idx = labels.argmax(dim=1).long().to(device)
            else:
                labels_idx = labels.long().to(device)
            modalities = [m.to(device) for m in modalities]
            outputs, attention_maps = model(*modalities)
            loss = criterion(outputs, labels_idx)
            pred_list.append(outputs.detach().cpu().numpy())
            true_list.append(labels_idx.detach().cpu().numpy())
            running_loss += loss.item()

    all_preds = np.vstack(pred_list)
    all_trues = np.concatenate(true_list)

    pred_classes = np.argmax(all_preds, axis=1)
    true_classes = all_trues

    cm = confusion_matrix(true_classes, pred_classes, labels=np.arange(event_num))
    class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-10) * 100

    return running_loss, class_accuracies