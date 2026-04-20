import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator

def draw_loss_curve(train_losses, val_losses, save_path='loss_curve.png', k=1, task='task1'):
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2, alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2, alpha=0.7)
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Loss Curve', fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig(save_path + 'loss_curve_' + task + '_' + str(k) + '.png')
    plt.clf()
    plt.close()

def draw_whole_confusion_matrix(y_true, y_pred, event_num, task, saved_path):

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(event_num))
    classes = [f'{i}' for i in range(event_num)]

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'}, annot_kws={"size": 5, "color": "black"})
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(np.arange(len(classes)) + 0.5, classes)
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
    for i in range(len(classes)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))
    plt.savefig(saved_path + task + '_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.figure(figsize=(20, 18))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转换为百分比

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Oranges',
                cbar_kws={'label': 'Percentage (%)'}, annot_kws={"size": 5, "color": "black"})

    plt.title('Normalized Confusion Matrix (%)', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(ticks=np.arange(len(classes)) + 0.5, labels=classes)
    plt.yticks(ticks=np.arange(len(classes)) + 0.5, labels=classes, rotation=0)

    for i in range(len(classes)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=2))

    plt.tight_layout()
    plt.savefig(saved_path + task + '_normalized_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

def draw_sub_confusion_matrix(y_true, y_pred, event_num, task, saved_path):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(event_num))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    classes_all = [f'{i}' for i in range(event_num)]

    num_figures = 1

    if event_num == 65:
        num_figures = 5  # 13 * 13 per figure
    elif event_num == 64:
        num_figures = 8  # 8 * 8 per figure
    elif event_num == 100:
        num_figures = 10  # 10 * 10 per figure

    step_size = event_num // num_figures

    # 3. Loop to generate sub-figures
    for k in range(num_figures):
        # Calculate start and end indices for this block
        start_idx = k * step_size
        end_idx = start_idx + step_size

        # Slice the matrices (Get the diagonal block)
        sub_cm = cm[start_idx:end_idx, :]
        sub_cm_norm = cm_norm[start_idx:end_idx, :]
        sub_classes = classes_all[start_idx:end_idx]

        plt.figure(figsize=(32, 12))

        current_max = sub_cm.max()
        vmax_count = current_max if current_max > 0 else 10

        ax = sns.heatmap(sub_cm, annot=True, fmt='d', cmap='Blues',
                         cbar_kws={'label': 'Count'}, annot_kws={"size": 8, "color": "black"},
                         vmin=0, vmax=vmax_count)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)  # Tick font size (Numbers)
        cbar.set_label('Count', fontsize=18)  # Label font size (Text)

        plt.title(f'Confusion Matrix (Part {k + 1}/{num_figures}) - Classes {start_idx}-{end_idx - 1}', fontsize=18)
        plt.ylabel('True Label', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=18)
        plt.xticks(np.arange(len(classes_all)) + 0.5, classes_all, fontsize=16, rotation=0)
        plt.yticks(np.arange(len(sub_classes)) + 0.5, sub_classes, fontsize=16, rotation=0)

        for i in range(len(sub_classes)):
            plt.gca().add_patch(plt.Rectangle((start_idx + i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        filename_count = f'{task}_part{k + 1}_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(os.path.join(saved_path, filename_count), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

        ## Normalized Confusion Matrix

        plt.figure(figsize=(32, 12))
        ax = sns.heatmap(sub_cm_norm, annot=True, fmt='.2f', cmap='Oranges',
                    cbar_kws={'label': 'Percentage (%)'},
                    annot_kws={"size": 8, "color": "black"},
                    vmin=0, vmax=100)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Percentage (%)', fontsize=18)

        plt.title(f'Normalized Matrix (Part {k + 1}/{num_figures}) - Classes {start_idx}-{end_idx - 1}', fontsize=18)
        plt.ylabel('True Label', fontsize=18)
        plt.xlabel('Predicted Label', fontsize=18)

        plt.xticks(np.arange(len(classes_all)) + 0.5, classes_all, fontsize=14,  rotation=0)
        plt.yticks(np.arange(len(sub_classes)) + 0.5, sub_classes, fontsize=14,  rotation=0)

        for i in range(len(sub_classes)):
            plt.gca().add_patch(plt.Rectangle((start_idx + i, i), 1, 1, fill=False, edgecolor='red', lw=2))

        filename_norm = f'{task}_part{k + 1}_normalized_confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(os.path.join(saved_path, filename_norm), dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    print(f"Saved {num_figures} parts for task: {task}")


def draw_sub_class_accuracy_curve(y_true, y_pred, event_num, task, saved_path):
    """
    Calculates global accuracy, but draws the accuracy curves in split sub-figures
    (parts) similar to the confusion matrix logic.
    """
    classes_all = np.arange(event_num)
    cm = confusion_matrix(y_true, y_pred, labels=classes_all)
    class_totals = cm.sum(axis=1)
    class_accuracies = cm.diagonal() / (class_totals + 1e-10) * 100

    # Calculate Global Mean Accuracy (only for classes that actually had samples)
    valid_classes_mask = class_totals > 0
    if np.any(valid_classes_mask):
        global_mean_acc = np.mean(class_accuracies[valid_classes_mask])
    else:
        global_mean_acc = 0.0

    num_figures = 1
    if event_num == 65:
        num_figures = 5  # 13 classes per figure
    elif event_num == 64:
        num_figures = 8  # 8 classes per figure
    elif event_num == 100:
        num_figures = 10  # 10 classes per figure

    step_size = event_num // num_figures

    for k in range(num_figures):
        start_idx = k * step_size
        end_idx = start_idx + step_size

        sub_classes = classes_all[start_idx:end_idx]
        sub_accuracies = class_accuracies[start_idx:end_idx]

        plt.figure(figsize=(12, 6))
        plt.plot(sub_classes, sub_accuracies, marker='o', linestyle='-', color='b',
                 linewidth=2, markersize=8, label='Class Accuracy')
        plt.axhline(y=global_mean_acc, color='r', linestyle='--', linewidth=2,
                    label=f'Global Mean ({global_mean_acc:.2f}%)')

        plt.title(f'Accuracy Profile (Part {k + 1}/{num_figures}) - Classes {start_idx}-{end_idx - 1}', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xlabel('Class Index', fontsize=12)

        plt.xticks(sub_classes, fontsize=10)
        plt.ylim(0, 105)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)

        filename = f'{task}_part{k + 1}_accuracy_curve.png'
        save_file = os.path.join(saved_path, filename)

        plt.tight_layout()
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close()

    print(f"Saved {num_figures} sub-accuracy curves for task: {task}")


def draw_sub_class_epoch_accuracy(class_acc_history, event_num, saved_path, task, k=None):
    """
    Draws the accuracy change of classes over epochs, split into multiple sub-figures
    to avoid overcrowding.
    """

    history_arr = np.array(class_acc_history)
    epochs = range(1, history_arr.shape[0] + 1)

    num_figures = 1
    if event_num == 65:
        num_figures = 5
    elif event_num == 64:
        num_figures = 8
    elif event_num == 100:
        num_figures = 10

    step_size = event_num // num_figures

    for p in range(num_figures):
        start_idx = p * step_size
        end_idx = start_idx + step_size

        sub_history = history_arr[:, start_idx:end_idx]
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, sub_history.shape[1]))

        for j in range(sub_history.shape[1]):
            real_class_idx = start_idx + j
            plt.plot(epochs, sub_history[:, j], label=f'Class {real_class_idx}',
                     color=colors[j], linewidth=2, alpha=0.8)

        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Validation Accuracy (%)', fontsize=14)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        title = f'Class Accuracy History (Part {p + 1}/{num_figures}) - Classes {start_idx}-{end_idx - 1}'
        if k is not None:
            title += f' (Fold {k})'
        plt.title(title, fontsize=16)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.ylim(0, 105)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Classes")

        suffix = f'_fold{k}' if k is not None else ''
        filename = f'{task}_dynamic_acc_part{p + 1}{suffix}.png'
        save_full_path = os.path.join(saved_path, filename)

        plt.tight_layout()
        plt.savefig(save_full_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved {num_figures} dynamic accuracy history parts for task: {task}")

