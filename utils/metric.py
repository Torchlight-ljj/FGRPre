from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score, confusion_matrix
import sklearn.metrics as metrics
import numpy as np
import argparse 
import pickle
import pandas as pd
import os 
import matplotlib.pyplot as plt
from scipy.stats import binom_test
import seaborn as sns
import shap
def plot_roc_curves(auc_values, fpr_values, tpr_values, method_names, save_path):
    if not (len(auc_values) == len(fpr_values) == len(tpr_values) == len(method_names)):
        raise ValueError("Lists should have the same size!")

    if len(auc_values)==4:
        colors = ['blue', 'green', 'red', 'cyan']
    else:
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'lime', 'orange', 'brown', 'olive']

    plt.figure(figsize=(10, 8), dpi=300)
    for i in range(len(auc_values)):
        plt.plot(fpr_values[i], tpr_values[i], label=f'{method_names[i]}: {auc_values[i][0]} ({auc_values[i][1]},{auc_values[i][2]})', color=colors[i % len(colors)], linewidth=2)

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')

    plt.legend(loc='lower right')
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, threshold=0.5):
    prob_ = (y_pred > threshold).astype(int)
    cm = confusion_matrix(y_true, prob_, labels=[0,1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Reds', xticklabels=class_names, yticklabels=class_names, square=True)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def shap_plot(shap_values, X, feature_names,save_path):
    plt.figure(figsize=(12,6))
    shap.summary_plot(shap_values, X, max_display=10, feature_names=feature_names, show=False)
    plt.gcf().axes[-1].set_aspect(60)
    plt.gcf().axes[-1].set_box_aspect(60)
    plt.savefig(save_path)
    plt.close()
    
def compute_params(preds,gts,threshold=0.5):
    rng_seed = 42  # 控制bootstrap采样的随机性
    n_bootstraps=1000
    auc_scores = []
    sensitivity_scores = []
    specificity_scores = []
    acc_scores = []
    rng = np.random.RandomState(rng_seed)
    auc = roc_auc_score(gts, preds)
    fpr, tpr, _ = roc_curve(gts, preds)
    
    prob_ = (preds > threshold).astype(int)
    # acc = accuracy_score(gts,prob_)
    # tn, fp, fn, tp = metrics.confusion_matrix(gts, prob_).ravel()
    # sensitivity0 = tp / (tp + fn)
    # specificity0 = tn / (tn + fp)   

    confidence_internal_acc = []
    confidence_interval_auc = []
    confidence_interval_sensitivity = []
    confidence_interval_specificity = []
    
    for j in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        try:
            indices = rng.randint(0, len(preds), len(preds))
            if len(np.unique(gts)) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue
            score = roc_auc_score(gts[indices], preds[indices])
            auc_scores.append(score)

            prob_ = (preds[indices] > threshold).astype(int)
            accuracy = accuracy_score(gts[indices], prob_)
            acc_scores.append(accuracy)

            tn, fp, fn, tp = metrics.confusion_matrix(gts[indices], prob_).ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)   
            sensitivity_scores.append(sensitivity)
            specificity_scores.append(specificity)
        except Exception as e:
            continue
    sorted_scores = np.array(auc_scores)
    sorted_scores.sort()
    auc = np.median(sorted_scores)
    confidence_interval_auc = [sorted_scores[int(0.025 * len(sorted_scores))],sorted_scores[int(0.975 * len(sorted_scores))]]

    sens_scores = np.array(sensitivity_scores)
    sens_scores.sort()
    sensitivity0 = np.median(sens_scores)
    confidence_interval_sensitivity = [sens_scores[int(0.025 * len(sens_scores))],sens_scores[int(0.975 * len(sens_scores))]]

    spec_scores = np.array(specificity_scores)
    spec_scores.sort()
    specificity0 = np.median(spec_scores)
    confidence_interval_specificity = [spec_scores[int(0.025 * len(spec_scores))],spec_scores[int(0.975 * len(spec_scores))]]

    acc_scores = np.array(acc_scores)
    acc_scores.sort()
    acc = np.median(acc_scores)
    confidence_internal_acc = [acc_scores[int(0.025 * len(acc_scores))],acc_scores[int(0.975 * len(acc_scores))]]
    
    return auc,sensitivity0,specificity0,confidence_interval_auc,confidence_interval_sensitivity,confidence_interval_specificity,acc,confidence_internal_acc,fpr,tpr

def calculate_nri_and_p_value(old_probs, new_probs, true_labels, threshold=0.5):
    if not (len(old_probs) == len(new_probs) == len(true_labels)):
        raise ValueError("All input lists must have the same length.")

    up_correct = up_incorrect = down_correct = down_incorrect = 0

    for old_p, new_p, true_label in zip(old_probs, new_probs, true_labels):
        old_class = old_p >= threshold
        new_class = new_p >= threshold

        if new_class and not old_class:  # 上移
            if true_label == new_class:
                up_correct += 1
            else:
                up_incorrect += 1
        elif old_class and not new_class:  # 下移
            if true_label == new_class:
                down_correct += 1
            else:
                down_incorrect += 1

    nri = (up_correct - up_incorrect) / len(old_probs) + (down_incorrect - down_correct) / len(old_probs)
    p_value = binom_test(up_correct + down_incorrect, len(old_probs), p=0.5)

    return nri, p_value


def calculate_nri(true_outcomes, predictions1, predictions2, threshold):
    # 将预测分为高风险和低风险
    high_risk_1 = predictions1 >= threshold
    high_risk_2 = predictions2 >= threshold

    # 初始化计数器
    up_event, down_event, up_nonevent, down_nonevent = 0, 0, 0, 0

    # 遍历每个样本
    for outcome, risk1, risk2 in zip(true_outcomes, high_risk_1, high_risk_2):
        if outcome:  # 事件发生
            if not risk1 and risk2:
                up_event += 1  # 低风险到高风险
            elif risk1 and not risk2:
                down_event += 1  # 高风险到低风险
        else:  # 事件未发生
            if not risk1 and risk2:
                up_nonevent += 1  # 低风险到高风险
            elif risk1 and not risk2:
                down_nonevent += 1  # 高风险到低风险

    # 计算比率
    n_event = sum(true_outcomes)
    n_nonevent = len(true_outcomes) - n_event
    up_event_rate = up_event / n_event if n_event else 0
    down_event_rate = down_event / n_event if n_event else 0
    up_nonevent_rate = up_nonevent / n_nonevent if n_nonevent else 0
    down_nonevent_rate = down_nonevent / n_nonevent if n_nonevent else 0

    # 计算NRI
    nri = (up_event_rate - down_event_rate) + (down_nonevent_rate - up_nonevent_rate)
    return nri

# def calculate_nri(true_labels, old_predictions, new_predictions):
#     # 计算事件和非事件的索引
#     event_indices = (true_labels == 1)
#     nonevent_indices = (true_labels == 0)

#     # 计算两个模型对于事件和非事件的分类改进
#     improvement_in_events = np.mean(new_predictions[event_indices] > old_predictions[event_indices])
#     improvement_in_nonevents = np.mean(new_predictions[nonevent_indices] < old_predictions[nonevent_indices])

#     # NRI 计算
#     nri = (improvement_in_events - improvement_in_nonevents)
#     return nri

def calculate_nri_se(true_labels, old_predictions, new_predictions,threshold):
    # 这里使用简化的方法来估计标准误差
    # 更精确的方法可能需要更复杂的统计计算
    n = len(true_labels)
    se = np.sqrt((1/n) * (calculate_nri(true_labels, old_predictions, new_predictions,threshold) * (1 - calculate_nri(true_labels, old_predictions, new_predictions,threshold))))
    return se
