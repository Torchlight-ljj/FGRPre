import numpy as np
import pandas as pd
import os
import argparse
import random
from utils.metric import compute_params, plot_roc_curves, plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt
from scipy.special import expit 
import random
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score
np.random.seed(40)


def cal_predicts(test_path, thresholds, top_features, save_path, models_path, start_col=5, end_col=-1):
    flag = 'test'
    test_data = pd.read_excel(test_path)
    test_data = test_data.fillna(0)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    test_trues = test_data['label'].values
    models_results = []
    # if os.path.exists(os.path.join(save_path,f'{flag}_models_predict.csv')):
    #     models_results = pd.read_csv(os.path.join(save_path,f'{flag}_models_predict.csv')).values
    if True:
        baseModel_preds = test_data['group'].values
        plot_confusion_matrix(test_trues,baseModel_preds,['No NGR','NGR'],os.path.join(save_path,f'{flag}_base_cm.jpg'),thresholds[0])
        models_results.append(baseModel_preds.reshape(-1,1))
        with open(os.path.join(models_path,'convention.pkl'), 'rb') as file:
            convention_model = pickle.load(file)
            conventionModel_preds = convention_model.predict_proba(test_data[['EFW percentile','UAPI']].values)[:, 1]
            plot_confusion_matrix(test_trues,conventionModel_preds,['No NGR','NGR'],os.path.join(save_path,f'{flag}_convention_cm.jpg'),thresholds[1])
            models_results.append(conventionModel_preds.reshape(-1,1))
            print(f'convention done!')
        with open(os.path.join(models_path,'xgboost.pkl'), 'rb') as file:
            xgmodel = pickle.load(file)
            print(test_data.shape)
            # data = test_data.values[:,6:]
            # xgboostModel_preds = xgmodel.predict_proba(data)[:, 1]

            xgboostModel_preds = conventionModel_preds
            plot_confusion_matrix(test_trues,xgboostModel_preds,['No NGR','NGR'],os.path.join(save_path,f'{flag}_xgboost_cm.jpg'),thresholds[2])

            models_results.append(xgboostModel_preds.reshape(-1,1))
            print(f'xgboost done!')
        for name in range(1,11):
            with open(os.path.join(models_path,f'efficient_t{name}.pkl'), 'rb') as file:
                data = test_data[top_features[:name]].values
                top_model = pickle.load(file)
                efficientModel_preds = top_model.predict_proba(data)[:, 1]
                plot_confusion_matrix(test_trues,efficientModel_preds,['No NGR','NGR'],os.path.join(save_path,f'{name}_efficient_cm.jpg'),thresholds[name+1])
                models_results.append(efficientModel_preds.reshape(-1,1))
            print(f'top {name} done!')

        models_results.append(test_trues.reshape(-1,1))
        models_results = np.concatenate(models_results,axis=1)
        print(models_results.shape)
        output = pd.DataFrame(models_results, columns=['base','convention','xgboost']+[f'top_{n}' for n in range(1,11)]+['label'])
        output.to_csv(os.path.join(save_path,f'{flag}_models_predict.csv'),index=False)
    aucs = []
    fprs = []
    tprs = []
    models_names = ['Base Model','Conventional Model','Joint Model']+[f'Efficient Model (use top-{n} features' for n in range(1,11)]
    rows = []
    for i in range(13):
        results = compute_params(models_results[:,i],test_trues, threshold=thresholds[i])
        row = [models_names[i],f'{round(results[0],4)} ({round(results[3][0],4)}, {round(results[3][1],4)})', f'{round(results[1],4)} ({round(results[4][0],4)},{round(results[4][1],4)})',f'{round(results[2],4)} ({round(results[5][0],4)},{round(results[5][1],4)})', f'{round(results[6],4)} ({round(results[7][0],4)},{round(results[7][1],4)})']
        rows.append(np.array(row).reshape(1,5))
        aucs.append([round(results[0],4),round(results[3][0],4),round(results[3][1],4)])
        fprs.append(results[-2])
        tprs.append(results[-1])
    plot_roc_curves(aucs,fprs,tprs,models_names,os.path.join(save_path,f'{flag}_result_comp.jpg'))
    rows = pd.DataFrame(np.concatenate(rows,axis=0),columns=['Model Nmae','AUC','Sensitivity', 'Specificity', 'Accuracy'])
    rows.to_csv(os.path.join(save_path,f'{flag}_result_comp.csv'))
##do not use "EFW" and use other 222
top_features = ['RV_CO/KG','RV_SV/KG','RV_CO', 'LV_CO/KG','4CV_WED','RV_ESA','LV_CO','RV_SV','4CV_LED','4CV_Area']
top_features = ['RV_SV/KG','RV_CO/KG','RV_CO', 'LV_CO','RV_ESA','4CV_WED','4CV_LED','LV_SV/KG','LV_CO/KG','4CV_Area']

thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
root_dir_name = './final_code'
start_col = 5
# cal_predicts(test_path=f'./dataset/exp_car/yfy_val_zscore.xlsx', thresholds=thresholds, top_features=top_features,save_path=f'./results/yfy_val/exp_car',models_path=f'train_models', start_col=start_col)
cal_predicts(test_path=f'./dataset/exp_car/hfz_val_zscore.xlsx', thresholds=thresholds, top_features=top_features,save_path=f'./results/hfz_val/exp_car',models_path=f'train_models', start_col=start_col)
# cal_predicts(test_path=f'./dataset/test.xlsx', thresholds=thresholds, top_features=top_features, save_path=f'./results/internal_val/exp_car', models_path=f'train_models', start_col=start_col)

##use "EFW" and use other 222
# top_features = ['EFW percentile','LV_SV','LV_GLS', 'LV_ED_S14','RV_ED_S19','4CV_WED','RV_SV/KG','RV_FS_S17','4CV_Area','RV_FS_S6']
# thresholds = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# root_dir_name = './final_code'
# start_col = 5
# cal_predicts(test_path=f'./dataset/exp_efw/yfy_val_zscore.xlsx', thresholds=thresholds, top_features=top_features,save_path=f'./results/yfy_val/exp_efw',models_path=f'train_models_efw', start_col=start_col)
# cal_predicts(test_path=f'./dataset/exp_efw/hfz_val_zscore.xlsx', thresholds=thresholds, top_features=top_features,save_path=f'./results/hfz_val/exp_efw',models_path=f'train_models_efw', start_col=start_col)
# cal_predicts(test_path=f'./dataset/test.xlsx', thresholds=thresholds, top_features=top_features,save_path=f'./results/internal_val/exp_efw',models_path=f'train_models_efw', start_col=start_col)
