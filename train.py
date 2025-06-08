import numpy as np
import pandas as pd
import os
import argparse
import random
from utils.metric import compute_params,shap_plot
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt


random.seed(40) 
np.random.seed(40)

def BaseModel(test_data):
    gt = test_data['label']
    pred = test_data['group']
    results = compute_params(pred,gt)
    return results

def ConventionModel(train_data, test_data, root_dir,  threshold=0.5):
    train_dataset = train_data[['EFW percentile','UAPI','label']].values
    test_dataset = test_data[['EFW percentile','UAPI','label']].values
    X_train, y_train = train_dataset[:,:-1],train_dataset[:,-1].astype(int)
    X_test, y_test = test_dataset[:,:-1],test_dataset[:,-1].astype(int)
    param_grid =  {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10.],  # 对数尺度
    'penalty': ['l2', 'l1', 'elasticnet','none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 5000, 10000,],
    'tol': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    'l1_ratio': [0,0.2,0.5,1]
    }

    model_path = os.path.join(root_dir,'train_models/convention.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            test_model = pickle.load(file)
    else:
        model = LogisticRegression()
        gsearch = GridSearchCV(model, param_grid, n_jobs=8, scoring='accuracy',cv=5)
        gsearch.fit(X_train, y_train)
        best_params = gsearch.best_params_
        print("conventionModel best_score_:", gsearch.best_params_, gsearch.best_score_)
        test_model = LogisticRegression(**best_params)
        test_model.fit(X_train, y_train)

        with open(model_path, 'wb') as file:
            pickle.dump(test_model, file)
        
    answer = test_model.predict_proba(X_test)[:, 1]
    results = compute_params(answer,y_test,threshold)
    return results,test_model

def EfficientModel(train_data, test_data, selected_idx, topn, root_dir,  threshold=0.5):
    train_dataset = train_data[selected_idx[:topn]+['label']]
    test_dataset = test_data[selected_idx[:topn]+['label']]
    train_dataset = train_dataset.values
    test_dataset = test_dataset.values
    
    X_train, y_train = train_dataset[:,:-1],train_dataset[:,-1].astype(int)
    X_test, y_test = test_dataset[:,:-1],test_dataset[:,-1].astype(int)
    # X_train = X_train[:,selected_idx]
    # X_test = X_test[:,selected_idx]
    param_grid =  {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1., 5., 10.], 
    'penalty': ['l2', 'l1', 'elasticnet'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'max_iter': [100, 500, 1000, 5000, 10000,20000],
    'tol': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    # 'l1_ratio': [0,0.2,0.5,1]
    }

    model_path = os.path.join(root_dir,f'train_models/efficient_t{topn}.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            test_model = pickle.load(file)
    else:
        model = LogisticRegression()
        gsearch = GridSearchCV(model, param_grid, n_jobs=8, scoring='accuracy',cv=5)
        gsearch.fit(X_train, y_train)
        best_params = gsearch.best_params_
        print("efficientModel best_score_:", gsearch.best_params_, gsearch.best_score_)

        test_model = LogisticRegression(**best_params)
        test_model.fit(X_train, y_train)

        with open(model_path, 'wb') as file:
            pickle.dump(test_model, file)
        
    answer = test_model.predict_proba(X_test)[:, 1]
    results = compute_params(answer,y_test,threshold)
    return results,test_model

def EfficientModelxg(train_data, test_data, selected_idx, root_dir,  threshold=0.5):
    train_dataset = train_data[selected_idx+['label']]
    test_dataset = test_data[selected_idx+['label']]
    train_dataset = train_dataset.values
    test_dataset = test_dataset.values
    
    X_train, y_train = train_dataset[:,:-1],train_dataset[:,-1].astype(int)
    X_test, y_test = test_dataset[:,:-1],test_dataset[:,-1].astype(int)

    param_grid = {
        'max_depth': [6, 7, 8],
        'n_estimators': [30, 50, 100, 300, 2000],
        'learning_rate': [0.1, 0.4, 0.01, 0.05, 0.5],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "min_child_weight": [2, 4, 6, 7, 8],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "subsample": [0.6, 0.8, 0.9],
        }
    model_path = './train_models/effixgboost.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            test_model = pickle.load(file)
    else:
        # gsearch = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='error'),
        #                             param_grid, n_jobs=-1, scoring='accuracy', cv=5)
        # gsearch.fit(X_train, y_train)
        # best_params = gsearch.best_params_
        # print("Best parameters:", best_params)
        # print("Best score:", gsearch.best_score_)
        # best_params = {'colsample_bytree': 0.7, 'gamma': 0.4, 'learning_rate': 0.1, 'max_depth': 6, 'min_child_weight': 4, 'n_estimators': 30, 'reg_alpha': 0.0001, 'reg_lambda': 0.01, 'subsample': 0.9}
        test_model = XGBClassifier(use_label_encoder=False, eval_metric='error')
        test_model.fit(X_train, y_train)

        with open(f'./train_models/effixgboost.pkl', 'wb') as file:
            pickle.dump(test_model, file)

    answer = test_model.predict_proba(X_test)[:, 1]
    results = compute_params(answer,y_test,threshold)
    return results,test_model
    
def XgboostModel(train_data, test_data, root_dir, threshold=0.5,start_col=5,end_col=-1):
    train_dataset = train_data.values
    test_dataset = test_data.values
    X_train, y_train = train_dataset[:,start_col:],train_dataset[:,4].astype(int)
    X_test, y_test = test_dataset[:,start_col:],test_dataset[:,4].astype(int)

    param_grid = {
        'max_depth': [6, 7, 8],
        'n_estimators': [30, 50, 100, 300, 2000],
        'learning_rate': [0.1, 0.4, 0.01, 0.05, 0.5],
        "gamma": [0.0, 0.1, 0.2, 0.3, 0.4],
        "reg_alpha": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "reg_lambda": [0.0001, 0.001, 0.01, 0.1, 1, 100],
        "min_child_weight": [2, 4, 6, 7, 8],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
        "subsample": [0.6, 0.8, 0.9],
        }
    model_path = os.path.join(root_dir,'train_models/xgboost.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            test_model = pickle.load(file)
    else:
        gsearch = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='error'),
                                    param_grid, n_jobs=-1, scoring='accuracy', cv=5)
        gsearch.fit(X_train, y_train)
        best_params = gsearch.best_params_
        with open('./fuck.txt','w') as f:
            f.write(f'Best parameters:{best_params}，score:{gsearch.best_score_}')
        # print("Best parameters:", best_params)
        # print("Best score:", gsearch.best_score_)
        test_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric='error')
        test_model.fit(X_train, y_train)

        with open(model_path, 'wb') as file:
            pickle.dump(test_model, file)
        
    answer = test_model.predict_proba(X_test)[:, 1]
    results = compute_params(answer,y_test,threshold)
    return results,test_model

def FeatureSelect(train_data, test_data, columns, model, top_n=5,start_col=5,end_col=-1,root_dir='./'):

    train_dataset = train_data.values
    test_dataset = test_data.values
    X_train, y_train = train_dataset[:,start_col:],train_dataset[:,4].astype(int)
    X_test, y_test = test_dataset[:,start_col:],test_dataset[:,4].astype(int)

    explainer = shap.TreeExplainer(model, feature_names=columns)
    train_shap_values = explainer.shap_values(X_train)
    test_shap_values = explainer.shap_values(X_test)

    train_shap_sum = np.abs(train_shap_values).mean(axis=0)
    train_importance_df = pd.DataFrame([columns, train_shap_sum.tolist()]).T
    train_importance_df.columns = ['feature', 'shap_importance']

    test_shap_sum = np.abs(test_shap_values).mean(axis=0)
    test_importance_df = pd.DataFrame([columns, test_shap_sum.tolist()]).T
    test_importance_df.columns = ['feature', 'shap_importance']

    train_importance_df = train_importance_df.sort_values('shap_importance', ascending=False)
    train_importance_df[:10].to_csv(os.path.join(root_dir,'results/train_feature_top10.csv'))

    test_importance_df = test_importance_df.sort_values('shap_importance', ascending=False)
    test_importance_df[:10].to_csv(os.path.join(root_dir,'results/test_feature_top10.csv'))

    # print(f'train:{train_importance_df[:10].index}.')
    # print(f'test:{test_importance_df[:10].index}.')

    # plt.figure(figsize=(12,6))
    # shap.summary_plot(train_shap_values, X_train, max_display=10, feature_names=columns, show=False)
    # plt.gcf().axes[-1].set_aspect(60)
    # plt.gcf().axes[-1].set_box_aspect(60)
    # plt.savefig(os.path.join(root_dir,'results/train_xg_shap.jpg'))
    # plt.close()

    # plt.figure(figsize=(12,6))
    # shap.summary_plot(test_shap_values, X_test, max_display=10, feature_names=columns, show=False)
    # plt.gcf().axes[-1].set_aspect(60)
    # plt.gcf().axes[-1].set_box_aspect(60)
    # plt.savefig(os.path.join(root_dir,'results/test_xg_shap.jpg'))
    # plt.close()
    train_results = [train_shap_values,X_train]
    test_results = [test_shap_values,X_test]

    return list(train_importance_df[:top_n].feature),train_results,test_results

   

if __name__ == "__main__":
    root_dir = './'
    start_col = 6
    dat = pd.read_excel(os.path.join(root_dir,'dataset/Final_data_zscore.xlsx'))
    
    ##dataset split
    Neg = dat[dat['label']==0]
    Pos = dat[dat['label']==1]
    Neg_ML_train = Neg.sample(int(len(Neg)*0.7))
    Neg_ML_test = Neg.drop(Neg_ML_train.index)
    Pos_ML_train = Pos.sample(int(len(Pos)*0.7))
    Pos_ML_test = Pos.drop(Pos_ML_train.index)
    ML_train = pd.concat([Neg_ML_train,Pos_ML_train],axis=0,ignore_index=True)
    ML_test = pd.concat([Neg_ML_test,Pos_ML_test],axis=0,ignore_index=True)
    # ML_train.to_excel(os.path.join(root_dir,'dataset/train.xlsx'),index=None)
    # ML_test.to_excel(os.path.join(root_dir,'dataset/test.xlsx'),index=None)


    baseModel_results = BaseModel(ML_test)
    convModel_results,_ = ConventionModel(ML_train, ML_test,root_dir, threshold=0.5)
    xgboostModel_results,xgmodel = XgboostModel(ML_train,ML_test, root_dir, threshold=0.5, start_col=start_col, end_col=-1)
    selected_features,train_results,test_results = FeatureSelect(ML_train,ML_test,dat.columns[start_col:], xgmodel, top_n=10, start_col=start_col, end_col=-1)

    for i in range(1,11):
        efficientModel_results,_ = EfficientModel(ML_train, ML_test, selected_idx=selected_features,topn=i,root_dir=root_dir, threshold=0.5)
        print(f'model {i} done!')

    shap_plot(train_results[0],train_results[1],dat.columns[start_col:],os.path.join(root_dir,'results/train_xg_shap.jpg'))
    shap_plot(test_results[0],test_results[1],dat.columns[start_col:],os.path.join(root_dir,'results/test_xg_shap.jpg'))


    # aucs = []
    # fprs = []
    # tprs = []
    # model_names = ['Base Model','Conventional Model','Joint Model','Efficient Model']

    # print(f'basemodel auc:{round(baseModel_results[0],4)} ({round(baseModel_results[3][0],4)}, {round(baseModel_results[3][1],4)}), sen:{round(baseModel_results[1],4)} ({round(baseModel_results[4][0],4)},{round(baseModel_results[4][1],4)}),spec:{round(baseModel_results[2],4)} ({round(baseModel_results[5][0],4)},{round(baseModel_results[5][1],4)}), acc:{round(baseModel_results[6],4)} ({round(baseModel_results[7][0],4)},{round(baseModel_results[7][1],4)})')
    # aucs.append([round(baseModel_results[0],4),round(baseModel_results[3][0],4),round(baseModel_results[3][1],4)])
    # fprs.append(baseModel_results[-2])
    # tprs.append(baseModel_results[-1])

    # print(f'convention auc:{round(convModel_results[0],4)} ({round(convModel_results[3][0],4)}, {round(convModel_results[3][1],4)}), sen:{round(convModel_results[1],4)} ({round(convModel_results[4][0],4)},{round(convModel_results[4][1],4)}),spec:{round(convModel_results[2],4)} ({round(convModel_results[5][0],4)},{round(convModel_results[5][1],4)}), acc:{round(convModel_results[6],4)} ({round(convModel_results[7][0],4)},{round(convModel_results[7][1],4)})')
    # aucs.append([round(convModel_results[0],4),round(convModel_results[3][0],4),round(convModel_results[3][1],4)])
    # fprs.append(convModel_results[-2])
    # tprs.append(convModel_results[-1])

    # print(f'xgboost auc:{round(xgboostModel_results[0],4)} ({round(xgboostModel_results[3][0],4)}, {round(xgboostModel_results[3][1],4)}), sen:{round(xgboostModel_results[1],4)} ({round(xgboostModel_results[4][0],4)},{round(xgboostModel_results[4][1],4)}),spec:{round(xgboostModel_results[2],4)} ({round(xgboostModel_results[5][0],4)},{round(xgboostModel_results[5][1],4)}), acc:{round(xgboostModel_results[6],4)} ({round(xgboostModel_results[7][0],4)},{round(xgboostModel_results[7][1],4)})')
    # aucs.append([round(xgboostModel_results[0],4),round(xgboostModel_results[3][0],4),round(xgboostModel_results[3][1],4)])
    # fprs.append(xgboostModel_results[-2])
    # tprs.append(xgboostModel_results[-1])

    # print(f'efficient auc:{round(efficientModel_results[0],4)} ({round(efficientModel_results[3][0],4)}, {round(efficientModel_results[3][1],4)}), sen:{round(efficientModel_results[1],4)} ({round(efficientModel_results[4][0],4)},{round(efficientModel_results[4][1],4)}),spec:{round(efficientModel_results[2],4)} ({round(efficientModel_results[5][0],4)},{round(efficientModel_results[5][1],4)}), acc:{round(efficientModel_results[6],4)} ({round(efficientModel_results[7][0],4)},{round(efficientModel_results[7][1],4)})')
    # aucs.append([round(efficientModel_results[0],4),round(efficientModel_results[3][0],4),round(efficientModel_results[3][1],4)])
    # fprs.append(efficientModel_results[-2])
    # tprs.append(efficientModel_results[-1])

    # plot_roc_curves(aucs,fprs,tprs,model_names,os.path.join(root_dir,'results/result.jpg'))
    
