import numpy as np
import pandas as pd
import os
import argparse
import random
from utils.metric import compute_params, plot_roc_curves
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

def xgb_shap_transform_scale(original_shap_values, Y_pred, which):    
    untransformed_base_value = original_shap_values.base_values[-1]    #Computing the original_explanation_distance to construct the distance_coefficient later on    
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]    
    base_value = expit(untransformed_base_value) #Computing the distance between the model_prediction and the transformed base_value    
    distance_to_explain = Y_pred[which] - base_value    #The distance_coefficient is the ratio between both distances which will be used later on    
    distance_coefficient = original_explanation_distance / distance_to_explain    #Transforming the original shapley values to the new scale    
    shap_values_transformed = original_shap_values / distance_coefficient    #Finally resetting the base_value as it does not need to be transformed    
    shap_values_transformed.base_values = base_value    
    shap_values_transformed.data = original_shap_values.data    #Now returning the transformed array    
    return shap_values_transformed

train_data = pd.read_excel('./dataset/train.xlsx')
test_data = pd.read_excel('./dataset/test.xlsx')
test_trues = test_data['label'].values

true_outcomes = train_data['label'].values
columns = train_data.columns[6:]
top_features = ['RV_SV/KG','RV_CO/KG','RV_CO', 'LV_CO','RV_ESA','4CV_WED','4CV_LED','LV_SV/KG','LV_CO/KG','4CV_Area']

def xg_visualization():
    with open('./train_models/xgboost.pkl', 'rb') as file:
        model = pickle.load(file)
        preds = model.predict_proba(train_data.values[:,6:])[:, 1]
        
        explainer = shap.TreeExplainer(model, feature_names=columns)
        shap_values = explainer(train_data.values[:,6:],y=true_outcomes)

        shap_columns = shap_values.feature_names
        new_columns_order = top_features + [col for col in shap_columns if col not in top_features]
        new_index_order = [shap_columns.index(col) for col in new_columns_order]

        shap_values.feature_names = new_columns_order
        x = shap_values.data
        shap_values.data = x[:,new_index_order]
        x = shap_values.values
        shap_values.values = x[:,new_index_order]
        
        train_shap_values = explainer.shap_values(train_data.values[:,6:])
        print(train_shap_values.shape)
        train_shap_sums = np.abs(train_shap_values).mean(axis=0)
        train_shap_sum =  train_shap_sums*100/train_shap_sums.sum()
        train_importance_df = pd.DataFrame([columns, train_shap_sum.tolist()]).T
        train_importance_df.columns = ['feature', 'shap_importance']
        train_importance_df = train_importance_df.sort_values('shap_importance', ascending=False)[:10].values
        with open('./results/shap_importance.txt','w') as f:
            f.write(str(train_importance_df))
        # print(train_importance_df)

        if False:
            feature_importances = model.get_booster().get_score(importance_type='weight')
            sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
            new_importances = []
            for item in sorted_importances:
                item = list(item)
                item[0] = columns[int(item[0][1:])]
                new_importances.append(item)

            features, importance_values = zip(*new_importances)
            
            plt.figure(figsize=(12, 8), dpi=300)
            plt.barh(range(len(features)), importance_values, align='center',color='red')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance in XGBoost Model')
            plt.gca().invert_yaxis() 
            plt.savefig('./visuals/xgboost_feature_importance.jpg')
            plt.close()

        if False:
            features, importance_values = zip(*train_importance_df)
            plt.figure(figsize=(12, 8), dpi=300)
            plt.barh(range(len(features)), importance_values, align='center',color='red')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance in SHAP Model')
            plt.gca().invert_yaxis() 
            plt.savefig('./visuals/shap_feature_importance.jpg')
            plt.close()

        if False:
            plt.figure(figsize=(12, 8), dpi=300)
            shap.summary_plot(train_shap_values, train_data.values[:,6:], max_display=10, plot_size=(12,8), feature_names=columns, show=False)
            # plt.scatter([],[],s=0.5)
            plt.savefig('./results/shap_values.jpg')
            plt.close()

        if True:
            plt.figure(figsize=(12, 20), dpi=300)
            
            shap.plots.heatmap(shap_values, max_display=11, show=False, plot_width=12, instance_order=shap_values.abs.max(-1))
            plt.scatter([],[],s=1)
            plt.savefig('./visuals/shap_heatmaps.jpg')
            plt.close()

        if False:
            plt.figure(figsize=(12, 20), dpi=300)
            shap.plots.bar(shap_values, max_display=11, show=False)
            plt.scatter([],[],s=1)
            plt.grid(False)
            plt.savefig('./visuals/shap_bar.jpg')
            plt.close()

        if False:
            #0，346，286，343
            for item, i in enumerate([0,236,435,454]):
                # after_shap_values = xgb_shap_transform_scale(shap_values,preds,i)

                # feature_indices = [after_shap_values.feature_names.index(col) for col in new_columns_order]
                # ordered_shap_values = after_shap_values[:, feature_indices]
                shap.waterfall_plot(shap_values[i], max_display=11, show=False)
                plt.savefig(f'./results/shap_waterfall_t{item}.jpg')
                plt.close()
            # shap.plots.force(explainer.expected_value, train_shap_values[352,:],train_data.iloc[352,5:-1], feature_names=['group','4CV_LED','LV_ED_S1', 'RV_SV_KG','LV_GSL','4CV_Area2','LV_CO/KG','RV_FS_S19','RV_ESA','4CV_WED'], matplotlib=True, show=False)
            # plt.savefig('./shap_force_t352_01.jpg')
def cal_predicts():
    test_data = pd.read_excel('./datasets/NC_test.xlsx')
    test_trues = test_data['label'].values
    threshold = 0.36
    if os.path.exists('./predicts/preds.csv'):
        dat = pd.read_csv('./predicts/preds.csv').values
        baseModel_preds = dat[:,0]
        conventionModel_preds = dat[:,1]
        xgboostModel_preds = dat[:,2]
        efficientModel_preds = dat[:,3]
        test_trues = dat[:,-1]
    else:
        baseModel_preds = test_data['group'].values
        with open('./models/convention.pkl', 'rb') as file:
            convention_model = pickle.load(file)
            conventionModel_preds = convention_model.predict_proba(test_data[['group','UAPI']].values)[:, 1]
        with open('./models/xgboost.pkl', 'rb') as file:
            xgmodel = pickle.load(file)
            data = test_data.values[:,5:-1]
            xgboostModel_preds = xgmodel.predict_proba(data)[:, 1]

        with open(f'./models/efficient_t10_032.pkl', 'rb') as file:
            data = test_data[['group','4CV_LED','LV_ED_S1', 'RV_SV_KG','LV_GSL','4CV_Area2','LV_CO/KG','RV_FS_S19','RV_ESA','4CV_WED']].values
            top_model = pickle.load(file)
            efficientModel_preds = top_model.predict_proba(data)[:, 1]

        baseModel_results = compute_params(baseModel_preds,test_trues)
        nri_base = (baseModel_results[1]+baseModel_results[2])/2
        
        for threshold in range(100):
            threshold = threshold/100
            aucs = []
            fprs = []
            tprs = []
            
            # convModel_results = compute_params(conventionModel_preds,test_trues,threshold=threshold)
            xgboostModel_results = compute_params(xgboostModel_preds,test_trues,threshold=threshold)
            # efficientModel_results = compute_params(efficientModel_preds,test_trues,threshold=threshold)
            nri_effi = (xgboostModel_results[1]+xgboostModel_results[2])/2
            # prob_ = (efficientModel_preds > threshold).astype(int)
            # acc = accuracy_score(test_trues,prob_)
            if nri_effi >= nri_base:
                print(threshold,nri_effi,nri_base)
            print(f'the threshold is {threshold}, eff: {nri_effi}, base:{nri_base}.')

# cal_predicts()
xg_visualization()