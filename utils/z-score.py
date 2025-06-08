import pandas as pd
import numpy as np
import argparse

def main(data_path, norm_path, use_cols, mode, save_path):
    df = pd.read_excel(data_path, usecols=['week','EFW percentile','UAPI','label','group']+use_cols)
    stats = pd.read_excel(norm_path)
    excluded_columns = ['EFW percentile','group','label']
    df['week'] = df['week'].apply(np.floor)
    columns_to_process = df.columns.drop(excluded_columns)
    grouped = df.groupby('week')
    for week, week_data in grouped:
        week_stat = stats[stats['week']==week]
        week_data = df[df['week']==week]
        for column in columns_to_process:
            if column != 'week':  
                mean = week_stat[column+'_mean'].values[0]
                sd = week_stat[column+'_std'].values[0]           
                z_scores = (week_data[column] - mean) / sd
                df.loc[week_data.index, column] = z_scores
    if mode == 'load':
        return df.values
    else:
        df.to_excel(save_path, index=False)
        print('Norm Finished!')

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--data', required=True, type=str, help='input the path of data file')
    parse.add_argument('--norm', default='../dataset/WeeklyStats.xlsx', type=str, help='input the path of norm file')
    #use efw
    # parse.add_argument('--cols', default=['LV_SV','LV_GLS', 'LV_ED_S14','RV_ED_S19','4CV_WED','RV_SV/KG','RV_FS_S17','4CV_Area','RV_FS_S6'], type=list, help='input the column names that need to be normalized')
    #not use efw
    parse.add_argument('--cols', default= ['RV_SV/KG','RV_CO/KG','RV_CO', 'LV_CO','RV_ESA','4CV_WED','4CV_LED','LV_SV/KG','LV_CO/KG','4CV_Area'], type=list, help='input the column names that need to be normalized')

    parse.add_argument('--mode', default='file', type=str, help='output mode: load: output to cpu, file: save to path')
    parse.add_argument('--save_path', default='', type=str, help='if mode is file, define this arg' )
    args = parse.parse_args()

    main(args.data,args.norm,args.cols,args.mode,args.save_path)
    