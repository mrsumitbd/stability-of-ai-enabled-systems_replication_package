import sys, os, time
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
# module_path = os.path.abspath(os.path.join('..', 'src'))
# if module_path not in sys.path:
#     sys.path.append(module_path)
from utility import *

def project_wise_analysis(proj_df, config_var, metric):
    # print("\n")
    # print(proj_df['Project'].unique()[0])
    # print("\n")
    ### OS

    if config_var.lower() == 'os':
        sub_df_os = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                 (proj_df['Python'] == '3.7') &
                                 (proj_df['Hardware'] == 'amd64')) |
                                (proj_df['OS'] == 'MacOS') |
                                (proj_df['OS'] == 'Windows')]

        comp_df_OS = pd.DataFrame(sub_df_os.groupby('OS')[metric].mean())
        comp_df_OS['pct_change'] = [
            calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['Linux-Xenial'][0]),
            calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['MacOS'][0]),
            calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['Windows'][0])
        ]

        comp_df_OS['p-value'] = [1.0,
                                 extract_p_value(sub_df_os.loc[proj_df['OS'] == 'Linux-Xenial'][metric],
                                                 sub_df_os.loc[proj_df['OS'] == 'MacOS'][metric]),
                                 extract_p_value(sub_df_os.loc[proj_df['OS'] == 'Linux-Xenial'][metric],
                                                 sub_df_os.loc[proj_df['OS'] == 'Windows'][metric])
                                 ]

        comp_df_OS['delta_val'] = [extract_cliffs_delta(sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric], sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric])[0],
                                   extract_cliffs_delta(sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric], sub_df_os.loc[sub_df_os['OS'] == 'MacOS'][metric])[0],
                                   extract_cliffs_delta(sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric], sub_df_os.loc[sub_df_os['OS'] == 'Windows'][metric])[0]]

        comp_df_OS['delta_size'] = [extract_cliffs_delta(sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric], sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric])[1],
                                   extract_cliffs_delta(sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric], sub_df_os.loc[sub_df_os['OS'] == 'MacOS'][metric])[1],
                                   extract_cliffs_delta(sub_df_os.loc[sub_df_os['OS'] == 'Linux-Xenial'][metric], sub_df_os.loc[sub_df_os['OS'] == 'Windows'][metric])[1]]
        comp_df_OS.to_csv(f"/Users/musfiqurrahman/Documents/variability_deltas/{proj_df['Project'].unique()[0]}_{config_var}_{metric}.csv", index=False)
        #print(comp_df_OS)
        return comp_df_OS

    ### Dist
    elif config_var.lower() == 'dist':
        sub_df_dist = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                   (proj_df['Python'] == '3.7') &
                                   (proj_df['Hardware'] == 'amd64')) |
                                  (proj_df['OS'] == 'Linux-Bionic') |
                                  (proj_df['OS'] == 'Linux-Focal')]

        comp_df_dist = pd.DataFrame(sub_df_dist.groupby('OS')[metric].mean())

        comp_df_dist['pct_change'] = [
            calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Bionic'][0]),
            calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Focal'][0]),
            calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Xenial'][0])
        ]

        comp_df_dist['p-value'] = [
            extract_p_value(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric],
                            sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Bionic'][metric]),
            extract_p_value(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric],
                            sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Focal'][metric]), 1.0
        ]

        comp_df_dist['delta_val'] = [extract_cliffs_delta(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric], sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric])[0],
                                   extract_cliffs_delta(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric], sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Bionic'][metric])[0],
                                   extract_cliffs_delta(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric], sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Focal'][metric])[0]]

        comp_df_dist['delta_size'] = [extract_cliffs_delta(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric], sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric])[1],
                                   extract_cliffs_delta(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric], sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Bionic'][metric])[1],
                                   extract_cliffs_delta(sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Xenial'][metric], sub_df_dist.loc[sub_df_dist['OS'] == 'Linux-Focal'][metric])[1]]

        comp_df_dist.to_csv(f"/Users/musfiqurrahman/Documents/variability_deltas/{proj_df['Project'].unique()[0]}_{config_var}_{metric}.csv", index=False)
        #print(comp_df_dist)
        return comp_df_dist


    ### Hardware
    elif config_var.lower() == 'hardware':
        sub_df_hw = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                 (proj_df['Python'] == '3.7') &
                                 (proj_df['Hardware'] == 'amd64')) |
                                (proj_df['Hardware'] == 'arm64')]

        comp_df_hw = pd.DataFrame(sub_df_hw.groupby('Hardware')[metric].mean())

        comp_df_hw['pct_change'] = [
            calculate_pct_change(comp_df_hw.loc['amd64'][0], comp_df_hw.loc['amd64'][0]),
            calculate_pct_change(comp_df_hw.loc['amd64'][0], comp_df_hw.loc['arm64'][0])
        ]

        comp_df_hw['p-value'] = [1.0,
                                 extract_p_value(sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric],
                                                 sub_df_hw.loc[sub_df_hw['Hardware'] == 'arm64'][metric])
                                 ]
        comp_df_hw['delta_val'] = [extract_cliffs_delta(sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric], sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric])[0],
                                   extract_cliffs_delta(sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric], sub_df_hw.loc[sub_df_hw['Hardware'] == 'arm64'][metric])[0]]

        comp_df_hw['delta_size'] = [extract_cliffs_delta(sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric], sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric])[1],
                                   extract_cliffs_delta(sub_df_hw.loc[sub_df_hw['Hardware'] == 'amd64'][metric], sub_df_hw.loc[sub_df_hw['Hardware'] == 'arm64'][metric])[1]]
        comp_df_hw.to_csv(f"/Users/musfiqurrahman/Documents/variability_deltas/{proj_df['Project'].unique()[0]}_{config_var}_{metric}.csv", index=False)
        #print(comp_df_hw)
        return comp_df_hw

    ### Python Versions
    elif config_var.lower() == 'python':
        sub_df_py = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                 (proj_df['Python'] == '3.7') &
                                 (proj_df['Hardware'] == 'amd64')) |
                                (proj_df['Python'] == '3.8') |
                                (proj_df['Python'] == '3.6')]

        comp_df_py = pd.DataFrame(sub_df_py.groupby('Python')[metric].mean())

        comp_df_py['pct_change'] = [
            calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.6'][0]),
            calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.7'][0]),
            calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.8'][0])
        ]

        comp_df_py['p-value'] = [
            extract_p_value(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric],
                            sub_df_py.loc[sub_df_py['Python'] == '3.6'][metric]),
            1.0,
            extract_p_value(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric],
                            sub_df_py.loc[sub_df_py['Python'] == '3.8'][metric])
        ]

        comp_df_py['delta_val'] = [extract_cliffs_delta(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric], sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric])[0],
                                   extract_cliffs_delta(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric], sub_df_py.loc[sub_df_py['Python'] == '3.6'][metric])[0],
                                   extract_cliffs_delta(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric], sub_df_py.loc[sub_df_py['Python'] == '3.8'][metric])[0]]

        comp_df_py['delta_size'] = [extract_cliffs_delta(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric], sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric])[1],
                                   extract_cliffs_delta(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric], sub_df_py.loc[sub_df_py['Python'] == '3.6'][metric])[1],
                                   extract_cliffs_delta(sub_df_py.loc[sub_df_py['Python'] == '3.7'][metric], sub_df_py.loc[sub_df_py['Python'] == '3.8'][metric])[1]]
        comp_df_py.to_csv(f"/Users/musfiqurrahman/Documents/variability_deltas/{proj_df['Project'].unique()[0]}_{config_var}_{metric}.csv", index=False)
        #print(comp_df_py)
        return comp_df_py

    else:
        raise ValueError("Invalid value for the variable config_var.")

def add_credits(df):
    credit_list = []
    for idx, row in df.iterrows():
        if row['OS'].startswith("Linux"):
            credit_list.append(10*(row['Processing_Time']/60))
        elif row['OS'] == 'MacOS':
            credit_list.append(50*(row['Processing_Time']/60))
        else:
            credit_list.append(20*(row['Processing_Time']/60))

    df['Processing_Time'] = df['Processing_Time']/60
    df['Expense'] = credit_list
    #df['Credits'] = credit_list
    #df['Expense'] = [100 * 0.0006 * c for c in df['Credits'].tolist()]
    return df


def stat_significance_report(p_val_df):
    if p_val_df.index.name == 'Python':
        sig_dict = {"3.6" : False,
                    "3.8" : False}

        if (p_val_df.loc['3.6']['pct_change'] != 0) and (p_val_df.loc['3.6']['p-value'] < 0.05) and (p_val_df.loc['3.6']['delta_size'] != 'negligible'):
            sig_dict["3.6"] = True
        if (p_val_df.loc['3.8']['pct_change'] != 0) and (p_val_df.loc['3.8']['p-value'] < 0.05) and (p_val_df.loc['3.8']['delta_size'] != 'negligible'):
            sig_dict["3.8"] = True

        return sig_dict

    elif p_val_df.index.name == "OS" and "MacOS" in p_val_df.index:
        sig_dict = {"MacOS" : False,
                    "Windows" : False}

        if (p_val_df.loc['MacOS']['pct_change'] != 0) and (p_val_df.loc['MacOS']['p-value'] < 0.05) and (p_val_df.loc['MacOS']['delta_size'] != 'negligible'):
            sig_dict["MacOS"] = True
        if (p_val_df.loc['Windows']['pct_change'] != 0) and (p_val_df.loc['Windows']['p-value'] < 0.05) and (p_val_df.loc['Windows']['delta_size'] != 'negligible'):
            sig_dict["Windows"] = True

        return sig_dict

    elif p_val_df.index.name == "OS" and "MacOS" not in p_val_df.index:
        sig_dict = {"Linux-Bionic" : False,
                    "Linux-Focal" : False}

        if (p_val_df.loc['Linux-Bionic']['pct_change'] != 0) and (p_val_df.loc['Linux-Bionic']['p-value'] < 0.05) and (p_val_df.loc['Linux-Bionic']['delta_size'] != 'negligible'):
            sig_dict["Linux-Bionic"] = True
        if (p_val_df.loc['Linux-Focal']['pct_change'] != 0) and (p_val_df.loc['Linux-Focal']['p-value'] < 0.05) and (p_val_df.loc['Linux-Focal']['delta_size'] != 'negligible'):
            sig_dict["Linux-Focal"] = True

        return sig_dict

    if p_val_df.index.name == 'Hardware':
        sig_dict = {"arm64" : False}

        if (p_val_df.loc['arm64']['pct_change'] != 0) and (p_val_df.loc['arm64']['p-value'] < 0.05) and (p_val_df.loc['arm64']['delta_size'] != 'negligible'):
            sig_dict["arm64"] = True
        return sig_dict


def combine_sig_result(full_df, config_var, metric):
    sig_combined_full = {"3.6": [],
                         "3.8" : [],
                         'MacOS' : [],
                         "Windows" : [],
                         "Linux-Bionic" : [],
                            "Linux-Focal" : [],
                         "arm64" : []}

    if config_var.lower() == 'python':
        sig_combined = {"3.6" : [],
                            "3.8" : []}

    elif config_var.lower() == 'os':
        sig_combined = {"MacOS" : [],
                            "Windows" : []}

    elif config_var.lower() == 'dist':
        sig_combined = {"Linux-Bionic" : [],
                            "Linux-Focal" : []}

    elif config_var.lower() == 'hardware':
        sig_combined = {"arm64" : []}

    for proj in full_df['Project'].unique().tolist():
        for key, value in stat_significance_report(project_wise_analysis(full_df.loc[df['Project']==proj], config_var, metric)).items():
            sig_combined_full[key].append((proj, int(value)))
            sig_combined[key].append(int(value))


    if config_var.lower() == 'os':
        print(f"Only Linux vs MacOS: {set([item for item in sig_combined_full['MacOS'] if item[1] == 1]).difference(set([item for item in sig_combined_full['Windows'] if item[1] == 1]))}")
        print(f"Only Linux vs Windows: {set([item for item in sig_combined_full['Windows'] if item[1] == 1]).difference(set([item for item in sig_combined_full['MacOS'] if item[1] == 1]))}")
        print(f"Both OSes: {set([item for item in sig_combined_full['MacOS'] if item[1] == 1]).intersection(set([item for item in sig_combined_full['Windows'] if item[1] == 1]))}")

    elif config_var.lower() == 'dist':
        print(f"Only Linux-Xenial vs Linux-Bionic: {set([item for item in sig_combined_full['Linux-Bionic'] if item[1] == 1]).difference(set([item for item in sig_combined_full['Linux-Focal'] if item[1] == 1]))}")
        print(f"Only Linux-Xenial vs Linux-Focal: {set([item for item in sig_combined_full['Linux-Focal'] if item[1] == 1]).difference(set([item for item in sig_combined_full['Linux-Bionic'] if item[1] == 1]))}")
        print(f"Both Distributions: {set([item for item in sig_combined_full['Linux-Focal'] if item[1] == 1]).intersection(set([item for item in sig_combined_full['Linux-Bionic'] if item[1] == 1]))}")

    elif config_var.lower() == 'python':
        print(f"Only 3.7 vs 3.6: {set([item for item in sig_combined_full['3.6'] if item[1] == 1]).difference(set([item for item in sig_combined_full['3.8'] if item[1] == 1]))}")
        print(f"Only 3.7 vs 3.8: {set([item for item in sig_combined_full['3.8'] if item[1] == 1]).difference(set([item for item in sig_combined_full['3.6'] if item[1] == 1]))}")
        print(f"Both Versions: {set([item for item in sig_combined_full['3.6'] if item[1] == 1]).intersection(set([item for item in sig_combined_full['3.8'] if item[1] == 1]))}")

    elif config_var.lower() == 'hardware':
        print(f"{[item for item in sig_combined_full['arm64'] if item[1] == 1]}")

    return sig_combined


def generate_summary_report(sig_dict):
    if "3.6" in sig_dict.keys():
        print(f"{sig_combined['3.6'].count(1)} out of {len(sig_combined['3.6'])} or {round(100 * np.mean(sig_combined['3.6']), 2)}% projects show a statistically significant difference between Python 3.6 and Python 3.7.\n")
        print(f"{sig_combined['3.8'].count(1)} out of {len(sig_combined['3.8'])} or {round(100 * np.mean(sig_combined['3.8']), 2)}% projects show a statistically significant difference between Python 3.8 and Python 3.7.\n")

        count_both = 0

        for i in range(len(sig_combined['3.6'])):
            if sig_combined['3.6'][i] == 1 == sig_combined['3.8'][i]:
                count_both += 1
        print(f"{count_both} out of {len(sig_combined['3.6'])} or {round(100 * (count_both/len(sig_combined['3.6'])), 2)}% projects show a statistically significant difference between both pairs.\n")

    elif "Windows" in sig_dict.keys():
        print(f"{sig_combined['Windows'].count(1)} out of {len(sig_combined['Windows'])} or {round(100 * np.mean(sig_combined['Windows']), 2)}% projects show a statistically significant difference between Linux (Xenial) and Windows.\n")
        print(f"{sig_combined['MacOS'].count(1)} out of {len(sig_combined['MacOS'])} or {round(100 * np.mean(sig_combined['MacOS']), 2)}% projects show a statistically significant difference between Linux (Xenial) and MacOS.\n")

        count_both = 0

        for i in range(len(sig_combined['MacOS'])):
            if sig_combined['MacOS'][i] == 1 == sig_combined['Windows'][i]:
                count_both += 1
        print(f"{count_both} out of {len(sig_combined['MacOS'])} or {round(100 * (count_both/len(sig_combined['MacOS'])), 2)}% projects show a statistically significant difference between both pairs.\n")

    elif "Linux-Bionic" in sig_dict.keys():
        print(f"{sig_combined['Linux-Bionic'].count(1)} out of {len(sig_combined['Linux-Bionic'])} or {round(100 * np.mean(sig_combined['Linux-Bionic']), 2)}% projects show a statistically significant difference between Linux-Xenial and Linux-Bionic.\n")
        print(f"{sig_combined['Linux-Focal'].count(1)} out of {len(sig_combined['Linux-Focal'])} or {round(100 * np.mean(sig_combined['Linux-Focal']), 2)}% projects show a statistically significant difference between Linux-Xenial and Linux-Focal.\n")

        count_both = 0

        for i in range(len(sig_combined['Linux-Focal'])):
            if sig_combined['Linux-Focal'][i] == 1 == sig_combined['Linux-Bionic'][i]:
                count_both += 1
        print(f"{count_both} out of {len(sig_combined['Linux-Focal'])} or {round(100 * (count_both/len(sig_combined['Linux-Focal'])), 2)}% projects show a statistically significant difference between both pairs.\n")

    elif "arm64" in sig_dict.keys():
        print(f"{sig_combined['arm64'].count(1)} out of {len(sig_combined['arm64'])} or {round(100 * np.mean(sig_combined['arm64']), 2)}% projects show a statistically significant difference between amd64 and arm64.\n")


def generate_report_df(full_df):
    df_row_list = []
    for project in full_df['Project'].unique():
        proj_df = full_df.loc[df['Project']==project]
        for metric in ["Score", "Processing_Time", "Expense"]:
            sub_df_os = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                         (proj_df['Python'] == '3.7') &
                                         (proj_df['Hardware'] == 'amd64')) |
                                        (proj_df['OS'] == 'MacOS') |
                                        (proj_df['OS'] == 'Windows')]

            comp_df_OS = pd.DataFrame(sub_df_os.groupby('OS')[metric].mean())

            #calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['MacOS'][0]),
            #calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['Windows'][0])

            sub_df_dist = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                   (proj_df['Python'] == '3.7') &
                                   (proj_df['Hardware'] == 'amd64')) |
                                  (proj_df['OS'] == 'Linux-Bionic') |
                                  (proj_df['OS'] == 'Linux-Focal')]

            comp_df_dist = pd.DataFrame(sub_df_dist.groupby('OS')[metric].mean())


            # calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Bionic'][0]),
            # calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Focal'][0]),


            sub_df_hw = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                 (proj_df['Python'] == '3.7') &
                                 (proj_df['Hardware'] == 'amd64')) |
                                (proj_df['Hardware'] == 'arm64')]

            comp_df_hw = pd.DataFrame(sub_df_hw.groupby('Hardware')[metric].mean())


            # calculate_pct_change(comp_df_hw.loc['amd64'][0], comp_df_hw.loc['arm64'][0])

            sub_df_py = proj_df.loc[((proj_df['OS'] == 'Linux-Xenial') &
                                 (proj_df['Python'] == '3.7') &
                                 (proj_df['Hardware'] == 'amd64')) |
                                (proj_df['Python'] == '3.8') |
                                (proj_df['Python'] == '3.6')]

            comp_df_py = pd.DataFrame(sub_df_py.groupby('Python')[metric].mean())


            # calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.6'][0]),
            # calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.8'][0])

            df_row_list.append([project, metric,
                                calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['MacOS'][0]),
                                calculate_pct_change(comp_df_OS.loc['Linux-Xenial'][0], comp_df_OS.loc['Windows'][0]),
                                calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Bionic'][0]),
                                calculate_pct_change(comp_df_dist.loc['Linux-Xenial'][0], comp_df_dist.loc['Linux-Focal'][0]),
                                calculate_pct_change(comp_df_hw.loc['amd64'][0], comp_df_hw.loc['arm64'][0]),
                                calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.6'][0]),
                                calculate_pct_change(comp_df_py.loc['3.7'][0], comp_df_py.loc['3.8'][0])
                                ])

    pd.DataFrame(df_row_list, columns=['Project', 'Metric', 'os_mac', 'os_windows', 'dist_bionic', 'dist_focal',
                                       'hw_arm64', 'py_3_6', 'py_3_8']).to_csv('analysis_report/analysis_report.csv')

if __name__ == '__main__':
    # following code combines all projects into one dataframe
    df_list = []
    data_folder_path = 'output'
    for file in list_files(data_folder_path, all=False, extension='csv'):
        tmp_df = pd.read_csv(f'{data_folder_path}/{file}')
        df_list.append(tmp_df)
    df = pd.concat(df_list, ignore_index=True)

    df['Python'] = df['Python'].astype(str)

    df = add_credits(df)

    df.to_csv('/Users/musfiqurrahman/Documents/full_df.csv')
    print("Generating report for OS:")

    print("\nwith respect to Score:")

    sig_combined = combine_sig_result(df, "OS", "Score")
    generate_summary_report(sig_combined)

    print("\nwith respect to Processing Time:")

    sig_combined = combine_sig_result(df, "OS", "Processing_Time")
    generate_summary_report(sig_combined)


    print("\nwith respect to Expense:")

    sig_combined = combine_sig_result(df, "OS", "Expense")
    generate_summary_report(sig_combined)

    print("\n\n")


    print("Generating report for Dist:")

    print("\nwith respect to Score:")

    sig_combined = combine_sig_result(df, "dist", "Score")
    generate_summary_report(sig_combined)

    print("\nwith respect to Processing Time:")

    sig_combined = combine_sig_result(df, "dist", "Processing_Time")
    generate_summary_report(sig_combined)


    print("\nwith respect to Expense:")

    sig_combined = combine_sig_result(df, "dist", "Expense")
    generate_summary_report(sig_combined)


    print("\n\n")

    print("Generating report for Python:")

    print("\nwith respect to Score:")

    sig_combined = combine_sig_result(df, "Python", "Score")
    generate_summary_report(sig_combined)

    print("\nwith respect to Processing Time:")

    sig_combined = combine_sig_result(df, "Python", "Processing_Time")
    generate_summary_report(sig_combined)


    print("\nwith respect to Expense:")

    sig_combined = combine_sig_result(df, "Python", "Expense")
    generate_summary_report(sig_combined)


    print("\n\n")

    print("Generating report for CPU:")

    print("\nwith respect to Score:")

    sig_combined = combine_sig_result(df, "Hardware", "Score")
    generate_summary_report(sig_combined)

    print("\nwith respect to Processing Time:")

    sig_combined = combine_sig_result(df, "Hardware", "Processing_Time")
    generate_summary_report(sig_combined)


    print("\nwith respect to Expense:")

    sig_combined = combine_sig_result(df, "Hardware", "Expense")
    generate_summary_report(sig_combined)


    generate_report_df(df)
