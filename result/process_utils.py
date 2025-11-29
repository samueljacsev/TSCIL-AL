import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from pathlib import Path
import sys
import numpy as np

# If this notebook lives in `result/` (as yours does), parent() is repo root
repo_root = Path.cwd().resolve().parent  # -> /home/sam/Downloads/gh/TSCIL-AL
sys.path.insert(0, str(repo_root))

from utils.metrics import compute_performance



def get_filtered_files(data_filter, cycles_filter):
    cycles_filter = [str(c) for c in cycles_filter]
    filtered_files = []
    for foldername in data_filter:
        foldername = foldername.lower()
        files = os.listdir(foldername)
        for file in files:
            if file.split('_')[-4] in cycles_filter:
                filtered_files.append(os.path.join(foldername, file))
    return filtered_files


def generate_score_files(filtered_files):
    for path in filtered_files:
        df = pd.read_csv(path)
        df = df.groupby(['task', 'cycle',]).mean() # mean over all (default=5) runs
        df.drop(columns=['run'], inplace=True)
        df['task'] = df.index.get_level_values(0)
        df['cycle'] = df.index.get_level_values(1)
        df = df.reset_index(drop=True)
        task_cols = [c for c in df.columns if c.startswith('task_')]

        # round all task columns to 2 decimal places
        df[task_cols] = df[task_cols].round(2)

        # calculate the average accuracy for each cycle (over all tasks at the current cycle)
        df['average_accuracy'] = df.apply(
        lambda row: np.mean([v for v in row[task_cols] if v > 1e-5]), axis=1).round(2)

        df['learning_accuracy'] = df.apply(
        lambda row: round(row[f'task_{int(row["task"]) + 1}'], 2), axis=1).round(2)

        # Reorder columns to match original structure: task, cycle, then task_cols, then metrics
        ordered_cols = ['task', 'cycle'] + task_cols + ['average_accuracy', 'learning_accuracy']
        df = df[ordered_cols]

        dir = os.path.dirname(path)
        file = path[len(dir)+1:]

        df.to_csv(os.path.join(dir+'_scores', 'score_'+ file), index=False)
        

def filter_methods(names, filters = []):
    filtered_names = []
    
    for name in names:
        for f in filters:
            if f.lower() in name.lower():
                filtered_names.append(name)
                break
    return filtered_names


def aggregate_al_strategies(folder, strategy_fileter, budget, total, acc):  
    files = os.listdir(folder)
    files = filter_methods(files, strategy_fileter)

    meta_df = None

    strategy_scores = {}
    for file in files:
        if file.split('_')[-4] == str(budget) and file.split('_')[-2] == str(total):
            score_fn = os.path.join(folder, file)
            score_df = pd.read_csv(score_fn)
            strategy_scores[file] = (score_df[acc])
            if meta_df is None:
                meta_df = score_df[['task', 'cycle']]
            
    # merge all dataframes
    merged_df = pd.concat(strategy_scores.values(), axis=1)
    merged_df.columns = strategy_scores.keys()
    merged_df = merged_df.rename(columns={col: col.split('_')[1] +'-'+ col.split('_')[2] for col in merged_df.columns})
    merged_df = merged_df.reset_index()
    merged_df = pd.concat([meta_df, merged_df], axis=1)

    return merged_df

def add_aser_er_means(df):
    aser_except_random = [col for col in df.columns if "random" not in col.lower() and "-aser" in col.lower()]
    er_except_random = [col for col in df.columns if "random" not in col.lower() and "-er" in col.lower()]
    df['ASER-mean'] = df[aser_except_random].mean(axis=1)
    df['ER-mean'] = df[er_except_random].mean(axis=1)


def task_level_filter(df):
    n_cycles = df['cycle'].nunique()

    return df[df['cycle']==n_cycles-1].reset_index(drop=True)


# Highlight the highest value per row (excluding non-numeric columns)
def highlight_max(row):
    # Identify columns to exclude (typically 'cycle', 'task', 'index')
    exclude_cols = ['cycle', 'task', 'index']
    
    # Create styles array
    styles = [''] * len(row)
    
    # Get numeric values, excluding certain columns
    numeric_mask = row.index.isin([col for col in row.index if col not in exclude_cols])
    numeric_values = row[numeric_mask]
    
    # Only process if we have numeric values
    if len(numeric_values) > 0 and numeric_values.dtype in [np.float64, np.int64, np.float32, np.int32]:
        # Find max value
        max_val = numeric_values.max()
        
        # Highlight all columns that have the max value
        for col in numeric_values.index:
            col_value = numeric_values[col]
            if pd.isna(col_value) or pd.isna(max_val):
                continue
            if abs(col_value - max_val) < 1e-10:  # Use small tolerance for float comparison
                max_pos = row.index.get_loc(col)
                styles[max_pos] = 'background-color: grey; font-weight: bold'
    
    return styles


def plot_progress(df, loc='best', figsize=(13, 7), title=None, X_label=None, Y_label=None, task=None):
    import matplotlib.pyplot as plt
    
    strategy_cols = [col for col in df.columns if col not in ['cycle', 'index', 'task']]
    print(strategy_cols)
    if task is not None:
        df = df[df['task'] == task].reset_index(drop=True)

    aser_cols = sorted([col for col in strategy_cols if col.endswith('-ASER')])
    er_cols = sorted([col for col in strategy_cols if col.endswith('-ER')])

    assert len(aser_cols) == len(er_cols), "Mismatch in number of ASER and ER strategies"

    x_ticklabels = [t+1 for t in df['task'].values.tolist()]

    if df['cycle'].nunique() > 1:
        cycles = df['cycle'].values.tolist()
        tasks = df['task'].values.tolist()
        x_ticklabels = [(t+1, a+1) for t, a in zip(tasks, cycles)]

    # Create color mapping based on base name (without -ASER or -ER suffix)
    colors = plt.cm.tab10.colors
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each pair with same color, ASER solid, ER dotted
    color_idx = 0
    for i, (aser_col, er_col) in enumerate(zip(aser_cols, er_cols)):
        # Use black for random strategies
        if 'random' in aser_col.lower():
            color = 'black'
        else:
            color = colors[color_idx % len(colors)]
            color_idx += 1
        ax.plot(df.index, df[aser_col], marker='o', linestyle='-', color=color, label=aser_col)
        ax.plot(df.index, df[er_col], marker='o', linestyle=':', color=color, label=er_col)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x_ticklabels)
    ax.set_xlabel(X_label if X_label is not None else 'Index')
    ax.set_ylabel(Y_label if Y_label is not None else 'Accuracy')
    ax.set_title(title if title is not None else f'Accuracy per Cycle {"in task " + str(task) if task is not None else "for all tasks"}')
    ax.legend(loc=loc)
    plt.tight_layout()
    plt.show()



    # -------------- FINAL EVALUATION METHODS --------------- #

def get_multiple_al_methods_summary(filtered_files):
    keys = ['_'.join(f.split('/')[1].split('_')[:2]) for f in filtered_files]
    acc_multiple_dataset_multiple_run = {}

    for k, f in zip(keys, filtered_files):
        #print(f"Processing file: {f} with key: {k}")
        acc_df = pd.read_csv(f)

        Acc_multiple_run_test = []
        task_cols = [col for col in acc_df.columns if 'task_' in col]
        df_task_level = task_level_filter(acc_df)
        n_runs = df_task_level['run'].nunique()
        for run in range(n_runs):
            Acc_tasks = {'test':  []}
            for row in df_task_level[df_task_level['run']==run].iterrows():
                Acc_tasks['test'].append(row[1][task_cols].values)
            Acc_multiple_run_test.append(Acc_tasks['test'])
        Acc_multiple_run_test = np.array(Acc_multiple_run_test)
        acc_multiple_dataset_multiple_run[k] = Acc_multiple_run_test
    return acc_multiple_dataset_multiple_run


def ger_multiple_methods_summary_df(acc_multiple_dataset_multiple_run):
    columns = ['Method', 'Avg End Acc', 'Avg End Acc error', 'Avg End Fgt',
           'Avg End Fgt error', 'Avg Cur Acc', 'Avg Cur Acc error',
           'Avg Acc', 'Avg Acc error']
    rows = []
    for k, data in acc_multiple_dataset_multiple_run.items():
        #print(f"Computing performance for key: {k}")
        avg_end_acc, avg_end_fgt, avg_cur_acc, avg_acc, _ = compute_performance(data)

        # Ensure we store floats (numpy scalars are okay too)
        rows.append({
            'Method': k,
            'Avg End Acc': np.around(avg_end_acc[0], decimals=2),
            'Avg End Acc error': np.around(avg_end_acc[1], decimals=2),
            'Avg End Fgt': np.around(avg_end_fgt[0], decimals=2),
            'Avg End Fgt error': np.around(avg_end_fgt[1], decimals=2),
            'Avg Cur Acc': np.around(avg_cur_acc[0], decimals=2),
            'Avg Cur Acc error': np.around(avg_cur_acc[1], decimals=2),
            'Avg Acc': np.around(avg_acc[0], decimals=2),
            'Avg Acc error': np.around(avg_acc[1], decimals=2)
        })

    rows = sorted(rows, key=lambda x: x['Method'], reverse=True)
    results_df = pd.DataFrame.from_records(rows, columns=columns)

    # Round numeric columns once (keeps AL method as-is)
    num_cols = results_df.select_dtypes(include='number').columns
    results_df[num_cols] = results_df[num_cols].round(2)

    return results_df


def plot_score_values(results_df, DATASET, loc='best'):
    # Prepare data for plotting
    metrics = ['Avg End Acc', 'Avg End Fgt', 'Avg Cur Acc']
    error_metrics = ['Avg End Acc error', 'Avg End Fgt error', 'Avg Cur Acc error']
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Modern blue, coral red, emerald green


    # Set up the plot
    fig, ax = plt.subplots(figsize=(11, 7))

    # Bar width and positions
    n_methods = len(results_df)
    n_metrics = len(metrics)
    width = 0.25
    x = np.arange(n_methods)

    # Plot bars for each metric
    for idx, (metric, error_metric, color) in enumerate(zip(metrics, error_metrics, colors)):
        values = results_df[metric].values
        raw_errors = results_df[error_metric].values
        # Clip errors so they don't make bars go below 0 or above 100
        errors = np.minimum(raw_errors, np.minimum(values, 100 - values))
        
        offset = (idx - n_metrics/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, yerr=errors, capsize=4, 
                    alpha=0.85, color=color, label=metric, edgecolor='white', linewidth=1.2)

    # Customize plot
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Metric Value (%)', fontsize=12)
    ax.set_title(f'{DATASET} - Learning Methods Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Method'].str.replace('_', ' '), rotation=30, ha='right', fontsize=10)
    ax.legend(loc=loc, fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.6, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()