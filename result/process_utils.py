import pandas as pd
import numpy as np
import os

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
        # unpack task column
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


def aggregate_al_strategies(folder, strategy_fileter, budget, total, acc, group=['cycle', 'task']):   
    files = os.listdir(folder)
    files = filter_methods(files, strategy_fileter)

    strategy_scores = {}
    for file in files:
        if file.split('_')[-4] == str(budget) and file.split('_')[-2] == str(total):
            score_fn = os.path.join(folder, file)
            score_df = pd.read_csv(score_fn)
            if group != []:
                score_df = score_df.groupby(group).mean()
            strategy_scores[file] = (score_df[acc])
            
    # merge all dataframes
    merged_df = pd.concat(strategy_scores.values(), axis=1)
    merged_df.columns = strategy_scores.keys()
    merged_df = merged_df.rename(columns={col: col.split('_')[1] +'-'+ col.split('_')[2] for col in merged_df.columns})
    merged_df = merged_df.reset_index()

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
        x_ticklabels = [(t+1, a+1) for a, t in zip(tasks, cycles)]

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
