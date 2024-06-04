import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro, kstest
from sklearn.metrics import cohen_kappa_score
from pingouin import intraclass_corr
from typing import List, Tuple

import warnings

# Suppress warnings for scipy functions
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

MODEL_15_SCORE_COLUMNS = [f"Q{i}" for i in range(1,16)]         # [Q1, Q2, ... Q15]
EXPERT_TOTAL_COLUMNS = ['Expert1', 'Expert2', 'Experts_Avg']    
EXPERT1_COLUMNS = [f"Expert1 Q{i}" for i in range(1, 16)]       # [Expert1 Q1, Expert1 Q2, ... Expert1 Q15]
EXPERT2_COLUMNS = [f"Expert2 Q{i}" for i in range(1, 16)]       # [Expert2 Q1, Expert2 Q2, ... Expert2 Q15]

TOPICS = {
    'NE': 'Nocturnal Enuresis',
    'DE': 'Delayed Ejaculation',
    'SB': 'Spina Bifida',
    'FF': 'Flat Feet',
    'CH': 'Cluster Headache',
    'TF': 'Trigger Finger',
    'PN': 'Pudendal Nerve',
    'ISA': 'Insulin Self-Administration'
}

def filter_df_by_topics(df: pd.DataFrame, topics_dict: dict, topic_keys: list, return_cols: list) -> pd.DataFrame:
    """
    Filter DataFrame rows based on specific topics.

    Parameters:
    df (DataFrame): DataFrame to be filtered.
    topics_dict (dict): Dictionary mapping topic keys to topic values.
    topic_keys (list): List of topic keys to filter by.
    return_cols (list): List of columns to return in the filtered DataFrame.

    Returns:
    DataFrame: Filtered DataFrame containing specified columns.
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a DataFrame.")
    if not isinstance(topics_dict, dict):
        raise ValueError("Input 'topics_dict' must be a dictionary.")
    if not all(key in topics_dict for key in topic_keys):
        raise ValueError("Some topic keys are not found in topics_dict.")

    # Construct list of topics from topic keys
    topic_list = [topics_dict[key] for key in topic_keys if key in topics_dict]
    
    # Filter DataFrame by topics and return specified columns
    filtered_df = df[df['Topic'].isin(topic_list)].reset_index(drop=True)
    return filtered_df[return_cols]

def calculate_experts_avg_of_questions(df, expert1_columns, expert2_columns, column_loc=None) -> pd.DataFrame:
    """
    Calculate question-wise expert average for each pair of expert columns and insert them into the DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing the group data.
    expert1_columns (list): List of column names corresponding to expert 1.
    expert2_columns (list): List of column names corresponding to expert 2.
    column_loc (int, optional): Location where the new columns should be inserted. Defaults to len(df.columns).

    Returns:
    DataFrame: DataFrame with question-wise expert averages inserted.
    """
    if column_loc is None:
        column_loc = len(df.columns)

    for i, (q_expert1, q_expert2) in enumerate(zip(expert1_columns, expert2_columns), start=1):
        # Calculate the mean for the pair of columns
        questionwise_expert_avg = np.where(
            df[q_expert2].notna(),
            df[[q_expert1, q_expert2]].mean(axis=1),
            df[q_expert1]
        )
        # Insert the calculated means into the DataFrame
        df.insert(loc=column_loc,
                  column=f"Experts_Avg Q{i}",
                  value=questionwise_expert_avg)
    
    return df

def sum_columns(df: pd.DataFrame, score_columns_list: List[str]) -> pd.DataFrame:
    """
    Sum the values of columns starting with "Q" and store the result in a new column named "Model Total".

    Parameters:
    df (DataFrame): DataFrame containing the score columns.
    score_columns_list (List[str]): List of column names to sum.

    Returns:
    DataFrame: DataFrame with the sum of values stored in a new column named "Model Total".
    """
    # Calculate the sum of values in the specified score_columns
    total_scores = df[score_columns_list].sum(axis=1)
    
    # Insert the 'Model Total' column as the last column in the DataFrame
    df.insert(loc=len(df.columns), column='Model Total', value=total_scores)
    
    return df

def binarize_value(value: float, limit: float = 4) -> int:
    """Converts a value to 0 if it's less than `limit`, otherwise to 1."""
    return 0 if value < limit else 1

def merge_dataframes(main_df: pd.DataFrame, scores_df: pd.DataFrame, 
                     selected_columns: List[str], 
                     how: str = 'inner', on: str = 'Video ID'
                     ) -> pd.DataFrame:
    """
    Merge selected columns from scores_df into main_df based on specified parameters.

    Parameters:
    main_df (DataFrame): The main DataFrame into which the selected columns from scores_df will be merged.
    scores_df (DataFrame): The DataFrame containing the scores to be merged.
    selected_columns (List[str]): A list of column names from scores_df to be merged into main_df.
    how (str, optional): The type of merge to be performed ('inner', 'outer', 'left', or 'right'). Defaults to 'inner'.
    on (str, optional): The column name to join on. Defaults to 'Video ID'.

    Returns:
    DataFrame: The merged DataFrame containing the selected columns from scores_df merged into main_df.

    Raises:
    Exception: If an error occurs during the merging process.
    """
    try:
        return pd.merge(main_df, scores_df[selected_columns], how=how, on=on)
    except Exception as e:
        raise e

def set_plot_properties(plot_obj: plt.Axes, **kwargs) -> None:
    """
    Set various properties of a plot object based on keyword arguments.

    Parameters:
    plot_obj (plt.Axes): The plot object whose properties will be set.
    **kwargs: Additional keyword arguments to set specific properties of the plot object.
        Supported keyword arguments:
        - xlim (tuple): Tuple specifying the x-axis limits (left, right).
        - ylim (tuple): Tuple specifying the y-axis limits (bottom, top).
        - xlabel (str): Label for the x-axis.
        - ylabel (str): Label for the y-axis.
        - title (str): Title of the plot.
        - xticks_rotation (int): Rotation angle (in degrees) for x-axis tick labels.

    Returns:
    None

    Example:
    ```
    set_plot_properties(ax, xlim=(0, 1), ylim=(0, 10), xlabel='X Axis', ylabel='Y Axis', title='Plot Title', xticks_rotation=45)
    ```
    """
    plot_obj.set_xlim(kwargs.get('xlim', plot_obj.get_xlim()))
    plot_obj.set_ylim(kwargs.get('ylim', plot_obj.get_ylim()))
    plot_obj.set_xlabel(kwargs.get('xlabel', plot_obj.get_xlabel()))
    plot_obj.set_ylabel(kwargs.get('ylabel', plot_obj.get_ylabel()))
    plot_obj.set_title(kwargs.get('title', plot_obj.get_title()))
    plt.xticks(rotation=kwargs.get('xticks_rotation', 0))

def create_plot(plot_type: str, data: pd.DataFrame, x=None, ax=None, color=None, **kwargs) -> None:
    """
    Create a seaborn plot of specified type with given data and properties.

    Parameters:
    plot_type (str): Type of plot to create. Supported types: 'countplot', 'histplot', 'boxplot', 'bar', 'barh', 'heatmap'.
    data (pd.DataFrame): DataFrame containing the data to plot.
    x (str, optional): Variable to plot on the x-axis.
    ax (plt.Axes, optional): Axes object to draw the plot onto.
    color (str, optional): Color of the plot elements.
    **kwargs: Additional keyword arguments to customize the plot.

    Supported keyword arguments:
    - order (list): Order of categories for 'countplot'.
    - bins (int): Number of bins for 'histplot'.
    - width (float): Width of the boxes for 'boxplot'.
    - values (array-like): Values for 'bar' plot.
    - columns (array-like): Columns for 'bar' plot.
    - figsize (tuple): Figure size (width, height) in inches.

    Returns:
    None

    Example:
    ```
    create_plot('countplot', data=df, x='category', order=['A', 'B', 'C'], color='blue', figsize=(8, 6))
    ```
    """
    if plot_type not in ['countplot', 'histplot', 'boxplot', 'bar', 'barh', 'heatmap']:
        raise ValueError("Invalid plot_type. Supported types: 'countplot', 'histplot', 'boxplot', 'bar', 'barh', 'heatmap'.")

    figsize = kwargs.pop('figsize', None)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
        if figsize is not None:
            fig.set_size_inches(figsize)

    if plot_type == 'countplot':
        order = kwargs.pop('order', None)
        sns.countplot(data=data, ax=ax, x=x, color=color, order=order)

    elif plot_type == 'histplot':
        bins = kwargs.pop('bins', None)
        sns.histplot(data=data, ax=ax, x=x, color=color, bins=bins)

    elif plot_type == 'boxplot':
        width = kwargs.pop('width', 0.5)
        sns.boxplot(data=data, ax=ax, width=width)

    elif plot_type == 'bar' or plot_type == 'barh':
        values = kwargs.pop('values', None)
        columns = kwargs.pop('columns', None)
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.coolwarm

        if plot_type == 'bar':
            bars = ax.bar(columns, values, color=cmap(norm(values)))
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, '%.2f' % height, ha='center', va='bottom')
                
        elif plot_type == 'barh':
            bars = ax.barh(columns, values, color=cmap(norm(values)))
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, '%.2f' % width, ha='left', va='center')

    elif plot_type == 'heatmap':
        sns.heatmap(data, annot=True, fmt=".2f", cmap='coolwarm', vmin=0, vmax=1, linewidths=.5, linecolor='black', ax=ax)

    set_plot_properties(ax, **kwargs)
    plt.tight_layout()

def test_normality(data) -> None:
    """
    Perform Shapiro-Wilk and Kolmogorov-Smirnov normality tests on the input data and print the p-value.

    Parameters:
    data (array-like): The data to be tested for normality.
    """
    # Input validation
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise ValueError("Input data must be an array-like object (list, tuple, or numpy array).")
    
    # Shapiro-Wilk Test
    _, p_value_sw = shapiro(data)
    
    # Kolmogorov-Smirnov Test
    _, p_value_ks = kstest(data, 'norm')
    
    alpha = 0.05
    
    # Shapiro-Wilk Test
    if p_value_sw > alpha:
        print(p_value_sw, "Shapiro-Wilk Test: Data looks normally distributed (fail to reject H0)")
    else:
        print(p_value_sw, "Shapiro-Wilk Test: Data does not look normally distributed (reject H0)")
    
    # Kolmogorov-Smirnov Test
    if p_value_ks > alpha:
        print(p_value_ks, "Kolmogorov-Smirnov Test: Data looks normally distributed (fail to reject H0)")
    else:
        print(p_value_ks, "Kolmogorov-Smirnov Test: Data does not look normally distributed (reject H0)")

def concordance_corr_coef(x: pd.Series, y: pd.Series) -> float:
    """
    Calculate the concordance correlation coefficient between two variables.

    Parameters:
    x (pd.Series): First variable.
    y (pd.Series): Second variable.

    Returns:
    float: The concordance correlation coefficient.
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    
    ccc = (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)
    
    return ccc

def calculate_statistics(df: pd.DataFrame, col1: str, col2: str, all_ratings: List[int] = [1, 2, 3, 4, 5]) -> dict:
    """
    Calculate various statistics including paired t-test, Wilcoxon signed-rank test,
    Weighted Kappa, ICC3, and CCC between two columns in a DataFrame.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    col1 (str): Name of the first column.
    col2 (str): Name of the second column.

    Returns:
    dict: A dictionary containing the calculated statistics.
    """
    # Input validation
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"'{col1}' does not exist in the DataFrame.")
    if col2 not in df.columns:
        raise ValueError(f"'{col2}' does not exist in the DataFrame.")

    # Drop rows with NaN values in either col1 or col2
    df = df[['Video ID', col1, col2]]
    df = df.dropna(subset=['Video ID', col1, col2])

    # Perform Wilcoxon signed-rank test
    try:
        _, p_value_wilcoxon = wilcoxon(df[col1], df[col2])
    except Exception as e:
        raise RuntimeError(f"Error in performing Wilcoxon signed-rank test: {str(e)}")
    
    # Perform paired t-test
    try:
        _, p_value_ttest = ttest_rel(df[col1], df[col2])
    except Exception as e:
        raise RuntimeError(f"Error in performing paired t-test: {str(e)}")

    # Calculate Intraclass Correlation Coefficient
    try:
        df_long = df.melt(id_vars=['Video ID'], var_name='Rater', value_name='Score')
        icc_results = intraclass_corr(data=df_long, targets='Video ID', raters='Rater', ratings='Score')
        icc3 = icc_results.loc[icc_results['Type'] == 'ICC3', 'ICC'].values[0]
    except Exception as e:
        raise RuntimeError(f"Error in calculating Intraclass Correlation Coefficient: {str(e)}")

    # Calculate Concordance Correlation Coefficient
    try:
        ccc = concordance_corr_coef(df[col1], df[col2])
    except Exception as e:
        raise RuntimeError(f"Error in calculating Concordance Correlation Coefficient: {str(e)}")

    # Calculate Weighted Kappa
    kappa = cohen_kappa_score(df[col1].round(0), df[col2].round(0), 
                              weights='quadratic', labels=all_ratings)

    # Store the statistics in a dictionary
    statistics = {
        "Wilcoxon signed-rank test p-value": p_value_wilcoxon,
        "Paired t-test p-value": p_value_ttest,
        "ICC3": icc3,
        "CCC": ccc,
        "Weighted Kappa": kappa,
    }

    return statistics

def questionwise_agreement(df: pd.DataFrame,
                             rater1_columns: List[str],
                             rater2_columns: List[str],
                             rater1_name: str,
                             rater2_name: str,
                             topic_name: str,
                             metric: str = 'Weighted Kappa',
                             figsize: Tuple[int, int] = (6, 5)) -> None:
    """
    Compare two sets of columns question-wise using statistical measures and visualize the comparison using a bar plot.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        rater1_columns (List[str]): List of column names for the first rater.
        rater2_columns (List[str]): List of column names for the second rater.
        rater1_name (str): Name of the first rater.
        rater2_name (str): Name of the second rater.
        topic_name (str): Name of the topic or domain.
        figsize (Tuple[int, int], optional): Figure size for the plot. Default is (12, 5).
    """

    # Create an empty DataFrame to store statistics
    stat_df = pd.DataFrame(columns=MODEL_15_SCORE_COLUMNS)

    for i, (q_col1, q_col2) in enumerate(zip(rater1_columns, rater2_columns), start=1):
        statistics = calculate_statistics(df, q_col1, q_col2)
        stat_df[f"Q{i}"] = pd.Series(statistics)

    stat_df = stat_df.loc[[metric]]

    # Reverse values order
    values = stat_df.iloc[0].values[::-1]
    columns = stat_df.columns[::-1]
    
    create_plot('barh', data=stat_df, columns=columns, values=values,
                figsize=figsize, xlim=(min(0, min(values)), 1),
                xlabel=metric, ylabel='Questions', 
                title=f'{topic_name}: Agreement between {rater1_name} and {rater2_name}',
                xticks_rotation=0)
    