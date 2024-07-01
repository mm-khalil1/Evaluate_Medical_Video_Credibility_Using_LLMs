import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro, kstest, kendalltau, spearmanr
from irrCAC.raw import CAC
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from pingouin import intraclass_corr
from typing import List, Optional, Dict

import warnings

# Suppress warnings for scipy functions: 'UserWarning: Sample size too small for normal approximation.'
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

QUESTIONS_COLUMNS = [f'Q{i}' for i in range(1,16)]         # [Q1, Q2, ... Q15]
EXPERT_TOTAL_COLUMNS = ['Expert1', 'Expert2', 'Experts_Avg']    
EXPERT1_COLUMNS = [f'Expert1 Q{i}' for i in range(1, 16)]       # [Expert1 Q1, Expert1 Q2, ... Expert1 Q15]
EXPERT2_COLUMNS = [f'Expert2 Q{i}' for i in range(1, 16)]       # [Expert2 Q1, Expert2 Q2, ... Expert2 Q15]
EXPERTS_AVG_COLUMNS = [f'Experts_Avg Q{i}' for i in range(1, 16)]   # [Experts_Avg Q1, Experts_Avg Q2, ... Experts_Avg Q15]

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

def filter_df_by_topics(df: pd.DataFrame, topics_dict: dict, topic_keys: list, 
                        return_cols: list = None) -> pd.DataFrame:
    '''
    Filter DataFrame rows based on specific topics.

    Parameters:
        df (DataFrame): DataFrame to be filtered.
        topics_dict (dict): Dictionary mapping topic keys to topic values.
        topic_keys (list): List of topic keys to filter by.
        return_cols (list, optional): List of columns to return in the filtered DataFrame. Defaults to None.

    Returns:
        DataFrame: Filtered DataFrame containing specified columns.
    '''
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a DataFrame.")
    if not isinstance(topics_dict, dict):
        raise ValueError("Input 'topics_dict' must be a dictionary.")
    if not all(key in topics_dict for key in topic_keys):
        raise ValueError('Some topic keys are not found in topics_dict.')

    # Construct list of topics from topic keys
    topic_list = [topics_dict[key] for key in topic_keys if key in topics_dict]
    
    # Filter DataFrame by topics and return specified columns
    filtered_df = df[df['Topic'].isin(topic_list)].reset_index(drop=True)

    if return_cols is not None:
        return filtered_df[return_cols]
    else:
        return filtered_df

def calculate_experts_avg_of_questions(df: pd.DataFrame, expert1_columns, expert2_columns, 
                                       column_loc=None) -> pd.DataFrame:
    '''
    Calculate question-wise expert average for each pair of expert columns and insert them into the DataFrame.

    Parameters:
        df (DataFrame): DataFrame containing the group data.
        expert1_columns (list): List of column names corresponding to expert 1.
        expert2_columns (list): List of column names corresponding to expert 2.
        column_loc (int, optional): Location where the new columns should be inserted. Defaults to len(df.columns).

    Returns:
        DataFrame: DataFrame with question-wise expert averages inserted.
    '''
    if len(expert1_columns) != len(expert2_columns):
        raise ValueError('Number of columns is not equal in expert1_columns and expert2_columns.')

    if column_loc is None:
        column_loc = len(df.columns)

    for i, (q_expert1, q_expert2) in enumerate(zip(expert1_columns, expert2_columns), start=1):
        # Calculate the mean for the pair of columns
        questionwise_expert_avg = np.where(
            df[q_expert2].notna(),
            df[[q_expert1, q_expert2]].mean(axis=1),
            df[q_expert1]
        )

        col = f'Experts_Avg Q{i}'
        # Insert the calculated means into the DataFrame
        if col not in df.columns:
            df.insert(loc=column_loc + i - 1,
                    column=col,
                    value=questionwise_expert_avg)
        else:
            df[col] = questionwise_expert_avg  # Update existing column if it already exists
    
    return df

def binarize_value(value: float, limit: float = 4) -> int:
    '''Converts a value to 0 if it's less than `limit`, otherwise to 1.'''
    return 0 if value < limit else 1

def merge_dataframes(main_df: pd.DataFrame, scores_df: pd.DataFrame, selected_columns: List[str], 
                     how: str = 'inner', on: str = 'Video ID') -> pd.DataFrame:
    '''
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
        ValueError: If any of the selected columns are not found in scores_df.
    '''
    # Check if all selected_columns are present in scores_df
    if not all(col in scores_df.columns for col in selected_columns):
        missing_cols = [col for col in selected_columns if col not in scores_df.columns]
        raise ValueError(f'Columns {missing_cols} not found in scores_df.')

    try:
        return pd.merge(main_df, scores_df[selected_columns], how=how, on=on)
    except Exception as e:
        raise e

def set_plot_properties(plot_obj: plt.Axes, **kwargs) -> None:
    '''
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
    set_plot_properties(ax, xlim=(0, 1), ylim=(0, 10), xlabel='X Axis', ylabel='Y Axis', 
                        title='Plot Title', xticks_rotation=45)
    ```
    '''
    plot_obj.set_xlim(kwargs.get('xlim', plot_obj.get_xlim()))
    plot_obj.set_ylim(kwargs.get('ylim', plot_obj.get_ylim()))
    plot_obj.set_xlabel(kwargs.get('xlabel', plot_obj.get_xlabel()))
    plot_obj.set_ylabel(kwargs.get('ylabel', plot_obj.get_ylabel()))
    plot_obj.set_title(kwargs.get('title', plot_obj.get_title()))
    plt.xticks(rotation=kwargs.get('xticks_rotation', 0))

def create_plot(plot_type: str, data: pd.DataFrame, x=None, ax=None, color=None, **kwargs) -> tuple:
    '''
    Create a seaborn plot of specified type with given data and properties.

    Parameters:
        plot_type (str): Type of plot to create. Supported types: 'countplot', 'histplot', 'boxplot', 'bar', 'barh', 'heatmap'.
        data (pd.DataFrame): DataFrame containing the data to plot.
        x (str, optional): Variable to plot on the x-axis.
        ax (plt.Axes, optional): Axes object to draw the plot onto.
        color (str, optional): Color of the plot elements.
        **kwargs: Additional keyword arguments to customize the plot.

    Some supported keyword arguments:
        - order (list): Order of categories for 'countplot'.
        - bins (int): Number of bins for 'histplot'.
        - width (float): Width of the boxes for 'boxplot'.
        - values (array-like): Values for 'bar' plot.
        - columns (array-like): Columns for 'bar' plot.
        - figsize (tuple): Figure size (width, height) in inches.

    Returns:
        fig (plt.Figure): The figure object associated with the plot.
        ax (plt.Axes): The axes object associated with the plot.

    Example:
    ```
    create_plot('countplot', data=df, x='category', order=['A', 'B', 'C'], color='blue', figsize=(8, 6))
    ```
    '''
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
        values = data.values.flatten()
        if len(data.columns) > 1:
            columns = data.columns
        elif len(data.index) > 1:
            columns = data.index
        else:
            raise Warning("data shape is 1x1")
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = plt.cm.coolwarm

        if plot_type == 'bar':
            kwargs.update({'ylim': (min(0, min(values)-0.05), max(1, max(values)+0.05))})
            bars = ax.bar(columns, values, color=cmap(norm(values)))
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height, '%.2f' % height, ha='center', va='bottom')
                
        elif plot_type == 'barh':
            values = values[::-1]
            columns = columns[::-1]
            kwargs.update({'xlim': (min(0, min(values)-0.05), max(1, max(values)+0.05))})
            bars = ax.barh(columns, values, color=cmap(norm(values)))
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2, '%.2f' % width, ha='left', va='center')

    elif plot_type == 'heatmap':
        sns.heatmap(data, annot=True, fmt='.2f', cmap='coolwarm', 
                    vmin=0, vmax=1, linewidths=.5, linecolor='black', ax=ax)

    set_plot_properties(ax, **kwargs)
    plt.tight_layout()

    return fig, ax

def test_normality(data) -> None:
    '''
    Perform Shapiro-Wilk and Kolmogorov-Smirnov normality tests on the input data and print the p-value.

    Parameters:
        data (array-like): The data to be tested for normality.
    '''
    # Input validation
    if not isinstance(data, (list, tuple, np.ndarray)):
        raise ValueError('Input data must be an array-like object (list, tuple, or numpy array).')
    
    # Shapiro-Wilk Test
    _, p_value_sw = shapiro(data)
    
    # Kolmogorov-Smirnov Test
    _, p_value_ks = kstest(data, 'norm')
    
    alpha = 0.05
    
    # Shapiro-Wilk Test
    if p_value_sw > alpha:
        print(p_value_sw, 'Shapiro-Wilk Test: Data looks normally distributed (fail to reject H0)')
    else:
        print(p_value_sw, 'Shapiro-Wilk Test: Data does not look normally distributed (reject H0)')
    
    # Kolmogorov-Smirnov Test
    if p_value_ks > alpha:
        print(p_value_ks, 'Kolmogorov-Smirnov Test: Data looks normally distributed (fail to reject H0)')
    else:
        print(p_value_ks, 'Kolmogorov-Smirnov Test: Data does not look normally distributed (reject H0)')

def weights_matrix(labels, weights_type='ordinal') -> np.ndarray:
    """
    Generates a weights matrix for ordinal data.

    Args:
        labels: List of ordinal category labels.
        weights_type: Type of weights to use ('linear', 'quadratic', or 'ordinal').

    Returns:
        A weights matrix of size (Q x Q), where Q is the number of ordinal categories.
    """
    Q = len(labels)
    if Q == 0:
        raise ValueError("Length of labels is 0!")
    max_label = max(labels)
    min_label = min(labels)

    weights = np.zeros((Q, Q))

    for i in range(Q):
        for j in range(Q):
            if weights_type == 'linear':
                weights[i, j] = 1 - abs(labels[i] - labels[j]) / (max_label - min_label)
            elif weights_type == 'quadratic':
                weights[i, j] = 1 - (abs(labels[i] - labels[j]) / (max_label - min_label)) ** 2
            elif weights_type == 'ordinal':
                nij = max(labels[i], labels[j]) - min(labels[i], labels[j]) + 1
                weights[i][j] = nij * (nij - 1) / 2
            else:
                raise ValueError("Invalid weights_type. Choose 'linear', 'quadratic' or 'ordinal'.")
    if weights_type == 'ordinal':
        weights = 1 - weights / np.max(weights)
    return weights

def bp_kappa(rater1, rater2, labels, weights_type='ordinal'):
    """
    Calculates Brennan-Prideger Kappa between two raters for ordinal data.

    Args:
        rater1: A list of ratings (indices corresponding to labels) from the first rater.
        rater2: A list of ratings (indices corresponding to labels) from the second rater.
        labels: List of ordinal category labels.
        weights_type: A string specifying the weighting scheme ('linear', 'quadratic', or 'ordinal').

    Returns:
        The observed agreement between the two raters.
    """
    if len(rater1) != len(rater2):
        raise ValueError('Rater lists must be of equal length')

    cm = confusion_matrix(rater1, rater2, labels=labels)
    w = weights_matrix(labels, weights_type)
    N = len(rater1)  # Total number of observations
    Q = len(labels)  # Number of ordinal categories

    # pa = 0
    # for i in range(N):
    #     pa += 1 - (abs(rater1[i] - rater2[i]) / (Q - 1)) ** 2
    # pa /= (N)

    pa = np.sum(w * cm) / N
    pe = w.sum() / Q ** 2
    bp = (pa - pe) / (1 - pe)
    return bp

def calculate_statistics(df: pd.DataFrame, col1: str, col2: str, 
                         categories: List[int], weights_type: str = 'quadratic') -> Dict[str, float]:
    '''
    Calculate various statistics between two columns in a DataFrame, including Brennan-Prediger Kappa, 
    Weighted Kappa, Intra-class Correlation Coefficient, and Gwet's AC2.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    col1 (str): Name of the first column.
    col2 (str): Name of the second column.
    categories (List[int]): Unique categories in the data.
    weights_type (str): Type of weights to use ('linear', 'quadratic', 'ordinal', etc.).

    Returns:
        Dict[str, float]: A dictionary containing the calculated statistics.
    
    Raises:
        ValueError: If the specified columns do not exist in the DataFrame.
    '''
    # Input validation
    if col1 not in df.columns:
        raise ValueError(f"'{col1}' does not exist in the DataFrame.")
    if col2 not in df.columns:
        raise ValueError(f"'{col2}' does not exist in the DataFrame.")

    # Drop rows with NaN values in either col1 or col2
    df = df[['Video ID', col1, col2]]
    df = df.dropna(subset=['Video ID', col1, col2])

    _, p_value_wilcoxon = wilcoxon(df[col1], df[col2])
    
    _, p_value_ttest = ttest_rel(df[col1], df[col2])
    
    # Calculate Intraclass Correlation Coefficient
    df_long = df.melt(id_vars=['Video ID'], var_name='Rater', value_name='Score')
    icc_results = intraclass_corr(data=df_long, targets='Video ID', raters='Rater', ratings='Score')
    icc3 = icc_results.loc[icc_results['Type'] == 'ICC3', 'ICC'].values[0]

    abs_mean_diff = np.mean(np.abs(df[col1] - df[col2]))
    
    bp = bp_kappa(df[col1].round(0), df[col2].round(0), labels=categories, weights_type=weights_type)

    # Calculate Gwet's AC2
    cac = CAC(df[[col1, col2]], weights=weights_type, categories=categories)
    gwet_ac = cac.gwet().get('est').get('coefficient_value')

    spearman_rho, _ = spearmanr(df[col1], df[col2])
    
    kendall, _ = kendalltau(df[col1], df[col2])

    if weights_type not in ['linear', 'quadratic']:
        weights_type = 'quadratic'
    w_kappa = cohen_kappa_score(df[col1].round(0), df[col2].round(0), labels=categories, weights=weights_type)

    # Store the statistics in a dictionary
    statistics = {
        "Weighted Kappa": w_kappa,
        "Kendall's Tau": kendall,
        "Spearman's Rho": spearman_rho,
        # "ICC3": icc3,
        "Brennan-Prediger Kappa": bp,
        "Gwet's AC2": gwet_ac,

        # "Wilcoxon signed-rank test p-value": p_value_wilcoxon,
        # "Paired t-test p-value": p_value_ttest,
        # "Absolute Mean Difference": abs_mean_diff,
    }
    return statistics

def calculate_stat_df(data: pd.DataFrame, 
                      rater1_columns: List[str], 
                      rater2_columns: List[str],
                      categories: List[int], 
                      agreement_coef: Optional[str] = None,
                      weights_type: str = 'quadratic',
                      output_df_column_names: Optional[List[str]] = None) -> pd.DataFrame:
    '''
    Calculate statistics for pairs of rater columns and return them in a DataFrame.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be analyzed.
        rater1_columns (List[str]): A list of column names for the first set of raters.
        rater2_columns (List[str]): A list of column names for the second set of raters.
        categories (List[int]): List of category labels.
        agreement_coef (Optional[str]): The specific agreement coefficient to extract from the statistics (e.g., 'Weighted Kappa').
        weights_type (str): Type of weights to use ('linear', 'quadratic', 'ordinal', etc.).
        output_df_column_names (Optional[List[str]]): A list of column names for the output statistics DataFrame.
    
    Returns:
        pd.DataFrame: A DataFrame containing the calculated statistics. 
        If `agreement_coef` is provided, only the specified coefficient's values are returned.
    '''
    if output_df_column_names is None:
        output_df_column_names = rater2_columns

    stats = {
        col_name: calculate_statistics(data, r1, r2, categories, weights_type)
        for col_name, r1, r2 in zip(output_df_column_names, rater1_columns, rater2_columns)
    }
    stats_df = pd.DataFrame(stats)
    if agreement_coef:
        stats_df = stats_df.loc[[agreement_coef]]
    return stats_df