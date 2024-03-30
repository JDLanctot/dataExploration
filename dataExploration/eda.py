import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.weightstats import ztest
from typing import List, Optional
from dataExploration.visualization import plot_regression

__all__ = []
__all__.extend([
    'explore_data',
    'compare_means',
    'compare_means_anova',
    'check_normality',
    'calculate_correlation',
    'perform_regression_and_plot'
])

def explore_data(df: pd.DataFrame) -> None:
    # DROP THE ID COLUMN WHEN EXPLORING THE DATA, THIS WON'T EFFECT THE DF OUTSIDE OF THIS SCOPE
    df = df.drop(['id'], axis=1)

    # PRINT SAMPLE
    print('-'*90)
    print('Here is a sample of the data:')
    print(df.head())

    # PRINT DATASET INFO
    print('-'*90)
    print('This dataset has', df.shape[0], 'rows/observations, and ', df.shape[1], 'columns')
    print('The dataset has columns of types:')
    print(df.dtypes.value_counts())

    # PRINT VARIABLE INFO
    print('-'*90)
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category')

    cat_col = df.select_dtypes(include=['category'])
    for col in cat_col:
        print('Unique Values of {} are \n'.format(col),df[col].unique())
        print('-'*90)

    # PRINT DESCRIPTIVES
    print('Here are the descriptives of the dataset:')
    print(df.describe(include='all').T)

def compare_means(df: pd.DataFrame, colname: str, compare_col: str, compare_vals: List, compare_labels: List, alpha=0.05) -> None:
    # Validate inputs
    if len(compare_vals) < 2 or len(compare_labels) < 2:
        raise ValueError("compare_vals and compare_labels must both be at least of length 2.")

    # Extract the two groups based on compare_vals and focusing on colname for comparison
    group1 = df[df[compare_col] == compare_vals[0]][colname]
    group2 = df[df[compare_col] == compare_vals[1]][colname]

    # Determine the test based on the size of the groups
    total_samples = len(group1) + len(group2)
    if total_samples < 30:
        print('The sample size is less than 30, so we will be doing a T-Test')
        # Perform an independent t-test (Welch's t-test for unequal variances)
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)

        # Calculate degrees of freedom for determining critical t-values
        # Adjustment for Welch's t-test degrees of freedom calculation
        dfree = min(len(group1), len(group2)) - 1

        # Critical t-value for two-tailed test
        critical_t_two_tailed = stats.t.ppf(1 - alpha/2, dfree)

        # Critical t-value for one-tailed test
        critical_t_one_tailed = stats.t.ppf(1 - alpha, dfree)

        # Confidence intervals for the mean difference
        # Note: Calculation of confidence interval should consider the standard error of the difference between means
        ci_lower, ci_upper = stats.t.interval(alpha, dfree, loc=(group1.mean() - group2.mean()), scale=stats.sem(group1 - group2, nan_policy='omit'))

        # Print results with descriptive labels
        print(f"Comparing means of '{colname}' between {compare_labels[0]} and {compare_labels[1]}:")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_val:.3f}")
        print(f"Degrees of freedom: {dfree}")
        print(f"Critical t-value for two-tailed test: ±{critical_t_two_tailed:.3f}")
        print(f"Critical t-value for one-tailed test: {critical_t_one_tailed:.3f}")
        print(f"95% confidence interval for the difference in means of '{colname}': ({ci_lower:.3f}, {ci_upper:.3f})")

        # Conclusions with descriptive labels
        # Two-tailed test
        if abs(t_stat) > critical_t_two_tailed:
            print(f"Reject the null hypothesis for a two-tailed test: significant difference in means of '{colname}' between {compare_labels[0]} and {compare_labels[1]}.")
        else:
            print(f"Fail to reject the null hypothesis for a two-tailed test: no significant difference in means of '{colname}' between {compare_labels[0]} and {compare_labels[1]}.")

        # One-tailed test (assuming testing if mean of first group is greater than second)
        if t_stat > critical_t_one_tailed:
            print(f"Reject the null hypothesis for a one-tailed test: {compare_labels[0]} mean of '{colname}' is significantly greater than {compare_labels[1]} mean of '{colname}'.")
        else:
            print(f"Fail to reject the null hypothesis for a one-tailed test: {compare_labels[0]} mean of '{colname}' is not significantly greater than {compare_labels[1]} mean of '{colname}'.")
    else:
        print(f"Performing a z-test because the total sample size is 30 or more.")
        z_stat, p_val = ztest(group1, group2)
        dfree = min(len(group1), len(group2)) - 1
        # Z critical value for two-tailed test, 95% CI
        critical_z_two_tailed = stats.norm.ppf(1 - alpha/2)
        critical_z_one_tailed = stats.norm.ppf(1 - alpha)
        # Calculate the standard error of the difference between the means
        se_diff = np.sqrt(group1.var(ddof=1)/len(group1) + group2.var(ddof=1)/len(group2))
        # Confidence intervals
        ci_lower = (group1.mean() - group2.mean()) - critical_z_two_tailed * se_diff
        ci_upper = (group1.mean() - group2.mean()) + critical_z_two_tailed * se_diff

        # Print results with descriptive labels
        print(f"Comparing means of '{colname}' between {compare_labels[0]} and {compare_labels[1]}:")
        print(f"Z-statistic: {z_stat:.3f}")
        print(f"P-value: {p_val:.3f}")
        print(f"Degrees of freedom: {dfree}")
        print(f"Critical t-value for two-tailed test: ±{critical_z_two_tailed:.3f}")
        print(f"Critical t-value for one-tailed test: {critical_z_one_tailed:.3f}")
        print(f"95% confidence interval for the difference in means of '{colname}': ({ci_lower:.3f}, {ci_upper:.3f})")

        # Conclusions with descriptive labels
        # Two-tailed test
        if abs(z_stat) > critical_z_two_tailed:
            print(f"Reject the null hypothesis for a two-tailed test: significant difference in means of '{colname}' between {compare_labels[0]} and {compare_labels[1]}.")
        else:
            print(f"Fail to reject the null hypothesis for a two-tailed test: no significant difference in means of '{colname}' between {compare_labels[0]} and {compare_labels[1]}.")

        # One-tailed test (assuming testing if mean of first group is greater than second)
        if z_stat > critical_z_one_tailed:
            print(f"Reject the null hypothesis for a one-tailed test: {compare_labels[0]} mean of '{colname}' is significantly greater than {compare_labels[1]} mean of '{colname}'.")
        else:
            print(f"Fail to reject the null hypothesis for a one-tailed test: {compare_labels[0]} mean of '{colname}' is not significantly greater than {compare_labels[1]} mean of '{colname}'.")

def compare_means_anova(df: pd.DataFrame, colname: str, compare_col: str, compare_vals: List, compare_labels: List, alpha=0.05) -> None:
    # Validate inputs
    if len(compare_vals) != len(compare_labels):
        raise ValueError("compare_vals and compare_labels must have the same length.")

    groups = [df[df[compare_col] == val][colname] for val in compare_vals]

    # Perform ANOVA
    f_stat, p_val = stats.f_oneway(*groups)

    # Print results with descriptive labels
    print(f"ANOVA results for comparing means of '{colname}' across groups defined in '{compare_col}':")
    print(f"F-statistic: {f_stat:.3f}")
    print(f"P-value: {p_val:.3f}")

    # Conclusions
    if p_val < alpha:
        print(f"Reject the null hypothesis: At least one group mean of '{colname}' is significantly different from others at α = {alpha}.")
    else:
        print(f"Fail to reject the null hypothesis: No significant difference in means of '{colname}' across groups at α = {alpha}.")

def check_normality(df: pd.DataFrame, colname: str, y_col: str, compare_col: Optional[str] = None, compare_vals: Optional[List] = None, compare_labels: Optional[List[str]] = None) -> None:
    # Check normality for the whole column
    stat, p = stats.shapiro(df[colname].dropna())
    print(f"Shapiro-Wilk Test for {colname} (whole dataset):")
    print(f"Statistics={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print("Data looks normally distributed (fail to reject H0).\n")
    else:
        print("Data does not look normally distributed (reject H0).\n")

    stat, p = stats.shapiro(df[y_col].dropna())
    print(f"Shapiro-Wilk Test for {y_col} (whole dataset):")
    print(f"Statistics={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print("Data looks normally distributed (fail to reject H0).\n")
    else:
        print("Data does not look normally distributed (reject H0).\n")

    # If groups are specified, check normality for each group
    if compare_col and compare_vals and compare_labels and len(compare_vals) == len(compare_labels):
        for val, label in zip(compare_vals, compare_labels):
            group_data = df[df[compare_col] == val][colname].dropna()
            stat, p = stats.shapiro(group_data)
            print(f"Shapiro-Wilk Test for {colname} in group '{label}':")
            print(f"Statistics={stat:.3f}, p={p:.3f}")
            if p > 0.05:
                print(f"Data in group '{label}' looks normally distributed (fail to reject H0).\n")
            else:
                print(f"Data in group '{label}' does not look normally distributed (reject H0).\n")

    # If groups are specified, check normality for each group
    if compare_col and compare_vals and compare_labels and len(compare_vals) == len(compare_labels):
        for val, label in zip(compare_vals, compare_labels):
            group_data = df[df[compare_col] == val][y_col].dropna()
            stat, p = stats.shapiro(group_data)
            print(f"Shapiro-Wilk Test for {y_col} in group '{label}':")
            print(f"Statistics={stat:.3f}, p={p:.3f}")
            if p > 0.05:
                print(f"Data in group '{label}' looks normally distributed (fail to reject H0).\n")
            else:
                print(f"Data in group '{label}' does not look normally distributed (reject H0).\n")

def calculate_correlation(df: pd.DataFrame, colname1: str, colname2: str, compare_col: Optional[str] = None, compare_vals: Optional[List] = None, compare_labels: Optional[List[str]] = None, method: str = 'pearson') -> None:
    # Validate method
    if method not in ['pearson', 'spearman']:
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    # Calculate correlation for the whole dataset
    if method == 'pearson':
        corr, p_value = stats.pearsonr(df[colname1], df[colname2])
    else:
        corr, p_value = stats.spearmanr(df[colname1], df[colname2])

    print(f"{method.capitalize()} correlation between {colname1} and {colname2} (whole dataset):")
    print(f"Correlation coefficient={corr:.3f}, p-value={p_value:.3f}\n")

    # If groups are specified, calculate correlation within each group
    if compare_col and compare_vals and compare_labels and len(compare_vals) == len(compare_labels):
        for val, label in zip(compare_vals, compare_labels):
            group_df = df[df[compare_col] == val]
            if method == 'pearson':
                corr, p_value = stats.pearsonr(group_df[colname1], group_df[colname2])
            else:
                corr, p_value = stats.spearmanr(group_df[colname1], group_df[colname2])

            print(f"{method.capitalize()} correlation between {colname1} and {colname2} in group '{label}':")
            print(f"Correlation coefficient={corr:.3f}, p-value={p_value:.3f}\n")

def perform_regression_and_plot(df: pd.DataFrame, colname1: str, colname2: str, compare_col: Optional[str] = None, compare_vals: Optional[List] = None, compare_labels: Optional[List[str]] = None) -> None:
    # Check if group comparison is needed
    if compare_col and compare_vals and compare_labels:
        for i, val in enumerate(compare_vals):
            group_df = df[df[compare_col] == val]
            plot_regression(group_df[colname1], group_df[colname2], colname1, colname2, f'Group {compare_labels[i]}')
    else:
        plot_regression(df[colname1], df[colname2], colname1, colname2)
