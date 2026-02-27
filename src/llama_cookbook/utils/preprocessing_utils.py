# function to remove html tags from  a string
from bs4 import BeautifulSoup
import html
import ast
import pandas as pd
from typing import Dict, List
import re
import numpy as np
import matplotlib.pyplot as plt

def clean_html(raw_html: str) -> str:
    # Remove malformed character references
    cleantext = raw_html
    try:
        cleantext = BeautifulSoup(raw_html, "html.parser").get_text()
        cleantext = html.unescape(cleantext)
    except Exception as e:
        pass
    return cleantext

def get_distribution(df, column= 'crowd_product_type', visualize=True):
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    counts['percentile'] = counts['count'] / counts['count'].sum() * 100
    counts = counts.sort_values('count', ascending=False)
    counts['cumulative_count'] = counts['count'].cumsum()
    counts['cumulative_percentile'] = 100 * counts['cumulative_count'] / counts['count'].sum()
    # plot cumulative_percentile vs index and count vs index in the side by side plot
    if visualize:
        visualize_distribution(counts)
    return counts

def visualize_distribution(counts, start = 0):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(start, len(counts)), counts['cumulative_percentile'][start:], marker='o')
    plt.xlabel('Index')
    plt.ylabel('Cumulative Percentile')
    plt.title(f'Cumulative Percentile Plot')
    plt.subplot(1, 2, 2)
    plt.plot(range(start, len(counts)), counts['count'][start:], marker='o')
    plt.xlabel('Index')
    plt.ylabel('Count')
    plt.title(f'Count Plot')
    plt.tight_layout()
    plt.show()

def partition_dataset(df, column = 'crowd_product_type', on= 'cumulative_percentile', threshold=95):
    counts = get_distribution(df, column=column)
    if on == 'index':
        selected_types = counts[column][:threshold]
        print(f"partitioning on top {threshold} {column}")
    elif on in ['cumulative_percentile', 'cumulative_count']:
        selected_types = counts[counts[on] <= threshold][column]
        print(f"partitioning on data with {on} <= {threshold}")
    elif on in ['percentile', 'count']:
        selected_types = counts[counts[on] >= threshold][column]
        print(f"partitioning on data with {on} >= {threshold}")
    else:
        raise ValueError(f"Invalid 'on' value: {on}. Must be one of 'cumulative_percentile', 'cumulative_count', 'percentile', or 'count'.")
    last_index = len(selected_types) - 1
    print(f"Selected {len(selected_types)} {column} out of {len(counts)} total types.")
    print(f"selected upto: count: {counts['count'][last_index]}, percentile: {counts['percentile'][last_index]:.2f}%, cumulative_count: {counts['cumulative_count'][last_index]}, cumulative_percentile: {counts['cumulative_percentile'][last_index]:.2f}%")
    df1, df2 = df[df[column].isin(selected_types)], df[~df[column].isin(selected_types)]
    print(len(df1), len(df2))
    return df1, df2


def stratified_sample(df, column, n_samples):
    # Calculate group probabilities
    group_counts = df[column].value_counts()
    probs = df[column].map(group_counts / group_counts.sum())

    # Sample indices with probabilities
    sample_indices = np.random.choice(df.index, size=n_samples, replace=False, p=probs / probs.sum())
    return df.loc[sample_indices]