"""
File: sankey.py
Description: A simple library for building sankey diagrams from a dataframe
Author: John Rachlin, modified by Benjamin Pierce
Date: 10/1/23
"""

import plotly.graph_objects as go
import pandas as pd
import json


def load_data_from_json(file_path):
    """
    load data from JSON file and return it as pandas DataFrame

    Args:
        file_path (str): path to the JSON file

    Returns:
        pd.DataFrame: pandas DataFrame containing data from JSON file
    """

    with open(file_path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)


def convert_decade_data(df):
    """
    convert DataFrame containing year information into DataFrame with 'decade' column

    Args:
        df (pd.DataFrame): input DataFrame containing year information

    Returns:
        df (pd.DataFrame): DataFrame with additional 'decade' column
    """

    # calculate decade
    df['birth_year'] = df['BeginDate'].astype(int)
    df['decade'] = (df['birth_year'] // 10) * 10

    return df[['Nationality', 'Gender', 'decade']]


def aggregate_data(df, group_columns):
    """
    aggregate data in a DataFrame based on specified columns

    Args:
        df (pd.DataFrame): input DataFrame.
        group_columns (list): list of columns to group by

    Returns:
        grouped_data (pd.DataFrame): DataFrame with aggregated data
    """

    # aggregate the data based on specified columns
    grouped_data = df.groupby(group_columns).size().reset_index(name='count')

    return grouped_data


def filter_data(df, filter_criteria):
    """
    filter data in DataFrame based on specified filter criteria

    Args:
        df (pd.DataFrame): input DataFrame
        filter_criteria (dict): dictionary of column-value pairs for filtering

    Returns:
        filtered_data (pd.DataFrame): DataFrame with filtered data
    """

    if filter_criteria is None:
        return df
    else:
        # filter out rows with specified criteria
        filtered_data = df.copy()
        for col, value in filter_criteria.items():
            if value is not None:
                filtered_data = filtered_data[filtered_data[col] != value]
            else:
                filtered_data = filtered_data[df[col].isna()]
        return filtered_data


def filter_by_threshold(df, threshold):
    """
    filter data in DataFrame by threshold count

    Args:
        df (pd.DataFrame): input DataFrame
        threshold (int): threshold count for filtering

    Returns:
       df (pd.DataFrame): DataFrame with data filtered by threshold
    """

    if threshold is None:
        return df
    else:
        # filter out rows below threshold
        return df[df['count'] >= threshold]


def _code_mapping(df, src, targ):
    """
    map labels in a DataFrame to integer codes

    Args:
        df (pd.DataFrame): input DataFrame
        src (str): source column name
        targ (str): target column name

    Returns:
        df (pd.DataFrame): DataFrame with labels replaced by integer codes
        labels (list): list of unique labels
    """

    # get the distinct labels from src/targ columns
    labels = list(set(list(df[src]) + list(df[targ])))

    # generate n integers for n labels
    codes = list(range(len(labels)))

    # create a map from label to code
    lc_map = dict(zip(labels, codes))

    # substitute names for codes in the dataframe
    df = df.replace({src: lc_map, targ: lc_map})

    # Return modified dataframe and list of labels
    return df, labels


def make_sankey(df, *cols, vals=None, save=None, **kwargs):
    """
    create a Sankey diagram from a DataFrame and specified columns

    Args:
        df (pd.DataFrame): input DataFrame
        *cols: variable-length arguments representing source and target columns
        vals (str): column name representing values
        save (str): file path to save diagram
        **kwargs: additional keyword arguments

    Returns:
        fig (go.Figure): a Plotly Figure containing the Sankey diagram
    """

    if len(cols) < 2:
        raise ValueError('Must pass at least 2 columns!')

    if len(cols) == 2:
        # 2 layered Sankey diagram
        src, targ = cols

    else:
        # multilayered Sankey diagram
        temp_dfs = []

        for i in range(len(cols)-1):
            temp = df[[cols[i], cols[i+1], vals]]
            temp.columns = ['src', 'targ', 'count']
            temp_dfs.append(temp)

        stacked_df = pd.concat(temp_dfs, axis=0)
        df = stacked_df

        src = df.columns[0]
        targ = df.columns[1]

    if vals:
        values = df[vals]
    else:
        values = [1] * len(df)

    # convert df labels to integer values
    df, labels = _code_mapping(df, src, targ)

    link = {'source': df[src], 'target': df[targ], 'value': values,
            'line': {'color': 'black', 'width': 1}}

    node_thickness = kwargs.get("node_thickness", 50)

    node = {'label': labels, 'pad': 50, 'thickness': node_thickness,
            'line': {'color': 'black', 'width': 1}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)

    # For dashboarding, you will want to return the fig
    # rather than show the fig.
    return fig

    # This requires installation of kaleido library
    # https://pypi.org/project/kaleido/
    # See: https://anaconda.org/conda-forge/python-kaleido
    # conda install -c conda-forge python-kaleido

    # if save:
    #    fig.write_image(save)


def generate_and_show_sankey(data, group_columns, vals, filter_criteria=None, threshold=None, title=None):
    """
    generate and display a Sankey diagram based on data and specified parameters

    Args:
        data (pd.DataFrame): input DataFrame
        group_columns (list): list of columns to group by
        filter_criteria (dict): dictionary of column-value pairs for filtering
        threshold (int): threshold count for filtering
        title (str): title for Sankey diagram

    Returns:
        None
    """

    # aggregate data
    #grouped_data = aggregate_data(data, group_columns)

    # filter out data based on specified criteria
    #filtered_data = filter_data(grouped_data, filter_criteria)

    # filter out rows below threshold
    #filtered_data = filter_by_threshold(filtered_data, threshold)

    # create and display Sankey diagram
    sankey_diagram = make_sankey(data, *group_columns, vals=vals, title=title)
    sankey_diagram.show()
