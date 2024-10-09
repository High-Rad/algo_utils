"""
Algorithm Utilities: This module contains various functions for data analysis.
"""

import os
from datetime import datetime
from time import gmtime, time
from typing import Optional, Dict, Tuple, Callable, List

import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_col_to_name


__all__ = ['write_to_excel', 'calculate_runtime', 'print_full_df', 'scans_sort_key', 'pairs_sort_key',
           'sort_dataframe_by_key', 'CaseName', 'PairName']


def write_to_excel(df: pd.DataFrame, writer: pd.ExcelWriter, columns_order: List[str], column_name_as_index: str,
                   sheet_name: str = 'Sheet1',
                   f1_scores: Optional[Dict[str, Tuple[str, str]]] = None):
    """
    Write a dataframe to an excel file with the following format:
        - The dataframe is written to the given sheet name
        - The columns are ordered according to the columns_order list
        - The column_name_as_index is used as the index of the dataframe
        - The first row is the column names
        - The first column is the index
        - The last 5 rows are the mean, std, min, max, and sum of the numerical columns
        - The last 5 rows are highlighted with different colors
        - The f1_scores dictionary is used to calculate the harmonic mean of the precision and recall columns
        - The f1_scores dictionary is a mapping from the f1 column name to a tuple of the precision and recall column
          names
        - The min and max values of each column are highlighted with different colors

    Notes
    -----
    The Excel file should be closed after calling this function.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas dataframe to write to the Excel file.
    writer : pd.ExcelWriter
        A pandas ExcelWriter object to write to.
    columns_order : List[str]
        A list of column names in the order they should appear in the Excel file.
    column_name_as_index : str
        The name of the column to use as the index of the dataframe.
    sheet_name : str
        The name of the sheet to write to. Default is 'Sheet1'.
    f1_scores : Optional[Dict[str, Tuple[str, str]]]
        A dictionary mapping from the f1 column name to a tuple of the precision and recall column names.
        Default is None. If not None, the harmonic mean of the precision and recall columns will be calculated.
    """

    df = df.set_index(column_name_as_index)

    def add_agg_to_df(_df):
        # Separate the numerical and non-numerical columns
        num_df = _df.select_dtypes(include=[np.number])
        non_num_df = _df.select_dtypes(exclude=[np.number])

        # Perform aggregation on the numerical columns
        agg_df = num_df.agg(['mean', 'std', 'min', 'max', 'sum'])

        # Create a new dataframe with NaN for the non-numerical columns
        nan_df = pd.DataFrame(np.nan, index=agg_df.index, columns=non_num_df.columns)

        # Concatenate the NaN dataframe with the aggregated numerical dataframe
        agg_full_df = pd.concat([nan_df, agg_df], axis=1)

        # Concatenate the original dataframe with the full aggregated dataframe
        result_df = pd.concat([_df, agg_full_df], ignore_index=False)

        return result_df

    # df = pd.concat([df, df.agg(['mean', 'std', 'min', 'max', 'sum'])], ignore_index=False)
    df = add_agg_to_df(df)

    workbook = writer.book
    # cell_format = workbook.add_format()
    cell_format = workbook.add_format({'num_format': '#,##0.00'})
    cell_format.set_font_size(16)

    columns_order = [c for c in columns_order if c != column_name_as_index]
    df.to_excel(writer, sheet_name=sheet_name, columns=columns_order, startrow=1, startcol=1, header=False, index=False)
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'font_size': 16,
        'valign': 'top',
        'border': 1})

    max_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#E6FFCC'})
    min_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#FFB3B3'})
    last_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#C0C0C0',
        'border': 1,
        'num_format': '#,##0.00'})

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 1)

    n = df.shape[0] - 5
    for col in np.arange(len(columns_order)) + 1:
        for i, measure in enumerate(['AVERAGE', 'STDEV', 'MIN', 'MAX', 'SUM'], start=1):
            col_name = xl_col_to_name(col)
            worksheet.write(f'{col_name}{n + i + 1}', f'{{={measure}({col_name}2:{col_name}{n + 1})}}')

    if f1_scores is not None:
        for col_name in f1_scores:
            f1_col_name = xl_col_to_name(columns_order.index(col_name) + 1)
            p_col_name = xl_col_to_name(columns_order.index(f1_scores[col_name][0]) + 1)
            r_col_name = xl_col_to_name(columns_order.index(f1_scores[col_name][1]) + 1)
            worksheet.write(f'{f1_col_name}{n + 2}', f'{{=HARMEAN({p_col_name}{n + 2}:{r_col_name}{n + 2})}}')
            for i in range(1, 5):
                worksheet.write(f'{f1_col_name}{n + 2 + i}', " ")

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4),
                                 {'type': 'formula',
                                  'criteria': '=B2=B$' + str(len(df.axes[0])),
                                  'format': max_format})

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4),
                                 {'type': 'formula',
                                  'criteria': '=B2=B$' + str(
                                      len(df.axes[0]) - 1),
                                  'format': min_format})

    for i in range(len(df.axes[0]) - 4, len(df.axes[0]) + 1):
        worksheet.set_row(i, None, last_format)

    for col_num, value in enumerate(columns_order):
        worksheet.write(0, col_num + 1, value, header_format)
    for row_num, value in enumerate(df.axes[0].astype(str)):
        worksheet.write(row_num + 1, 0, value, header_format)

    # Fix first column
    column_len = df.axes[0].astype(str).str.len().max() + df.axes[0].astype(str).str.len().max() * 0.5
    worksheet.set_column(0, 0, column_len, cell_format)

    # Fix all  the rest of the columns
    for i, col in enumerate(columns_order):
        # find length of column i
        column_len = df[col].astype(str).str.len().max()
        # Setting the length if the column header is larger
        # than the max column value length
        column_len = max(column_len, len(col))
        column_len += column_len * 0.5
        # set the column length
        worksheet.set_column(i + 1, i + 1, column_len, cell_format)


def calculate_runtime(t: float) -> str:
    """
    Calculates the time passed from the given time stamp t in the format HH:MM:SS.

    Parameters
    ----------
    t : float
        A time stamp in seconds.

    Returns
    -------
    str
        The time passed from the time stamp t in the format HH:MM:SS.

    Notes
    -----
    The time stamp t should be in seconds and should be obtained using the time() function from the time module.

    Examples
    -----
    >>> from time import time
    ...
    >>> tm = time()
    ...
    >>> print(f'Runtime: {calculate_runtime(tm)}')
    """
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'


def print_full_df(x: pd.DataFrame):
    """
    Print the full dataframe without truncating the rows.

    Parameters
    ----------
    x : pd.DataFrame
        The dataframe to print.
    """
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def scans_sort_key(name: str, full_path_is_given: bool = False) -> Tuple[str, int, int, int]:
    """
    Sort the scans by their name. The scans should be named as follows:
        - The name should be in the format '<name>_DD_MM_YYYY.nii.gz' where:
            - <name> is a underscore separated string of characters representing the patient name initials.
            - DD is the day of the scan.
            - MM is the month of the scan.
            - YYYY is the year of the scan.
        - The scans will be sorted by patient name, year, month, and day.

    Parameters
    ----------
    name : str
        The name of the scan.
    full_path_is_given : bool
        A boolean flag indicating whether the full path of the scan is given. Default is False. If True, the name of
        the scan will be extracted from the basename of the full path.

    Returns
    -------
    Tuple[str, int, int, int]
        A tuple of the patient name, year, month, and day of the scan.
    """
    if full_path_is_given:
        name = os.path.basename(name)
    split = name.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def pairs_sort_key(name: str, full_path_is_given: bool = False) -> Tuple[str, int, int, int, int, int, int]:
    """
    Sort the pairs of scans by their name. The pairs should be named as follows:
        - The name should be in the format 'BL_<bl_scan_name>_FU_<fu_scan_name>.nii.gz' where:
            - <bl_scan_name> is the name of the baseline scan.
            - <fu_scan_name> is the name of the follow-up scan.
            - Each scan name should be in the format '<name>_DD_MM_YYYY.nii.gz' where:
                - <name> is a underscore separated string of characters representing the patient name initials.
                - DD is the day of the scan.
                - MM is the month of the scan.
                - YYYY is the year of the scan.
        - The pairs will be sorted by patient name, baseline year, baseline month, baseline day, follow-up year,
          follow-up month, and follow-up day.

    Parameters
    ----------
    name : str
        The name of the pair of scans.
    full_path_is_given : bool
        A boolean flag indicating whether the full path of the pair is given. Default is False. If True, the name of
        the pair will be extracted from the basename of the full path.

    Returns
    -------
    Tuple[str, int, int, int, int, int, int]
        A tuple of the patient name, baseline year, baseline month, baseline day, follow-up year, follow-up month, and
        follow-up day of the pair of scans.
    """
    if full_path_is_given:
        name = os.path.basename(name)
    name = name.replace('BL_', '')
    bl_name, fu_name = name.split('_FU_')
    bl_scan_key: Tuple[str, int, int, int] = scans_sort_key(bl_name)
    fu_scan_key: Tuple[str, int, int, int] = scans_sort_key(fu_name)
    assert bl_scan_key[0] == fu_scan_key[0], f'Patient names do not match: {bl_scan_key[0]} != {fu_scan_key[0]}'
    sort_key = bl_scan_key + fu_scan_key[1:]
    return sort_key


def sort_dataframe_by_key(dataframe: pd.DataFrame, column: str, key: Callable) -> pd.DataFrame:
    """
    Sort a dataframe by a column using the key.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The dataframe to sort.
    column : str
        The column to sort by.
    key : Callable
        The key function to use for sorting.

    Returns
    -------
    pd.DataFrame
        The sorted dataframe.
    """
    sort_ixs = sorted(np.arange(len(dataframe)), key=lambda i: key(dataframe.iloc[i][column]))
    return pd.DataFrame(columns=list(dataframe), data=dataframe.iloc[sort_ixs].values)


class CaseName:
    """
    Represents a case name with extracted patient name and date information.

    Parameters
    ----------
    case_name : str
        The input string representing the case name.

    Attributes
    ----------
    case_name : str
        The original input case name.
    patient_name : str
        The extracted patient name from the case name.
    date : datetime.date
        The extracted date associated with the case name.

    Notes
    -----
    This class parses the input `case_name` to extract the patient name and date.
    The `case_name` should follow a specific format: '<patien_name>_<date>'.
    It is assumed that the date part of the `case_name` follows the format: 'DD_MM_YYYY',
    which is then converted into a `datetime.date` object for easy handling, and that the patient name
    doesn't include numbers.

    Examples
    --------
    >>> case_name = "A_Ya_25_07_2023"
    >>> case = CaseName(case_name)
    >>> case.patient_name
    'A_Ya'
    >>> case.date
    datetime.date(2023, 7, 25)
    """

    def __init__(self, case_name):
        self.case_name = case_name
        self.patient_name = '_'.join(c for c in case_name.split('_') if not c.isdigit())
        self.date = datetime.strptime('_'.join(c for c in case_name.split('_') if c.isdigit()), '%d_%m_%Y').date()


class PairName:
    """
    Represents a pair of case names with associated information.

    Parameters
    ----------
    pair_name : str
        The input string representing the pair name.

    Attributes
    ----------
    pair_name : str
        The original input pair name.
    patient_name : str
        The common patient name extracted from the baseline and follow-up case names.
    bl_case_name : str
        The baseline case name extracted from the pair name.
    bl_date : datetime.date
        The date associated with the baseline case name.
    fu_case_name : str
        The follow-up case name extracted from the pair name.
    fu_date : datetime.date
        The date associated with the follow-up case name.

    Raises
    ------
    AssertionError
        If the extracted patient names from the baseline and follow-up
        case names do not match.

    Notes
    -----
    This class parses the input `pair_name` to extract relevant information
    such as patient name, baseline case name, baseline date, follow-up case
    name, and follow-up date. The `pair_name` should follow a specific format:
    'BL_<case_name>_FU_<case_name>'. It is assumed that the `CaseName` class
    has been defined elsewhere to handle case name parsing.

    Examples
    --------
    >>> pair_name = "BL_A_Ay_25_07_2023_FU_A_Ay_15_09_2023"
    >>> pair = PairName(pair_name)
    >>> pair.patient_name
    'A_Ay'
    >>> pair.bl_case_name
    'A_Ay_25_07_2023'
    >>> pair.bl_date
    datetime.date(2023, 7, 25)
    >>> pair.fu_case_name
    'A_Ay_15_09_2023'
    >>> pair.fu_date
    datetime.date(2023, 9, 15)
    """

    def __init__(self, pair_name):
        self.pair_name = pair_name

        bl_case_name = CaseName(pair_name.replace('BL_', '').split('_FU_')[0])
        fu_case_name = CaseName(pair_name.split('_FU_')[1])

        assert bl_case_name.patient_name == fu_case_name.patient_name

        self.patient_name = bl_case_name.patient_name

        self.bl_case_name = bl_case_name.case_name
        self.bl_date = bl_case_name.date

        self.fu_case_name = fu_case_name.case_name
        self.fu_date = fu_case_name.date
