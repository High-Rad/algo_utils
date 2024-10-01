import os
from time import gmtime, time
from typing import Optional, Dict, Tuple, Callable

import numpy as np
import pandas as pd
from xlsxwriter.utility import xl_col_to_name


def write_to_excel(df, writer, columns_order, column_name_as_index, sheet_name='Sheet1',
                   f1_scores: Optional[Dict[str, Tuple[str, str]]] = None):
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


def calculate_runtime(t):
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'


def print_full_df(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def scans_sort_key(name, full_path_is_given=False):
    if full_path_is_given:
        name = os.path.basename(name)
    split = name.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def pairs_sort_key(name, full_path_is_given=False):
    if full_path_is_given:
        name = os.path.basename(name)
    name = name.replace('BL_', '')
    bl_name, fu_name = name.split('_FU_')
    return (*scans_sort_key(bl_name), *scans_sort_key(fu_name))


def sort_dataframe_by_key(dataframe: pd.DataFrame, column: str, key: Callable) -> pd.DataFrame:
    """ Sort a dataframe from a column using the key """
    sort_ixs = sorted(np.arange(len(dataframe)), key=lambda i: key(dataframe.iloc[i][column]))
    return pd.DataFrame(columns=list(dataframe), data=dataframe.iloc[sort_ixs].values)
