"""
    Discretization 과정:
        1) CSV를 Table로 변환
        2) Table 데이터에 discretization 적용
        3) Table을 CSV로 변환

    Input : 오용 탐지에 사용되는 Known 데이터 셋
    Output : discretization이 수행된 Known 데이터 셋(Label이 공격인 데이터 한정)

    데이터 Discretization 수행에서 Orage library에 있는 discretization 사용
    따라서, CSV 데이터를 Table 데이터(Orange library 전용 데이터)로 변환 필요
"""

import numpy as np
import pandas as pd
import Orange
from pandas.api.types import (
    is_categorical_dtype, is_object_dtype,
    is_datetime64_any_dtype, is_numeric_dtype,
)

from Orange.data import (
    Table, Domain, DiscreteVariable, StringVariable, TimeVariable,
    ContinuousVariable,
)
__all__ = ['table_from_frame']

def table_from_frame(df,class_name, *, force_nominal=False):
    """
    Convert pandas.DataFrame to Orange.data.Table

    Parameters
    ----------
    df : pandas.DataFrame
    force_nominal : boolean
        If True, interpret ALL string columns as nominal (DiscreteVariable).

    Returns
    -------
    Table
    """

    def _is_discrete(s):
        return (is_categorical_dtype(s) or
                is_object_dtype(s) and (force_nominal or
                                        s.nunique() < s.size**.666))

    def _is_datetime(s):
        if is_datetime64_any_dtype(s):
            return True
        try:
            if is_object_dtype(s):
                pd.to_datetime(s, infer_datetime_format=True)
                return True
        except Exception:  # pylint: disable=broad-except
            pass
        return False

    # If df index is not a simple RangeIndex (or similar), put it into data
    if not (df.index.is_integer() and (df.index.is_monotonic_increasing or
                                       df.index.is_monotonic_decreasing)):
        df = df.reset_index()

    attrs, metas,calss_vars = [], [],[]
    X, M = [], []

    # Iter over columns
    for name, s in df.items():
        name = str(name)
        if name == class_name:
            discrete = s.astype('category').cat
            calss_vars.append(DiscreteVariable(name, discrete.categories.astype(str).tolist()))
            X.append(discrete.codes.replace(-1, np.nan).values)
        elif _is_discrete(s):
            discrete = s.astype('category').cat
            attrs.append(DiscreteVariable(name, discrete.categories.astype(str).tolist()))
            X.append(discrete.codes.replace(-1, np.nan).values)
        elif _is_datetime(s):
            tvar = TimeVariable(name)
            attrs.append(tvar)
            s = pd.to_datetime(s, infer_datetime_format=True)
            X.append(s.astype('str').replace('NaT', np.nan).map(tvar.parse).values)
        elif is_numeric_dtype(s):
            attrs.append(ContinuousVariable(name))
            X.append(s.values)
        else:
            metas.append(StringVariable(name))
            M.append(s.values.astype(object))

    return Table.from_numpy(Domain(attrs, calss_vars, metas),
                            np.column_stack(X) if X else np.empty((df.shape[0], 0)),
                            None,
                            np.column_stack(M) if M else None)

def column2df(col, num):
    record = [str(col[i]) for i in range(len(col))]
    return record

def table2df(tab):
    # Orange.data.Table().to_numpy() cannot handle strings
    # So we must build the array column by column,
    # When it comes to strings, python list is used
    record_list = [column2df(tab[i], i) for i in range(len(tab))]
    feature_name = [str(tab.domain[i]) for i in range(len(tab.domain))]
    return pd.DataFrame(record_list, columns=feature_name)

print("데이터 호출(CSV -> Table)")
Input_data = pd.read_csv('total_data.csv')

temp_table = table_from_frame(Input_data, 'class')
#temp_table.save("F:/data/ADD/201023_data/Known_attack_data.tab")

print("\nDiscretization 수행")
disc = Orange.preprocess.Discretize()
disc.method = Orange.preprocess.discretize.EntropyMDL()
disc_data = disc(temp_table)
#disc_data.save("F:/data/ADD/201023_data/disc_data.tab")
print("\nDiscretization 수행 완료")
print("Discretization Record : %s"%len(disc_data))

print("\nTable -> CSV")
temp_df = table2df(disc_data)

print("Dataframe Record : %s"%len(temp_df))
temp_df.to_csv("di_total_data.csv", index=False)