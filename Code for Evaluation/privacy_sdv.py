import random
import time
from operator import attrgetter

import pandas as pd
from sdv.metrics.tabular import *

import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', None)

other_metrics_dict = {
    'BNLikelihood': BNLikelihood,
    'BNLogLikelihood': BNLogLikelihood,
    'LogisticDetection': LogisticDetection,
    'SVCDetection': SVCDetection,
    'GMLogLikelihood': GMLogLikelihood,
    'CSTest': CSTest,
    'KSTest': KSComplement,
    'ContinuousKLDivergence': ContinuousKLDivergence,
    'DiscreteKLDivergence': DiscreteKLDivergence
}

privacy_categorical_metrics_dict = {
    'CategoricalCAP': CategoricalCAP,
    'CategoricalZeroCAP': CategoricalZeroCAP,
    'CategoricalGeneralizedCAP': CategoricalGeneralizedCAP,
    'CategoricalKNN': CategoricalKNN,
    'CategoricalNB': CategoricalNB,
    'CategoricalRF': CategoricalRF,
    # 'CategoricalEnsemble': CategoricalEnsemble
}

privacy_numerical_metrics_dict = {
    'NumericalMLP': NumericalMLP,
    'NumericalLR': NumericalLR,
    'NumericalSVR': NumericalSVR,
    # 'NumericalRadiusNearestNeighbor': NumericalRadiusNearestNeighbor # This metrics is too slow, so we will not use.
}


def create_table(real_data, synthetic_data, numerical_columns, key_fields, sensitive_fields, target_block_size=100):
    """
    The function to create table
    :param real_data: The original data
    :param synthetic_data: The synthetic data generate by tools, should be a List of pandas.DataFrame
    :param numerical_columns: A list of string, telling the function what columns are numerical columns
    :param key_fields: The key field that we need to calculate privacy and utility
    :param sensitive_fields: The sensitive field that we need to calculate privacy and utility
    :param target_block_size: How many rows in each block, the final result will be average of all blocks
    :return: result_n(c)_avg: the numerical(categorical) result for each metrics,
    numerical(categorical)_avg: the average of all metrics, total_score: the average of numerical and categorical
    """
    total_len = len(real_data)
    total_len = total_len // target_block_size * target_block_size

    real_n, real_c = split_table(real_data, numerical_col=numerical_columns)
    key_fields_n, key_fields_c, sensitive_fields_n, sensitive_fields_c = get_meta_data(real_n, real_c, key_fields,
                                                                                       sensitive_fields)
    data_dict_n = {
        'real': real_n
    }
    data_dict_c = {
        'real': real_c
    }
    for i, syn_data in enumerate(synthetic_data):
        syn_n, syn_c = split_table(syn_data, numerical_columns)
        data_dict_n[f"level{i}"] = syn_n
        data_dict_c[f"level{i}"] = syn_c

    data_dict_n = remove_nan_in_numerical(data_dict_n)

    meta_data_n = {
        'key_fields': key_fields_n,
        'sensitive_fields': sensitive_fields_n,
    }
    meta_data_c = {
        'key_fields': key_fields_c,
        'sensitive_fields': sensitive_fields_c,
    }
    metric_list_c = privacy_categorical_metrics_dict.keys()
    metric_list_n = privacy_numerical_metrics_dict.keys()

    result_n = []
    result_c = []

    for index in range(0, total_len, target_block_size):
        partial_n = get_part_of_data(data_dict_n, target_block_size, index)
        partial_c = get_part_of_data(data_dict_c, target_block_size, index)
        partial_real_n = partial_n['real']
        partial_real_c = partial_c['real']
        partial_n.pop('real')
        partial_c.pop('real')
        numerical = _create_table(metric_list_n, partial_real_n,
                                  partial_n, meta_data_n, mode='n')

        categorical = _create_table(metric_list_c, partial_real_c,
                                    partial_c, meta_data_c, mode='c')
        result_n.append(numerical)
        result_c.append(categorical)
    result_n_avg = sum(result_n) / len(result_n)
    result_c_avg = sum(result_c) / len(result_c)
    numerical_avg = result_n_avg.mean()
    numerical_avg['time'] *= len(synthetic_data)
    categorical_avg = result_c_avg.mean()
    categorical_avg['time'] *= len(synthetic_data)

    total_score = (numerical_avg + categorical_avg) / 2

    return result_n_avg, result_c_avg, numerical_avg, categorical_avg, total_score


def get_meta_data(real_data_n, real_data_c, key_fields, sensitive_fields):
    key_fields_c = [key for key in key_fields if key in real_data_c.columns]
    key_fields_n = [key for key in key_fields if key in real_data_n.columns]
    sensitive_fields_c = [key for key in sensitive_fields if key in real_data_c.columns]
    sensitive_fields_n = [key for key in sensitive_fields if key in real_data_n.columns]
    return key_fields_n, key_fields_c, sensitive_fields_n, sensitive_fields_c


def _create_table(metrics_list, _real_data, synthetic_data_dict: dict, meta_data, mode):
    global privacy_numerical_metrics_dict, privacy_categorical_metrics_dict
    # all_df = {}
    # for meta_data in meta_data_list:
    result = {}
    times = {}
    metrics_dict = privacy_numerical_metrics_dict if mode == 'n' else privacy_categorical_metrics_dict
    success_metrics = []
    for level, data in synthetic_data_dict.items():
        scores = []
        for metrics in metrics_list:
            _start = time.time()
            if metrics not in metrics_dict.keys():
                continue
            metric = metrics_dict[metrics]
            score = metric.compute(real_data=_real_data, synthetic_data=data, **meta_data)
            scores.append(score)
            if metrics not in success_metrics:
                success_metrics.append(metrics)
                times[metrics] = 0
            _end = time.time() - _start
            times[metrics] += _end
        result[level] = scores
    result['time'] = [times[key] for key in success_metrics]
    df = pd.DataFrame(result, index=success_metrics)
    # df['time'] = times
    # all_df[meta_data['sensitive_fields'][0]] = df

    print('one table generated')
    return df


def split_table(data_table: pd.DataFrame, numerical_col=None):
    if numerical_col:
        numerical_data = data_table[numerical_col]
        categorical_data = data_table.drop(numerical_col, axis=1).applymap(str)
        return numerical_data, categorical_data

    dtype_list = data_table.dtypes.apply(attrgetter('kind'))
    numerical_index = []
    categorical_index = []
    for i, dtype in enumerate(dtype_list):
        if dtype == 'f' or dtype == 'i':
            numerical_index.append(i)
        else:
            categorical_index.append(i)
    return data_table.iloc[:, numerical_index], data_table.iloc[:, categorical_index]


def get_metadata(table: pd.DataFrame):
    meta_data_list = []
    clm = list(table.columns)
    # for i, column in enumerate(clm):
    #     m_d = {
    #         'key_fields': clm[:i] + clm[i + 1:],
    #         'sensitive_fields': [column]
    #     }
    #     meta_data_list.append(m_d)
    index = random.randint(0, len(clm) - 1)
    index = [1, 2]
    meta_data_list.append({
        'key_fields': [clm[i] for i in range(len(clm)) if i not in index],
        'sensitive_fields': [clm[i] for i in index]
    })
    return meta_data_list


def remove_nan_in_numerical(numerical_table_dict: dict):
    for k, v in numerical_table_dict.items():
        numerical_table_dict[k] = v.fillna(0)
    return numerical_table_dict


def get_part_of_data(table_dict, n, index):
    new_dict = {}
    # index = random.randint(0, 14691 - n)
    # index = 1
    for k, v in table_dict.items():
        new_dict[k] = v.iloc[index:index + n, :]
        new_dict[k].reset_index(drop=True, inplace=True)
    return new_dict


if __name__ == '__main__':
    numerical_column = ['allwd_amt', 'sbmtd_amt', 'copay_amt', 'coin_amt', 'ded_amt']
    _real_data = pd.read_csv('SyntheticDataSmall_Professional 10082021.csv')
    synthetic_data_small = pd.read_csv('Professional_Obfuscated_SMALL.csv')
    synthetic_data_medium = pd.read_csv('Professional_Obfuscated_MEDIUM.csv')
    synthetic_data_large = pd.read_csv('Professional_Obfuscated_LARGE.csv')

    data_dict = {
        'real': _real_data,
        'small': synthetic_data_small,
        'medium': synthetic_data_medium,
        'large': synthetic_data_large,
    }

    data_dict = get_part_of_data(data_dict, 205, 0)
    _real_data = data_dict['real']
    data_dict.pop('real')
    data = data_dict.values()
    _key_fields = ['Payer.Industry.Person.Identifier..PIPI..secure.', 'Claim.ID..secure.', 'clm_ln_num',
                   'rndrng_prov_cntrct_sts', 'allwd_amt', 'coin_amt']
    _sensitive_fields = ['sbmtd_amt', 'copay_amt', 'Member.ID..secure.']
    rn, rc, an, ac, t = create_table(_real_data, data, numerical_column, _key_fields, _sensitive_fields)

    # real_data_n, real_data_c = split_table(_real_data, numerical_col=numerical_column)
    # synthetic_data_small_n, synthetic_data_small_c = split_table(synthetic_data_small, numerical_col=numerical_column)
    # synthetic_data_medium_n, synthetic_data_medium_c = split_table(synthetic_data_medium,
    #                                                                numerical_col=numerical_column)
    # synthetic_data_large_n, synthetic_data_large_c = split_table(synthetic_data_large, numerical_col=numerical_column)

    # drop the error column
    # synthetic_data_small_n = synthetic_data_small_n.drop(columns=['bill_prov_zip_cd', 'rndrng_prov_zip_cd'])
    # synthetic_data_medium_n = synthetic_data_medium_n.drop(columns=['bill_prov_zip_cd', 'rndrng_prov_zip_cd'])
    # synthetic_data_large_n = synthetic_data_large_n.drop(columns=['rndrng_prov_zip_cd'])
    # real_data_c = real_data_c.drop(columns=['bill_prov_zip_cd', 'rndrng_prov_zip_cd'])
    # synthetic_data_large_c = synthetic_data_large_c.drop(columns=['bill_prov_zip_cd'])

    # group the tables
    # synthetic_data_c = {
    #     'real': real_data_c,
    #     'small': synthetic_data_small_c,
    #     'medium': synthetic_data_medium_c,
    #     'large': synthetic_data_large_c
    # }
    # synthetic_data_n = {
    #     'real': real_data_n,
    #     'small': synthetic_data_small_n,
    #     'medium': synthetic_data_medium_n,
    #     'large': synthetic_data_large_n
    # }
    # synthetic_data_n = remove_nan_in_numerical(synthetic_data_n)
    #
    # mc_list_c = privacy_categorical_metrics_dict.keys()
    # mc_list_n = privacy_numerical_metrics_dict.keys()
    # meta_data_list_n = get_metadata(real_data_n)
    # meta_data_list_c = get_metadata(real_data_c)
    # total_length_n = len(real_data_n)
    # total_length_c = len(real_data_c)
    # num = 100
    # partial_data_n = get_part_of_data(synthetic_data_n, num, 0)
    # partial_data_c = get_part_of_data(synthetic_data_c, num, 0)
    # start = time.time()
    #
    # preal_data_n = partial_data_n['real']
    # partial_data_n.pop('real')
    # numerical_1 = _create_table(mc_list_n, preal_data_n,
    #                             partial_data_n, meta_data_list_n[0], mode='n')
    # preal_data_c = partial_data_c['real']
    # partial_data_c.pop('real')
    # categorical_1 = _create_table(mc_list_c, preal_data_c,
    #                               partial_data_c, meta_data_list_c[0], mode='c')
    #
    # partial_data_n = get_part_of_data(synthetic_data_n, num, 200)
    # partial_data_c = get_part_of_data(synthetic_data_c, num, 200)
    #
    # preal_data_n = partial_data_n['real']
    # partial_data_n.pop('real')
    # numerical_2 = _create_table(mc_list_n, preal_data_n,
    #                             partial_data_n, meta_data_list_n[0], mode='n')
    # preal_data_c = partial_data_c['real']
    # partial_data_c.pop('real')
    # categorical_2 = _create_table(mc_list_c, preal_data_c,
    #                               partial_data_c, meta_data_list_c[0], mode='c')
