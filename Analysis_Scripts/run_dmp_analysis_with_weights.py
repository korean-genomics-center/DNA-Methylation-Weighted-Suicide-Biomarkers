#%%
import math,sys
from itertools import chain
import numpy as np
import scipy
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, PerfectSeparationError
warnings.simplefilter('ignore', ConvergenceWarning)
import statsmodels.api as sm
import statsmodels.genmod.families.links as slink
import statsmodels.formula.api as smf
from  statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2
from joblib import Parallel, delayed
from functools import partial


def main(path_cpg, path_meta, col_sampleid, col_predict, col_positions, col_covariates, col_weights, n_process, path_save_dmp, is_predict_noncategorical = True, control_case_value_for_prediction = (0,1), get_weighted_mean = True, no_depth_weighting = False):
    table_meta = read_tsv(path_meta)
    table_cpg = read_tsv(path_cpg)
    # table_cpg = pd.read_csv(path_cpg, nrows = 10, sep = '\t')
    dict_sample_to_predict = get_sample_to_group_map(table_meta, col_sampleid, col_predict)
    dict_covariates = get_covariate_dictionary(table_meta, col_sampleid, col_covariates)
    dict_weights = get_covariate_dictionary(table_meta, col_sampleid, col_weights)
    
    list_table_cpg_index = list(table_cpg.index)
    list_indexes_split_by_core = split_index_with_core_number(list_table_cpg_index, n_process)
    
    with Parallel(n_jobs = n_process) as parallel:
        list_list_logreg_results = parallel(delayed(process_multiple_row_data)(table_cpg.loc[list_ind, :], dict_sample_to_predict, dict_covariates, dict_weights, col_positions, is_predict_noncategorical, control_case_value_for_prediction, get_weighted_mean, no_depth_weighting) for list_ind in list_indexes_split_by_core)
    list_logreg_results = list(chain(*list_list_logreg_results))
    
    save_dmp_result(list_logreg_results, path_save_dmp)

def read_tsv(path_table):
    return pd.read_csv(path_table, sep = '\t')

def get_sample_to_group_map(table_meta, col_sampleid, col_predict):
    list_sample = table_meta[col_sampleid].to_list()
    list_predict = table_meta[col_predict].to_list()
    
    return dict(zip(list_sample, list_predict))

def get_covariate_dictionary(table_meta, col_sampleid, covariates):
    dict_cov = dict()
    for cov in covariates:
        dict_cov[cov] = get_a_to_b_dictionary_from_dataframe(table_meta, col_sampleid, cov)
    return dict_cov        

def get_a_to_b_dictionary_from_dataframe(table, col_a, col_b):
    return dict(zip(table[col_a].to_list(), table[col_b].to_list()))

def split_index_with_core_number(list_index, n_jobs):
    num_index_per_core = math.ceil(len(list_index) / n_jobs)
    list_splitted_index = list()
    for ind_job in range(n_jobs):
        ind_start = ind_job * num_index_per_core
        ind_end = min((ind_job+1) * num_index_per_core, len(list_index))
        list_index_part = list_index[ind_start:ind_end]
        list_splitted_index.append(list_index_part)
    return list_splitted_index

def process_multiple_row_data(table_part, dict_sample_to_predict, dict_covariates, dict_weights, col_positions, is_predict_noncategorical, control_case_value_for_prediction, get_weighted_mean, no_depth_weighting):
    list_results = list()
    for _, row in table_part.iterrows():
        result = process_single_row_data(row, dict_sample_to_predict, dict_covariates, dict_weights, col_positions, is_predict_noncategorical, control_case_value_for_prediction, get_weighted_mean, no_depth_weighting)
        list_results.append(result)
    return list_results

def process_single_row_data(row_data, dict_sample_to_predict, dict_covariates, dict_weights, col_positions, is_predict_noncategorical, control_case_value_for_prediction, get_weighted_mean, no_depth_weighting):        
    dict_row_data = row_data.to_dict()
    
    list_sample = list(dict_sample_to_predict.keys())
    list_predict_value = list(map(dict_sample_to_predict.__getitem__, list_sample))
    list_depth = list(map(lambda x : dict_row_data.get(x), list(map(lambda x : f"{x}_Depth", list_sample))))
    list_beta = list(map(lambda x : dict_row_data[x]/100, list_sample))
    
    list_sample, list_predict_value, list_depth, list_beta = filter_out_na_sample(list_sample, list_predict_value, list_depth, list_beta)

    dict_covariates_org = reorganize_covariate_dictionary(dict_covariates, list_sample)
    dict_weights_org = reorganize_covariate_dictionary(dict_weights, list_sample)
    additional_weight = multiply_weights_into_single_array(dict_weights_org, len(list_sample))
    
    result = dict()
    for col_pos in col_positions:
        result[col_pos] = row_data[col_pos]
        
    pvalue = run_logistic_regression(list_predict_value, list_beta, list_depth, dict_covariates_org, additional_weight, no_depth_weighting, result)
    meandiff = None
    if is_predict_noncategorical:
        pass
    else:
        meandiff = get_mean_difference_between_two_groups(list_predict_value, list_beta, np.ones(len(list_beta)), control_case_value_for_prediction)
            
    result.update({
        "pvalue" : pvalue,
        "meth.diff" : meandiff
    })
    if get_weighted_mean:
        wmeandiff = get_mean_difference_between_two_groups(list_predict_value, list_beta, additional_weight, control_case_value_for_prediction)
        result["wmeth.diff"] = wmeandiff
        
    return result
   
def filter_out_na_sample(list_sample, list_predict_value, list_depth, list_beta):
    list_sample_ = list()
    list_predict_value_ = list()
    list_depth_ = list()
    list_beta_ = list()
    for sample, predict_val, depth, beta in zip(list_sample, list_predict_value, list_depth, list_beta):
        if pd.isna(beta):
            continue
        else:
            list_sample_.append(sample)
            list_predict_value_.append(predict_val)
            list_depth_.append(depth)
            list_beta_.append(beta)
    return list_sample_, list_predict_value_, list_depth_, list_beta_
 
def reorganize_covariate_dictionary(dict_covariates, total_samples):
    dict_covariates_org = dict()
    for cov in dict_covariates.keys():
        dict_covariates_org[cov] = list(map(dict_covariates[cov].__getitem__, total_samples))
    return dict_covariates_org

def multiply_weights_into_single_array(dict_weights, n_sample):
    weights = np.ones(n_sample)
    for _, weight_values in dict_weights.items():
        weights = weights * np.array(weight_values)
    return weights    
    
def run_logistic_regression(treatment, beta, depth, covariates, additional_weights, no_depth_weighting, position_info):
    data = make_dataframe_for_logit(treatment, beta, covariates)
    if len(data["Y"].unique()) == 1: # All data same
        return 1
    if no_depth_weighting:
        weight = np.array(additional_weights)
    else:
        weight = np.array(depth) * np.array(additional_weights)
    
    formula_alt = "Y ~ " + " + ".join(['X'] + list(covariates.keys()))
    model_alt = smf.glm(formula = formula_alt, data = data, family = sm.families.Binomial(link=slink.logit()), var_weights = np.array(weight))
    
    if len(covariates.keys()) > 0:
        formula_null = "Y ~ " + " + ".join(list(covariates.keys()))
    else:
        formula_null = "Y ~ 1"
    model_null = smf.glm(formula = formula_null, data = data, family = sm.families.Binomial(link=slink.logit()), var_weights = np.array(weight))

    alt_perfect, altresult = fit_model_with_checking_perfect_separation(model_alt)
    null_perfect, nullresult = fit_model_with_checking_perfect_separation(model_null)
    
    if null_perfect: # Covariate explains the methylation perfectly
        print(f"Covariate Perfectly Explains : {position_info}", file = sys.stderr, flush = True)
        return 1 
    elif alt_perfect: # Treatment as a perfect separating variable (Deviance as zero)
        print(f"Methylation Perfectly Explains : {position_info}", file = sys.stderr, flush = True)
        deviance = nullresult.deviance - 0 
        phi = 1
    else:
        deviance = nullresult.deviance - altresult.deviance
        phi = calculate_overdispersion_correction_value_MN(data, weight, altresult)
    deviance_corrected = deviance / phi
    pvalue = chi2.sf(deviance_corrected, 1)
    if np.isnan(pvalue):
        pvalue = 1
    
    return pvalue

def fit_model_with_checking_perfect_separation(model):
    perfect_separation = False
    try:
        fit_result = model.fit(disp = 0)
    except PerfectSeparationError:
        perfect_separation = True
        fit_result = None
    return perfect_separation, fit_result

def make_dataframe_for_logit(treatment, beta, covariates):
    dict_for_dataframe = {
        "X" : treatment,
        "Y" : beta
    }
    dict_for_dataframe.update(covariates)
    return pd.DataFrame(dict_for_dataframe)

def calculate_overdispersion_correction_value_MN(data, weight, altresult):
    fitted_values = np.array(altresult.fittedvalues)
    weight = np.array(weight)
    num_param = len(altresult.params)
    y = data["Y"] * weight
    uresids = (y - weight*fitted_values)/np.sqrt(fitted_values*(weight - weight * fitted_values))
    phi = sum(uresids ** 2) / (len(weight) - num_param)
    if phi <= 1:
        phi = 1
    return phi

def get_mean_difference_between_two_groups(list_predict_value, list_beta, weights, control_case_value_for_prediction):
    df_for_mean_diff_calcul = pd.DataFrame({
        "treatment" : list(map(str, list_predict_value)),
        "beta" : list_beta,
        "weight" : weights
    })
    df_for_mean_diff_calcul["weighted_beta"] = df_for_mean_diff_calcul["beta"] * df_for_mean_diff_calcul["weight"]
    dict_treatment_to_weighted_beta_sum = dict()
    for trt in df_for_mean_diff_calcul["treatment"].unique():
        df_trt = df_for_mean_diff_calcul[df_for_mean_diff_calcul["treatment"] == trt]
        dict_treatment_to_weighted_beta_sum[trt] = df_trt["weighted_beta"].sum() / df_trt["weight"].sum()
    if len(df_for_mean_diff_calcul["treatment"].unique()) == 2:
        control_mean_methyl = dict_treatment_to_weighted_beta_sum[str(control_case_value_for_prediction[0])]
        case_mean_methyl = dict_treatment_to_weighted_beta_sum[str(control_case_value_for_prediction[1])]
    else:
        control_mean_methyl = min(dict_treatment_to_weighted_beta_sum.values())
        case_mean_methyl = max(dict_treatment_to_weighted_beta_sum.values())
    diff_mean_methyl = (case_mean_methyl - control_mean_methyl) * 100 # Make it as percentage
    return diff_mean_methyl
        
def save_dmp_result(list_logreg_results, path_save):
    columns = list(list_logreg_results[0].keys())
    table_dmp = pd.DataFrame(columns=columns)
    for col in columns:
        table_dmp[col] = list(map(lambda x : x[col], list_logreg_results))
    _, qvalue = fdrcorrection(table_dmp["pvalue"].to_list())
    table_dmp["qvalue"] = qvalue
    table_dmp.to_csv(path_save, sep = '\t', index = False)
    

#%%
if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--methyl", help = "Path to TSV formatted table of methylation values with depth. ()")
    argparser.add_argument("--meta", help = "Path to TSV formatted meta table of sample information")
    argparser.add_argument("--col_id", help = "Column name of Sample IDs from meta table")
    argparser.add_argument("--col_target", help = "Column name of 'Regression Target' variable from meta table")
    argparser.add_argument("--cols_position", help = "Column names of positional variable from methylation table", nargs = "+", default = list())
    argparser.add_argument("--col_cov", help = "Column names of covariates from meta table", nargs = "+", default = list())
    argparser.add_argument("--col_weight", help = "Column names of weights per sample from meta table", nargs = "+", default = list())
    argparser.add_argument("--n_job", help = "Number of processes to utilize", default = 1, type = int)
    argparser.add_argument("--output", help = "Path to save DMP analysis output (TSV format)")
    argparser.add_argument("--flag_noncategorical", help = "Flag for non-categorical prediction (Do not calculate mean methylation difference)", action = "store_true", default = False)
    argparser.add_argument("--value_control_case", help = "Value of control and case for comparing two categories [control, case]", nargs = 2)
    argparser.add_argument("--flag_wmean", help = "Flag for calculating weighted mean with 'col_weight' argument", action = "store_true", default=False)
    argparser.add_argument("--flag_nodepthweight", help = "Flag for not using 'sequencing depth' for weighting the sample", action = "store_true", default=False)
    args = argparser.parse_args()
    dict_args = vars(args)    
    
    main(dict_args["methyl"], dict_args["meta"], dict_args["col_id"], dict_args["col_target"], dict_args["cols_position"], dict_args["col_cov"], dict_args["col_weight"], dict_args["n_job"], dict_args["output"], is_predict_noncategorical = dict_args["flag_noncategorical"], control_case_value_for_prediction = dict_args["value_control_case"], get_weighted_mean = dict_args["flag_wmean"], no_depth_weighting = dict_args["flag_nodepthweight"])