# DNA-Methylation-Weighted-Suicide-Biomarkers

**Custom scripts and figure data for the paper;  .**

A Python-based script was developed by modifying the existing methylKit (R; Bioconductor) workflow to perform differential methylation analysis weighted by the number of suicide attempts

### Content

* [Analysis](#analysis)
  - [Preprocessing Methylation Data](#preprocessing-methylation-data)
  - [Merge CpG Values into Single Table](#merge-cpg-values-into-single-table)
  - [Run Weighted Differential Methylation Analysis](#run-weighted-differential-methylation-analysis)
* [Data and Figures in the Manuscript](#data-and-figures-in-the-manuscript)
  - [Data](#data)
  - [Figures / Tables](#figures--tables)

## Analysis
### Preprocessing Methylation Data

* Read trimming ([fastp](https://github.com/OpenGene/fastp))
```bash
fastp --thread {n_jobs} --in1 {path_read1} --in2 {path_read2} --out1 {path_trimmed_read1} --out2 {path_trimmed_read2} --json {path_json_report} --trim_front1 15 --trim_front2 15 --trim_tail1 5 --trim_tail2 5 --length_required 0 --cut_front 1 --cut_tail 1 --cut_right 0 --cut_mean_quality 20 --average_qual 20 --n_base_limit 1 --detect_adapter_for_pe
```

* Read Alignment ([Bismark](https://github.com/FelixKrueger/Bismark))
```bash
# This will generate *bismark_bt2_pe.bam
bismark -q --phred33-quals --dovetail -L 20 -N 1 --path_to_bowtie2 {path_to_bowtie2} -p {n_jobs} --output_dir {dir_output} -1 {path_trimmed_read1} -2 {path_trimmed_read2} --samtools_path {path_to_samtools} {dir_bismark_reference}
```

* Deduplicate Alignment ([Bismark](https://github.com/FelixKrueger/Bismark))
```bash
# This will generate *bismark_bt2_pe.deduplicated.bam
deduplicate_bismark --output_dir {dir_output} --samtools_path {path_to_samtools} {path_to_aligned_bam}
```

* Extract Methylation Value ([Bismark](https://github.com/FelixKrueger/Bismark))
```bash
# This will generate *bismark_bt2_pe.deduplicated.CpG_report.txt.gz
bismark_methylation_extractor --gzip --bedGraph --cytosine_report --parallel {n_jobs} --genome_folder {dir_bismark_reference} --output {dir_output} --samtools_path {path_to_samtools} {path_to_deduplicated_bam}
```

* Convert Bismark Report into Plain Text ([Custom script](Analysis_Scripts/Preprocessing/convert_bismark_cpg_report_into_plain_text.R))
```bash
Rscript ./Analysis_Scripts/Preprocessing/convert_bismark_cpg_report_into_plain_text.R {path_to_cpg_report} {path_to_save_plain_text}
```

* Merge CpG values from both strands ([Custom script](Analysis_Scripts/Preprocessing/merge_cpg_pairs.py))
```bash
python ./Analysis_Scripts/Preprocessing/merge_cpg_pairs.py --input {path_to_plain_text} --output {path_to_save_strand_merged_plain_text}
```

### Merge CpG Values into Single Table
* **Construction of a meta table listing the file paths of CpG value tables for each sample**

  - Example)

    | Sample ID | Path to CpG Plain Text |
    | --- | --- |
    | Sample 1 | ./path_strand_merged_cpg_values_sample1.tsv |
    | Sample 2 | ./path_strand_merged_cpg_values_sample2.tsv |
    | ... | ... |


* **Merge CpG value tables into R object ([MethylKit](https://bioconductor.org/packages/release/bioc/html/methylKit.html); [Custom script](Analysis_Scripts/Weighted_Differential_Methylation_Analysis/merge_cpg_tables_as_methylkit_object.R))**
```
usage: merge_cpg_tables_as_methylkit_object.R [-h] [--input INPUT]
                                              [--input_case INPUT_CASE]
                                              [--col_id COL_ID]
                                              [--col_path COL_PATH]
                                              [--output OUTPUT]
                                              [--cov_low_count_cutoff COV_LOW_COUNT_CUTOFF]
                                              [--unite_min_per_group UNITE_MIN_PER_GROUP]
                                              [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         Path to tsv file contains sample id and file path
                        (Consider as Control)
  --input_case INPUT_CASE
                        Path to case tsv file contains sample id and file path
                        (Optional)
  --col_id COL_ID       Column to sample ID
  --col_path COL_PATH   Column to methylation file path
  --output OUTPUT       Directory to save output
  --cov_low_count_cutoff COV_LOW_COUNT_CUTOFF
                        Cutoff of Lower Count Coverage [default : 10]
  --unite_min_per_group UNITE_MIN_PER_GROUP
                        Minimum number of samples per group to cover a
                        region/base [default : NULL (do inner join)]
  --threads THREADS     The number of threads [default : 50]
```
Example Usage)
```bash
Rscript ./Analysis_Scripts/Weighted_Differential_Methylation_Analysis/merge_cpg_tables_as_methylkit_object.R --input {path_metatable} --col_id {column_to_sample_id} --col_path {column_to_cpg_plain_text_file_path} --output {path_to_save_methylKit_object} --threads {n_jobs}
```

* **Save MethylKit object into Plain Text ([Custom script](Analysis_Scripts/Weighted_Differential_Methylation_Analysis/save_methylkit_object_as_tsv.R))**
```
usage: save_methylkit_object_as_tsv.R
       [-h] [--tsv TSV] [--RDS RDS] [--col_id COL_ID] [--col_path COL_PATH]
       [--cov_low_count_cutoff COV_LOW_COUNT_CUTOFF] [--output OUTPUT]
       [--threads THREADS]

optional arguments:
  -h, --help            show this help message and exit
  --tsv TSV             Path to tsv file contains id and file path
  --RDS RDS             Path to RDS compressed MethylBase file (Pre-united
                        methyl CpG Table)
  --col_id COL_ID       Column to sample ID
  --col_path COL_PATH   Column to methylation file path
  --cov_low_count_cutoff COV_LOW_COUNT_CUTOFF
                        Cutoff of Lower Count Coverage [default : 10]
  --output OUTPUT       Output path of result
  --threads THREADS     The number of threads
```

Example Usage)
```bash
Rscript ./Analysis_Scripts/Weighted_Differential_Methylation_Analysis/save_methylkit_object_as_tsv.R --RDS {path_to_merged_methylkit_object} --output {path_save_plain_text}
```

### Run Weighted Differential Methylation Analysis ([Custom script](Analysis_Scripts/Weighted_Differential_Methylation_Analysis/run_dmp_analysis_with_weights.py))
This code is a modification of the **_calculateDiffMeth_** function from [methylKit](https://bioconductor.org/packages/release/bioc/html/methylKit.html) (R; Bioconductor) to allow the assignment of an additional weight per sample in addition to sequencing depth. When no additional weight is applied, it produces results identical to the original methylKit output. Since sequencing depth is inherently incorporated into the weight, the code requires as input the plain text table generated by the previous custom R script, in which sequencing depth is included.

Additionally, the code can perform logistic regression on continuous variables (**--flag_noncategorical**), in addition to analyzing methylation differences between two groups.


```
usage: run_dmp_analysis_with_weights.py [-h] [--methyl METHYL] [--meta META]
                                        [--col_id COL_ID]
                                        [--col_target COL_TARGET]
                                        [--cols_position COLS_POSITION [COLS_POSITION ...]]
                                        [--col_cov COL_COV [COL_COV ...]]
                                        [--col_weight COL_WEIGHT [COL_WEIGHT ...]]
                                        [--n_job N_JOB] [--output OUTPUT]
                                        [--flag_noncategorical]
                                        [--value_control_case VALUE_CONTROL_CASE VALUE_CONTROL_CASE]
                                        [--flag_wmean] [--flag_nodepthweight]

optional arguments:
  -h, --help            show this help message and exit
  --methyl METHYL       Path to TSV formatted table of methylation values with
                        depth. ()
  --meta META           Path to TSV formatted meta table of sample information
  --col_id COL_ID       Column name of Sample IDs from meta table
  --col_target COL_TARGET
                        Column name of 'Regression Target' variable from meta
                        table
  --cols_position COLS_POSITION [COLS_POSITION ...]
                        Column names of positional variable from methylation
                        table
  --col_cov COL_COV [COL_COV ...]
                        Column names of covariates from meta table
  --col_weight COL_WEIGHT [COL_WEIGHT ...]
                        Column names of weights per sample from meta table
  --n_job N_JOB         Number of processes to utilize
  --output OUTPUT       Path to save DMP analysis output (TSV format)
  --flag_noncategorical
                        Flag for non-categorical prediction (Do not calculate
                        mean methylation difference)
  --value_control_case VALUE_CONTROL_CASE VALUE_CONTROL_CASE
                        Value of control and case for comparing two categories
                        [control, case]
  --flag_wmean          Flag for calculating weighted mean with 'col_weight'
                        argument
  --flag_nodepthweight  Flag for not using 'sequencing depth' for weighting
                        the sample
```
Example Usage (For Case vs Control))
```bash
python ./Analysis_Scripts/Weighted_Differential_Methylation_Analysis/run_dmp_analysis_with_weights.py \
--methyl {path_to_merged_cpg_plain_text_with_depth} \
--meta {path_to_TSV_formatted_metatable} \
--col_id {column_of_sample_id_on_metatable} \
--col_target {column_of_variable_of_interest_on_metatable} \
--cols_position chr start end \
--col_cov {columns_of_covariates_on_metatable} \
--col_weight {column_of_weight_variable_on_metatable} \
--n_job {n_jobs} \
--output {path_to_save_output_of_weighted_differential_methylation_analysis} \
--flag_wmean
--value_control_case {control_value_on_col_target} {case_value_on_col_target}
```
Example Usage (For Continuous Variable of Interest))
```bash
python ./Analysis_Scripts/Weighted_Differential_Methylation_Analysis/run_dmp_analysis_with_weights.py \
--methyl {path_to_merged_cpg_plain_text_with_depth} \
--meta {path_to_TSV_formatted_metatable} \
--col_id {column_of_sample_id_on_metatable} \
--col_target {column_of_variable_of_interest_on_metatable} \
--cols_position chr start end \
--col_cov {columns_of_covariates_on_metatable} \
--col_weight {column_of_weight_variable_on_metatable} \
--n_job {n_jobs} \
--output {path_to_save_output_of_weighted_differential_methylation_analysis} \
--flag_noncategorical
```

## Data and Figures in the Manuscript

### Data
The underlying data for generating the main figures, supplementary figures, and tables of the manuscript are stored in the [./Data](Data) directory.
This directory includes a table of CpG values for each of the 365 CpG site biomarkers across all samples included in the study. To obtain the complete dataset, please contact the authors to request access.

Meta data including sensitive information, such as the number of suicide attempts for each individual, has been removed from this directory.

### Figures / Tables
The original files for the figures and tables in the manuscript are uploaded in the [./Figures](Figures) and [./Table](Table) directories, respectively. 

The scripts used to generate these figures and tables are organized in the [./Figure_Scripts](Figure_Scripts) directory. For scripts that utilize sensitive data or very large datasets, the input data paths have been masked.