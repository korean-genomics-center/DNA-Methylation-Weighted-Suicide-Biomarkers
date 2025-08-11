{
    #Install required packages and load packages
    print("Load Required Packages")
    installed_packages = installed.packages()[,"Package"]
    required_packages = c("BiocManager", "readr", "stringr", "data.table", "argparse")
    Bioc_required_packages = c("methylKit", "BiocParallel")
    for (req_package in required_packages){
        suppressMessages(library(req_package, character.only=TRUE))
        print(paste0("Load ", req_package))
    }

    print("Load BiocManager Packages")
    for (req_bioc_package in Bioc_required_packages){
        suppressMessages(library(req_bioc_package, character.only=TRUE))
        print(paste0("Load ", req_bioc_package))
    }
}

### parse arguments
{
    parser <- ArgumentParser()
    parser$add_argument("--input", help = "Path to tsv file contains sample id and file path (Consider as Control)")
    parser$add_argument("--input_case", required = FALSE, help = "Path to case tsv file contains sample id and file path (Optional)")
    parser$add_argument("--col_id", help = "Column to sample ID")
    parser$add_argument("--col_path", help = "Column to methylation file path")
    parser$add_argument("--output", help="Directory to save output")
    parser$add_argument("--cov_low_count_cutoff", help="Cutoff of Lower Count Coverage [default : 10]", required = FALSE, type = 'integer', default = 10)
    parser$add_argument("--unite_min_per_group", help="Minimum number of samples per group to cover a region/base [default : NULL (do inner join)]", required = FALSE, type = 'integer')
    parser$add_argument("--threads", help="The number of threads [default : 50]", type = 'integer', default = 50)
    args <- parser$parse_args();
}

read_tsv_custom <- function(path){
    table <- read_tsv(path, show_col_types = FALSE);
    return(table);
}

{
    numcore <- as.integer(args$threads);
    register(MulticoreParam(numcore));

    table <- read_tsv_custom(args$input)
    
    if(is.null(args$input_case)){
        total_ids <- table[[args$col_id]]
        total_paths <- table[[args$col_path]]
        sample_type <- c(rep(0,nrow(table)))
    }
    else{
        table_case <- read_tsv_custom(args$input_case)
        total_ids <- c(table_case[[args$col_id]], table[[args$col_id]])
        total_paths <- c(table_case[[args$col_path]], table[[args$col_path]])
        sample_type <- c(rep(1,nrow(table_case)),rep(0,nrow(table)))
    }
    print(total_ids)
    print(total_paths)
    print(sample_type)
}


{
    methyl_obj <- methRead(as.list(total_paths), sample.id =as.list(total_ids), assembly = "hg38",treatment = sample_type, context = "CpG", pipeline=list(fraction=FALSE, chr.col=1, start.col=2, end.col=2, coverage.col=4, strand.col=3, freqC.col=5), mincov = args$cov_low_count_cutoff)
    methyl_obj_coverage_filtered <- filterByCoverage(methyl_obj, lo.count = args$cov_low_count_cutoff, lo.perc = NULL, hi.count = NULL, hi.perc = 99.9)
    methyl_obj <- NULL
    methyl_united <- unite(methyl_obj_coverage_filtered, mc.cores = numcore, min.per.group = args$unite_min_per_group)
    print("Saving RDS formatted methylation table...")
    saveRDS(methyl_united, args$output, compress = TRUE)
}