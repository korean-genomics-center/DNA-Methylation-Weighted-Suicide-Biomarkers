{
    #Install required packages and load packages
    print("Install Required Packages")
    installed_packages = installed.packages()[,"Package"]
    required_packages = c("BiocManager", "stringr", "data.table")
    Bioc_required_packages = c("methylKit")
    for (req_package in required_packages){
        # if(! req_package %in% installed_packages){
        #     print(paste0("Installing ", req_package))
        #     suppressMessages(install.packages(req_package, repos="https://cran.biodisk.org/", quiet = TRUE))
        # }
        suppressMessages(library(req_package, character.only=TRUE))
        print(paste0("Load ", req_package))
    }

    #Update packages
    # print("Updating Packages")
    # suppressMessages(update.packages(ask=FALSE, checkBuilt=TRUE, repos="https://cran.biodisk.org/"))
    # suppressMessages(BiocManager::install(version = '3.14', ask = FALSE))

    print("Install BiocManager Packages")
    for (req_bioc_package in Bioc_required_packages){
        # if(! req_bioc_package %in% installed_packages){
        #     print(paste0("Installing ", req_bioc_package))
        #     BiocManager::install(req_bioc_package, version = '3.14', ask = FALSE, update = TRUE)
        # }
        suppressMessages(library(req_bioc_package, character.only=TRUE))
        print(paste0("Load ", req_bioc_package))
    }
}
{# fin and fout : fin for the cpg file. out for methylkit format output
    args<-commandArgs(TRUE)

fin<-args[1]  # group name 1 to compare
fout<-args[2]
}
{ #Open methylation file in bismark cytosine report (cpg_report) format
  myobj=methRead(fin,
                 sample.id="None",
                 assembly="hg38",
                 pipeline="bismarkCytosineReport",
                 treatment=0,
                 context="CpG",
                 mincov = 1)
}
{
  myobj$freqC= myobj$numCs/ myobj$coverage * 100

  df<-getData(myobj)[c('chr','start','strand','coverage','freqC')]
  fwrite(df, file=fout,row.names=FALSE, sep="\t",quote=FALSE)
}