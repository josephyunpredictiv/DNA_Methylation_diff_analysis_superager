#Ref: https://nbis-workshop-epigenomics.readthedocs.io/en/latest/content/tutorials/methylationArray/Array_Tutorial.html
#analyzing 450K methylation array
#Ref2: minfi: https://bioconductor.org/packages/release/bioc/vignettes/minfi/inst/doc/minfi.html
#Ref3: minfiData: https://bioconductor.org/packages/3.18/data/experiment/html/minfiData.html 
#Try RnBeads2.0

#rm(list = ls())

#dir.create("XX/R_libraries", showWarnings = FALSE, recursive = TRUE)
#.libPaths(c("XX/R_libraries", .libPaths()))


#Install1: The main package - RnBeads 
#if (!requireNamespace("BiocManager", quietly=TRUE))
#  install.packages("BiocManager")
#BiocManager::install("RnBeads", lib="XX/R_libraries")
#print("RnBeads installed")
#BiocManager::install("RnBeads.hg19", lib="XX/R_libraries")
#print("RnBeads.hg19 installed")

#Install2: hg19 annotation for RnBeads
#BiocManager::install("FDb.InfiniumMethylation.hg19", lib="XX/R_libraries", force=TRUE)
#print("FDb Infinium Methylation hg 19 installed")

#Install3: Dependency for RnBeads for Report pdf generation
#Install Ghostscript Mac from https://pages.uoregon.edu/koch/

#You may install RnBeads from source if the above installation does not work
source("http://rnbeads.org/data/install.R")


###################################################
### Parameter setting
###################################################
library(RnBeads)
# Directory where your data is located
data.dir <- "XX"
idat.dir <- file.path(data.dir, "/ADNI_iDAT_files/")
sample.annotation <- file.path(data.dir, "/analysis1_06_08_2024/sample_annotation.csv")
# Directory where the output should be written to
analysis.dir <- "XX/analysis1_06_08_2024"
# Directory where the report files should be written to
# The reports directory should not exist
report.dir <- file.path(analysis.dir, "reports")


###################################################
### Do not consider the sex chromosomes
###################################################
rnb.options(filtering.sex.chromosomes.removal=TRUE, identifiers.column="Sample_ID", assembly="hg19")


###################################################
### Main analysis
###################################################
rnb.run.analysis(dir.reports=report.dir, sample.sheet=sample.annotation, data.dir=idat.dir, data.type="infinium.idat.dir")
