#Ref: https://nbis-workshop-epigenomics.readthedocs.io/en/latest/content/tutorials/methylationArray/Array_Tutorial.html
#analyzing 450K methylation array
#Ref2: minfi: https://bioconductor.org/packages/release/bioc/vignettes/minfi/inst/doc/minfi.html
#Ref3: minfiData: https://bioconductor.org/packages/3.18/data/experiment/html/minfiData.html 
#Try RnBeads2.0
#Ref4: https://groups.google.com/g/epigenomicsforum/c/e_I0pFzYAF8. Fixing ff read error

rm(list = ls())

library(RnBeads)
library(ff)
library(ffbase)
library(dplyr)

setwd("/Users/josephyun/Desktop/AI/Data_Analysis/superager/analysis1_06_08_2024")
rnb.set <- load.rnb.set("/Users/josephyun/Desktop/AI/Data_Analysis/superager/analysis1_06_08_2024/reports/rnbSet_preprocessed/rnb.set.RData")
summarized.regions(rnb.set)
full_df <- mval(rnb.set, type="promoters", row.names=TRUE) 
dir.create("data")
setwd("data")
write.csv(head(full_df), "small_promoter_M_value.csv")
View(head(full_df))
dim(full_df)
#write.csv(full_df, "full_promoter_M_value.csv")

#full_df<- read.csv("full_promoter_M_value.csv")
results <- read.csv("diffMethTable_region_cmp9_promoters.csv")
sorted_results <- results[order(results$comb.p.val), ]
filtered_results <- subset(sorted_results, comb.p.val <= 0.05 & !is.na(symbol))
#View(filtered_results)
dim(filtered_results)
write.csv(filtered_results, "promoter_results_filtered_by_pvalue_0.05.csv")

row_names <- sorted_results %>% subset(!is.na(symbol)) %>% select(c("id", "symbol"))
#full_df$id <- rownames(full_df)
result_df <- full_df %>% left_join(row_names, by = c("X" = "id"))
result_df <- result_df %>% subset(!is.na(symbol))
result_df <- result_df %>% select(symbol, X, everything())
write.csv(result_df, "full_promoter_M_value.csv")

filter_df <- filtered_results %>% select("symbol")
filtered_result_df <- result_df %>% semi_join(filter_df, by = "symbol")
write.csv(filtered_result_df, "promoter_M_values_filtered_by_pvalue_0.05.csv")

filtered_result_df <- filtered_result_df %>% select(-X)
transposed_df <- as.data.frame(t(filtered_result_df %>% select(-symbol)))
colnames(transposed_df) <- filtered_result_df$symbol
transposed_df$group <- ifelse(substr(rownames(transposed_df), 1, 1) == "s", "s", "t")
write.csv(transposed_df, "promoter_M_values_filtered_by_pvalue_0.05_transposed.csv")


