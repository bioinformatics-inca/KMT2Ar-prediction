#"""
#@authors: LBBC team (https://sites.google.com/view/bioinformaticainca-en/home-en)

#"""

# load ML dataset
load("Dataset_ML_KMT2Ar.RData")

# log transformation of FPKM values
df_final_matrix <- log10(df_final[,c(7:ncol(df_final))])

# id patients in rownames
rownames(df_final_matrix) <- df_final$patient

# transpose matrix
df_final_matrix <- t(df_final_matrix)

# remove batch effect - by data study
library(limma)
df_final_matrix <- removeBatchEffect(df_final_matrix, batch = df_final$Study)

# return to FPKM values
df_final_matrix <- t(df_final_matrix)
df_final_matrix <- 10^df_final_matrix

# update values in initial dataset
df_final[,c(7:ncol(df_final))] <- df_final_matrix

# save ML dataset
save(df_final, file = "Dataset_ML_KMT2Ar.RData")
