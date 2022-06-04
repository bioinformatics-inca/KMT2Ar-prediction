#"""
#@authors: LBBC team (https://sites.google.com/view/bioinformaticainca-en/home-en)
#Cristiane Esteves
#"""


load("Dataset_ML_KMT2Ar.RData")

table(df_final$Study)
df_final$Study = NULL
df_final$Age.years = NULL

table(df_final$Leukemia)
table(df_final$Age_group)

head(df_final[,1:6])

# CLEANING AND PRE-PROCESSING
library(caret)


#Remove highly correlated variables
descrCor <-  cor(df_final[5:ncol(df_final)])
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .90)
df_leukemia <- df_final[,-highlyCorDescr]
#14834 genes
rm(descrCor,highCorr,highlyCorDescr)

write.csv(df_leukemia, "sb_leuk.csv")
