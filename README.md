# Novel diagnostic and therapeutic options for KMT2A-rearranged acute leukemias

The KMT2A (MLL) gene rearrangements (KMT2A-r) are associated with a diverse spectrum of acute
leukemias. Although most KMT2A-r are restricted to nine partner genes, we have recently revealed
that KMT2A-USP2 fusions are often missed during FISH screening of these genetic alterations.
Therefore, complementary methods are important for appropriate detection of any KMT2A-r. Here we
use a machine learning model to unravel the most appropriate markers for prediction of KMT2A-r in
various types of acute leukemia.

---------------------------------------------------------------------------------------------------------

## Best Model Scripts

- KMT2Ar-preProcess&Boruta.ipynb

Variance Analysis and feature selection with Borutapy. 247 genes were considered 
important for predicting the KMT2A gene rearrangement.

- KMT2Ar-Development-247genes.py 

Development of the models using 247 genes selected by Borutapy.

- KMT2Ar-Model_Analysis-247Genes.ipynb

Analysis of the developed models (confusion matrix; metrics; ROC curve; Shap values).

---------------------------------------------------------------------------------------------------------

