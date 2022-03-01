# Pipline for developing machine learning (Random Forest, Logistic Regression, Deep CNN) models of mycobacterium tuberculosis (MTB) drug resistance prediction 

## Introduction

This pipeline automates the process of machine learning (ML) model development for MTB first-line drug (RIF, INH, PZA and EMB) resistance classification. It covers each step from DNA-seq data download to ML model validation. In addition, the last step is to evaluate a statistical rule-based method on the same datasets used to validate ML models.
Please refer to our paper [Accurate and rapid prediction of tuberculosis drug resistance from genome sequence data using traditional machine learning algorithms and CNN](https://www.nature.com/articles/s41598-022-06449-4) for more details.
The following is the guide for running this pipeline.
![alt text](https://github.com/KuangXY3/MTB-AMR-classification-CNN/blob/master/ML_model_development_flowchart.png)

## Data preparation

Assume the working directory is /mnt/MTB_AMR_Pre and the scripts and original input files needed for running this pipeline are under the working directory.

### Install [SRA-Toolkit](https://github.com/ncbi/sra-tools/wiki/02.-Installing-SRA-Toolkit) for downloading fastq files. ([SRA-Tools](https://github.com/ncbi/sra-tools) introduction)
### Download DNA-seq fastq files of isolates with SRA assessions listed in 'uniqueSRA.json' into directory 'fastqDump' (a sample uniqueSRA.json is in the folder [sample_input_files](https://github.com/KuangXY3/MTB-AMR-classification-CNN/tree/master/sample_input_files)):

    python fasterq_download.py -f uniqueSRA.json -o fastqDump 

### Run [Ariba](https://github.com/sanger-pathogens/ariba/blob/master/README.md#introduction) in docker:

    docker run --rm -it -v /mnt/MTB_AMR_Pre:/data  sangerpathogens/ariba  /bin/bash
    python -m pip install joblib
    
Run Ariba for isolates listed in 'uniqueSRA.json' to output report files of genetic data and intermediate results, e.g., bam, vcf, contig files, into output directory 'aribaResult_withBam'.  
Provide the directory of fastq files using -i.  
Give the number of threads you want to use using -n.

    cd  /data
    ariba getref card out.card
    python runAribaInLoop_withBam.py -f uniqueSRA.json -i fastqDump -o aribaResult_withBam -n 8 

### Get summary from Ariba result for isolates listed in 'uniqueSRA.json':

    python run_summary_inLoop.py

### Training-data-creation-for-traditional-ML-methods (put the [sample_input_files](https://github.com/KuangXY3/MTB-AMR-classification-CNN/tree/master/sample_input_files) phenotype.tsv and lineage.xls in the working directory)
Select AMR genes, known variants and novel variants on coding regions that are detected on at least one sample as genetic features, and add 20 lineages together as the input feature set.  
Generate files of feature matrices, labels and SRA accessions in the same sample order for each drug based on phenotype and lineage availability.

    python get_feature_vector.py

## Traditional ML
### Random Forest and Logistic Regression 
Read features and labels from the output files of [last step](#Training-data-creation-for-traditional-ML-methods).  
Output multiple metrics (e.g. f-measure, sensitivity, specificity) to evaluate RF and LR models (10-fold CV).

    python RF_LR_validation_multiMetricCalculated.py

## Multi-input 1D CNN 
### Feature selection: 
Use 80% of samples  to get the importance score for each of the features from the [previous step](#Training-data-creation-for-traditional-ML-methods), 20% for validation to find the best feature importance cutoff that maximizes F score.

    python select_important_feaures.py  > feature_selection_tunning_output.txt

### Build and validate 1D CNN models.
As multi-inputs of the first layer, variant features are converted to normalized base counts of fixed length (21) of DNA fragments centered at focal variants' loci.  
Build our 1D CNN architecture.  
Train and test 4 models for the 4 first-line TB drugs, respectively.

    generateInput4Conv1D_withMultiInput_N_createCNN_trainNtest_on4drugs.py

Add coverage as an additional feature.

    generateInput4Conv1D_withMultiInput_N_createCNN_trainNtest_on4drugs_withCoverage.py


## Evaluate a rule-based method Mykobe

### Run [Mykrobe](https://github.com/Mykrobe-tools/mykrobe) for isolates listed in 'uniqueSRA.json' in docker to get its predictions:

    docker run --rm -it -v /mnt/MTB_AMR_Pre:/mnt  phelimb/mykrobe_predictor /bin/bash
    python -m pip install joblib
    python run_Mykrobe_inLoop.py

### Evaluate performance of a simple Ariba-based method and Mykrobe on 4 sets of data that were also used to train and test the ML models for the 4 drugs, respectively:

    AMR_prediction_validation_Ariba_Mykrobe.py

## Citation
### If you use code or idea here, please cite:
Kuang, X., Wang, F., Hernandez, K. M., Zhang, Z., & Grossman, R. L. (2022). [Accurate and rapid prediction of tuberculosis drug resistance from genome sequence data using traditional machine learning algorithms and CNN.](https://www.nature.com/articles/s41598-022-06449-4) Scientific Reports, 12(1), 1-10.