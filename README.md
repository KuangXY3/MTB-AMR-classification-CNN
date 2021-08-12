# Pipline for building and validating machine learning (ML) models of mycobacterium tuberculosis (MTB) anti-TB drug resistance prediction 

## Introduction

The pipeline consists of the steps below.
Please refer to our paper [Accurate and rapid prediction of first-line tuberculosis drug resistance from genome sequence data using traditional machine learning algorithms and CNN ]() for more details.

## Data preparation

Assume the working directory is /mnt/MTB_AMR_Pre and origianl input files needed for running the scripts 
are under the working directory.

### Download DNA-seq fastq files of isolates with SRA assessions listed in 'uniqueSRA.json' into directory 'fastqDump':

    python fasterq_download.py -f uniqueSRA.json -o fastqDump 

### Run [Ariba](https://github.com/sanger-pathogens/ariba/blob/master/README.md#introduction) in docker:

    docker run --rm -it -v /mnt/MTB_AMR_Pre:/data  sangerpathogens/ariba  /bin/bash
    python -m pip install joblib

Run Ariba for isolates listed in 'uniqueSRA.json' to output report files of genetic data and intermediate result, e.g., bam, vcf, contig files, into output directory aribaResult_withBam.
Provid the directory that the fastq files are located after -i.
Give number of threads you want to use after -n.

    cd  /data
    python runAribaInLoop_withBam.py -f uniqueSRA.json -i fastqDump -o aribaResult_withBam -n 8 

### Get summary from Ariba result for isolates listed in 'uniqueSRA.json':

    python run_summary_inLoop.py

### Generate input feature matrix and labels for traditional ML methods:
Select AMR genes, known variants and novel varaints on coding region that are detected on at least one sample as genetic feature, and plus 20 lineages as input feature set.
Generate files of feature matrices, labels and SRA accessions in same sample order for each drug based on phenotype and lineage availability.

    python get_feature_vector.py

## Traditional ML
### Random Forest and Logistic Regression     
Use 10-fold CV for validation:
Output multiple metrices (e.g. f-measure, sensitivity, specifivity) to evaluate RF and LR models

    python RF_LR_validation_multiMetricCalculated.py


## Multi-input 1D CNN 

### Feature selection: 
Use 80% of samples  to get importance score for each of the features from the [previous step](#Generate), 20% for validation to find best feature importance cutoff that maximizes F score.

    python select_important_feaures.py  > feature_selection_tunning_output.txt

### Variant features are converted to normalized base counts of fixed length (21) of DNA fragments 
### centered at focal variants' loci, which were selected by RF feauture selecttion tunning.
### This architecture is validated by using 10_fold CV.          

Prepare input features and labels into the format for our 1D CNN architecture;
Buid CNN model of our architecture
Train and test 4 models for the 4 first-line TB drugs, respectively.

    generateInput4Conv1D_withMultiInput_N_createCNN_trainNtest_on4drugs.py

Add coverage as an additional feature.

    generateInput4Conv1D_withMultiInput_N_createCNN_trainNtest_on4drugs_withCoverage.py


## Evaluate a rule-based method Mykobe

### Run [Mykrobe](https://github.com/Mykrobe-tools/mykrobe) for isolates listed in 'uniqueSRA.json' in docker to get its predictions:

    docker run --rm -it -v /mnt/MTB_AMR_Pre:/mnt  phelimb/mykrobe_predictor /bin/bash
    python -m pip install joblib
    python run_Mykrobe_inLoop.py

### Evaluate performance of a simple ariba-based method and Mykrobe on 4 sets of data that were also used to train and test the ML models for the 4 drugs, respectively:

    AMR_prediction_validation_Ariba_Mykrobe.py

## Citation
### If you use code or idea here please cite:
Paper sounce once published