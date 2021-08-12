# Pipline for building and validating machine learning models of mycobacterium tuberculosis (MTB) anti-TB drug resistance prediction 

## Introduction


The pipeline consists of the steps below.

## Data preparation

Assume the working directory is /mnt/MTB_AMR_Pre and all input files needed for running the scripts 
are under the working directory.

### Download fasta pairs of isolates:
Put uniqueSRA.json (the list of SRA assessions you want to download) in working directory
Download DNA-seq fastq files with SRA assessions listed in 'uniqueSRA.json' to directory 'fastqDump'

    python fasterq_download.py -f uniqueSRA.json -o fastqDump > fasterq.log 

Run [Ariba](https://github.com/sanger-pathogens/ariba/blob/master/README.md#introduction) in docker:

    docker run --rm -it -v /mnt/MTB_AMR_Pre:/data  sangerpathogens/ariba  /bin/bash
    python -m pip install joblib

### Run Ariba for isolates listed in 'uniqueSRA.json' to output variant report file and intermediate result, e.g., bam, vcf, contig files, into output directory aribaResult_withBam:
Provid the directory that the fastq files are located after -i
Give number of threads you want to use after -n

    cd  /data
    python runAribaInLoop_withBam.py -f uniqueSRA.json -i fastqDump -o aribaResult_withBam -n 8 > runAriba_withBam.log

### Get summary from Ariba result for all isolates:

    python run_summary_inLoop.py

### Run [Mykrobe](https://github.com/Mykrobe-tools/mykrobe) on all isolates in docker to evaluate its performance on the 4 sets of data for the 4 drugs, respectively:

    docker run --rm -it -v /mnt/MTB_AMR_Pre:/mnt  phelimb/mykrobe_predictor /bin/bash
    python -m pip install joblib
    python run_Mykrobe_inLoop.py

### Prepare input feature matrix and labels for traditional machine learning:
Select AMR genes, known variants and novel varaints on coding region that are detected on at least one sample, and plus 20
lineages as input feature set
Generate files of feature matrices, labels and SRA accessions in same sample order for each drug based on phenotype and lineage availability

    python get_feature_vector.py


## Evaluate rule-based methods

### Evaluate performance of a simple ariba-based method and Mykrobe on 4 sets of data which will be used to train and test 4 ML models for the 4 drugs, respectively:

    AMR_prediction_validation_Ariba_Mykrobe.py


## Traditional ML
## Random Forest and Logistic Regression     

### Feature selection: 
Use 80% of samples  to get importance score for each of the 283 features using RF feature importance algorithm, 20% for validation to find best feature importance cutoff that maximizes F score

    python select_important_feaures.py  > feature_selection_tunning_output.txt

### Training and testing by using 10-fold CV:
Output multiple metrices (e.g. f-measure, sensitivity, specifivity) to evaluate RF and LR models

    python RF_LR_validation_multiMetricCalculated.py


## Multi-input 1D CNN 

### Variant features are converted to normalized base counts of fixed length (21) of DNA fragments 
### centered at focal variants' loci, which were selected by RF feauture selecttion tunning.
### This architecture is validated by using 10_fold CV.          

Prepare input features and labels into the format for our 1D CNN architecture;
Buid CNN model of our architecture
Train and test 4 models for the 4 first-line TB drugs, respectively.

    generateInput4Conv1D_withMultiInput_N_createCNN_trainNtest_on4drugs.py

Add coverage as an additional feature.

    generateInput4Conv1D_withMultiInput_N_createCNN_trainNtest_on4drugs_withCoverage.py


## Citation
### If you use code or idea here please cite:
### Paper sounce once published