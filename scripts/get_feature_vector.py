"""
This scricpt is used to generate input feature vectors and label for training, 
evaluation and testing ML models for the 4 first-line TB drug resistance prediction
"""

import numpy as np
import csv
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

lineage = [
    "LAM",
    "Delhi",
    "Beijing",
    "EAI",
    "X-type",
    "Haarlem",
    "lineage4",
    "West African 2",
    "S-type",
    "Ural",
    "M. orygis",
    "M. bovis",
    "Cameroon",
    "Tur",
    "Uganda",
    "West African 1b",
    "BCG",
    "Ghana",
    "M. caprae",
    "M. microti",
]
firstLine_TB_4antibio = ["rifampicin", "ethambutol", "isoniazid", "pyrazinamide"]


def loadAccessions():
    """
    Loads in list of SRA accession numbers from finalSRAList.json
    there is no jason model in ariba docker
    """
    # sra_list = json.load(open("finalSRAList.json", "r"))
    sra_list = []
    text = open("uniqueSRA.json").read()
    tmp = text.split("\n")
    sra_list = [k[(k.index('"') + 1) : k.index('"', 8)] for k in tmp[1:-1]]
    return sra_list


def get_variable_names():
    """
    Select AMR associated know variants and gene presents, and novel variants that on coding regions  into
    feature vector for training and testing ML models,which are detected from at least one isolate
    """
    sra_list = loadAccessions()
    n_sra = len(sra_list)
    raw_feature = []

    for j in range(0, n_sra):
        sra = sra_list[j]
        # summary and report files are outputs of ariba, containing reference clusters that are matched by the sample
        # and information about the called variants and detected AMR associated genes respectively
        summary = "summary_output_full/" + sra + "_summary.csv"
        ariba_output = "aribaResult_withBam/outRun_" + sra + "/report.tsv"
        if os.path.isfile(ariba_output):
            df = pd.read_csv(ariba_output, sep="\t")
            df_summary = pd.read_csv(summary, sep=",")
            n_row = len(df)
            for i in range(0, n_row):
                # check if the sample matches the cluster presenting in ith line of report file
                if df_summary[df["cluster"][i] + ".match"][0] == "yes":
                    # Add gene present
                    if (
                        not (df["ref_name"][i] in raw_feature)
                        and df["known_var"][i] == "."
                    ):
                        raw_feature.extend([df["ref_name"][i]])
                    # Add novel variant that is located on a coding gene
                    if (
                        df["known_var"][i] == "0"
                        and df["gene"][i] == "1"
                        and not (
                            (df["ref_name"][i] + "." + df["ref_ctg_change"][i])
                            in raw_feature
                        )
                    ):
                        # cov% could be added in sumary. For insertion and deletion, there is no cov %
                        raw_feature.extend(
                            [df["ref_name"][i] + "." + df["ref_ctg_change"][i]]
                        )
                    # Add detected know variant
                    if (
                        df["known_var"][i] == "1"
                        and df["has_known_var"][i] == "1"
                        and not (
                            (df["ref_name"][i] + "." + df["known_var_change"][i])
                            in raw_feature
                        )
                    ):
                        raw_feature.extend(
                            [df["ref_name"][i] + "." + df["known_var_change"][i]]
                        )

    # when lineage is needed
    raw_feature.extend(lineage)
    np.savetxt("raw_fList.txt", raw_feature, fmt="%s")
    return raw_feature


def generate_sra_lineage_map(sourceF):
    """
    Generate a dictionary that map lineage info to corresponding isolates
    """
    df = pd.read_excel(sourceF, sheet_name="Sheet1")
    df.fillna("", inplace=True)
    sra_lineage = {}
    for i, j in df.iterrows():
        if j["SRA"] != "" and j["lineage"] != "":
            sra_lineage[j["SRA"]] = j["lineage"]
    return sra_lineage


def generate_featureVector_forOneIsoform(
    raw_features, summary_path, report_path, sra_acc, sra_lineage_dic
):
    """
    Create the feature vector for one sample by parsing the report file (report_pat) to 
    obtain the value for AMR associated variants and gene present, which are listed in 
    raw_feature, and adding corresponding lineage info.
    """
    ini_dic = {}
    f_vector = []
    feature_complete = 0
    for f in raw_features:
        ini_dic[f] = 0
    # loop the report file to change the value in the dic if match any key of the dic.
    df = pd.read_csv(report_path, sep="\t")
    df_summary = pd.read_csv(summary_path, sep=",")
    n_row = len(df)

    for i in range(0, n_row):
        if df_summary[df["cluster"][i] + ".match"][0] == "yes":
            if df["known_var"][i] == ".":
                ini_dic[df["ref_name"][i]] = 1

            if df["known_var"][i] == "0" and df["gene"][i] == "1":
                ini_dic[df["ref_name"][i] + "." + df["ref_ctg_change"][i]] = (1,)

            if df["known_var"][i] == "1" and df["has_known_var"][i] == "1":
                ini_dic[df["ref_name"][i] + "." + df["known_var_change"][i]] = 1

    ini_dic[sra_lineage_dic[sra_acc]] = 1

    for f in raw_features:
        f_vector.append(ini_dic[f])

    return f_vector


def generate_dic_nonGenFeature_label(nonGenFeature_label_file_path):
    """Save phenotype (label) and non genetic data into dic  phenotype_nonGenFeature"""
    col_add = [
        "sra_accession",
        "isolation_country",
        "genome_quality",
        "amikacin",
        "capreomycin",
        "ethambutol",
        "isoniazid",
        "kanamycin",
        "ofloxacin",
        "pyrazinamide",
        "rifampin",
        "rifampicin",
        "streptomycin",
    ]

    phenotype_nonGenFeature = {}
    with open(nonGenFeature_label_file_path, "r") as fh:
        for line in fh:
            line.rstrip()
            item = line.split("\t")
            single_sraDic = {}
            for i in range(1, len(col_add)):
                single_sraDic[col_add[i]] = item[i]

            phenotype_nonGenFeature[item[0]] = single_sraDic

    return phenotype_nonGenFeature


def generate_featureMatrics_labelList(
    raw_list, phenotype_nonGenFeature, sra_lineage_mapping_dic, drug
):
    """
    Loop all samples (over 10,000) to generate feature matrix in f_matrics wrote in featureM_X_'+antibio+'.txt',
    labels in y wrote in 'label_Y_'+antibio+'.txt'，and SRA in "sra_withFeature_"+drug+".txt", which will be input data
    for training ML models. These three files are corresponded based on the order of rows.
    They could be put in one file in next version.
    """
    sra_list = loadAccessions()
    n_sra = len(sra_list)
    f_matrics = []
    y = []
    f = open("sra_withFeature_" + drug + ".txt", "w")
    for j in range(0, n_sra):
        sra = sra_list[j]
        summary = "summary_output_full/" + sra + "_summary.csv"
        report = "aribaResult_withBam/outRun_" + sra + "/report.tsv"
        if (
            sra in phenotype_nonGenFeature.keys()
            and sra in sra_lineage_mapping_dic.keys()
        ):
            if os.path.isfile(report) and phenotype_nonGenFeature[sra][drug] != "":
                f.write(sra + "\n")
                single_fVector = generate_featureVector_forOneIsoform(
                    raw_list, summary, report, sra, sra_lineage_mapping_dic
                )
                f_matrics.append(single_fVector)
                # get lable list for rifampicin
                y.append(int(phenotype_nonGenFeature[sra][drug]))
    f.close()
    return (f_matrics, y)


raw_list = get_variable_names()

# Phenotype and lineage data are available in the supplementary file of the source paper
# https://www.nejm.org/doi/full/10.1056/nejmoa1800474.
# We organize phenotype data in 'phenotype.tsv' and lineage data in 'lineage.xls'
sra_lineage_map = generate_sra_lineage_map("lineage.xls")
phenotype_nonGenFeature = generate_dic_nonGenFeature_label("phenotype.tsv")

# Generate input data for training ML models for the 4 first-line TB drugs resistance prediction
for antibio in firstLine_TB_4antibio:
    f_matrics, y = generate_featureMatrics_labelList(
        raw_list, phenotype_nonGenFeature, sra_lineage_map, antibio
    )
    print(len(y))
    print(len(f_matrics))
    np.savetxt("featureM_X_" + antibio + ".txt", f_matrics, fmt="%d")
    np.savetxt("label_Y_" + antibio + ".txt", y, fmt="%d")
