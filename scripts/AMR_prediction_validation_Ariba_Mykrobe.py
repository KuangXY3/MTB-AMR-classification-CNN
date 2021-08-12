import pandas as pd
import os

# res2Drug={"ethambutol":'conferring resistance to ethambutol',"isoniazid":'conferring resistance to isoniazid',"pyrazinamide":'conferring resistance to pyrazinamide',"rifampicin":'conferring resistance to rifampicin'}
res2Drug = {
    "ethambutol": "ethambutol",
    "isoniazid": "isoniazid",
    "pyrazinamide": "pyrazinamide",
    "rifampicin": "rifampicin",
}

# function for loop prediction
# get prediction for drug named antibio on isolates listed in sra_l
def get_ariba_prediction(sra_l, antibio):

    # n_sra=len(df_phyno)
    ariba_pre = []

    for sra in sra_l:
        # get Ariba prediction into df_pre
        summary = "summary_output_full/" + sra + "_summary.csv"
        ariba_output = "aribaResult_withBam/outRun_" + sra + "/report.tsv"
        df = pd.read_csv(ariba_output, sep="\t")
        df_summary = pd.read_csv(summary, sep=",")
        n_row = len(df)
        dic1 = {}
        dic1["SRA"] = sra
        temp = res2Drug.copy()
        for i in range(0, n_row):
            for drug, drugResis in temp.items():
                # if this is nonsyn variant or gene present conferring  resistance to a drug, predicted AMR will be R for the drug
                if (
                    df["has_known_var"][i] == "1"
                    and df["ref_ctg_effect"][i] == "NONSYN"
                    and drugResis in df["var_description"][i]
                ) or (
                    df["gene"][i] == "1"
                    and df["var_only"][i] == "0"
                    and drugResis in df["free_text"][i]
                    and df_summary[df["cluster"][i] + ".match"][0] == "yes"
                ):
                    dic1[drug] = "R"
                    # assume one row in report won't show AMR for multiple drug
                    del temp[drug]
        for drug, drugResis in temp.items():
            dic1[drug] = "S"

        ariba_pre.append(dic1[antibio])

    return ariba_pre


# get Mykrobe prediction in df_pre
# get prediction for drug named antibio on isoforms listed in sra_l
def get_mykrobe_prediction(sra_l, antibio):

    mykrobe_pre = []
    for sra in sra_l:
        my_out = "mykrobeOut/result_" + sra + ".csv"
        if os.path.isfile(my_out):
            df = pd.read_csv(my_out, sep=",")
            n_row = len(df)
            for i in range(0, n_row):
                if df["drug"][i].lower() == antibio:
                    mykrobe_pre.append(df["susceptibility"][i])
        else:
            mykrobe_pre.append("S")

    return mykrobe_pre


def prediction_validation(antibio_drug):
    # list of isolate IDs for the 4 drugs saved in 4 files
    text = open("sra_withFeature_" + antibio_drug + ".txt").read()
    sra_list = text.split("\n")
    if sra_list[-1] == "":
        del sra_list[-1]
    text = open("label_Y_" + antibio_drug + ".txt").read()
    pheno = text.split("\n")
    if pheno[-1] == "":
        del pheno[-1]

    n_sra = len(pheno)
    if n_sra != len(sra_list):
        print("Inconsistant lengthes for pheno and sample sra")
    else:
        print("Consistant lengthes for pheno and sample sra")
    a_pre = get_ariba_prediction(sra_list, antibio_drug)
    m_pre = get_mykrobe_prediction(sra_list, antibio_drug)

    # df_phyno.to_csv('drug_resistance_phynotype.tsv',index=False,sep="\t",header=True)
    # df_pre.to_csv('ariba_prediction.tsv',index=False,sep="\t",header=True)
    # df_mykrobe_pre.to_csv('mykrobe_prediction.tsv',index=False,sep="\t",header=True)

    # compare pheno with prediction

    dic_pre = {"ariba": a_pre, "mykrobe": m_pre}
    for method in dic_pre:
        print(method)

        fp = 0
        fn = 0
        tp = 0
        tn = 0
        n_na = 0
        match = 0
        m_list = []
        for i_sra in range(0, n_sra):
            if dic_pre[method][i_sra] == "S" and pheno[i_sra] == "1":
                tp += 1
                match = 1
            elif dic_pre[method][i_sra] == "R" and pheno[i_sra] == "0":
                tn += 1
                match = 1
            elif dic_pre[method][i_sra] == "S" and pheno[i_sra] == "0":
                fp += 1
                match = 0
            elif dic_pre[method][i_sra] == "R" and pheno[i_sra] == "1":
                fn += 1
                match = 0
            m_list.append(match)

        print("tp:", tp, ",", "tn:", tn, ",", "fp:", fp, ",", "fn:", fn)
        print("Accuracy:" + str((tp + tn) / float(tp + tn + fp + fn)))
        print("Specificity:" + str(tn / float(tn + fp)))
        print("Sensitivity:" + str(tp / float(tp + fn)))
        Precision = tp / float(tp + fp)
        print("Precision:" + str(tp / float(tp + fp)))
        Recall = tp / float(tp + fn)
        print("F-Measure:" + str(2 * (Recall * Precision) / (Recall + Precision)))

    # df_match.to_csv(method+'_match.tsv',index=False,sep="\t",header=True)


def loop_validation_forDrugs():
    for key in res2Drug:
        print(key)
        prediction_validation(key)


loop_validation_forDrugs()
