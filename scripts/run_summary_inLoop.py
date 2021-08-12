import os
import subprocess

# out put summary files will be save in dir 'summary_output_full'
def run_summary():
    sra_list = loadAccessions()
    n_sra = len(sra_list)

    for j in range(0, n_sra):
        sra = sra_list[j]
        print(sra)
        cmd = [
            "ariba",
            "summary",
            "summary_output_full/" + sra + "_summary",
            "aribaResult_withBam/outRun_" + sra + "/report.tsv",
            "--preset",
            "all_no_filter",
        ]
        subprocess.call(cmd)


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


run_summary()
