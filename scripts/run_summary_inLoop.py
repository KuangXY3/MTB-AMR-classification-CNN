import os
import subprocess
import json

# out put summary files will be save in dir 'summary_output_full'
def run_summary():
    sra_list = json.load(open("uniqueSRA.json"))
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


run_summary()
