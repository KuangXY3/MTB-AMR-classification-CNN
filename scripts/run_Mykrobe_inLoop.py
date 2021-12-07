import os
import subprocess
from joblib import Parallel, delayed

# def main1(phyno):
#    text = open(phyno).read()
#    tmp = text.split('\n')
#    l=len(tmp)
#    for i in range(1,l):
#        item=tmp[i].split(',')
#        sra=item[0]
#        runMykrobe(sra)


def main():
    sra_list = loadAccessions()
    n_sra = len(sra_list)
    for i in range(0, n_sra):
        print(sra_list[i])
        print(i)
        runMykrobe(sra_list[i])


#    n_j=16
#    Parallel(n_jobs=n_j, prefer="threads")(delayed(runMykrobe)(sra) for sra in sra_list)


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


def runMykrobe(sra):
    fastq_dir = "fastqDump/"
    reads1 = fastq_dir + sra + "_1.fastq"
    reads2 = fastq_dir + sra + "_2.fastq"
    if os.path.isfile(reads1) and os.path.isfile(reads2):
        out_file = "mykrobeOut/result_" + sra + ".csv"
        if not (os.path.isfile(out_file)):
            cmd = [
                "mykrobe",
                "predict",
                sra,
                "tb",
                "--format",
                "csv",
                "-1",
                reads1,
                reads2,
                "--output",
                out_file,
            ]
            # print(cmd)

            subprocess.call(cmd)

    else:
        print("UGH! invalid path " + reads1 + " or " + reads2)


if __name__ == "__main__":
    main()
