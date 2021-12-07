import os
import subprocess
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
from joblib import Parallel, delayed


def main():
    file_sra, in_dir, out_dir, n_j = getArgs()
    sra_list = loadAccessions(file_sra)
    Parallel(n_jobs=n_j, prefer="threads")(
        delayed(runAriba)(sra, in_dir, out_dir) for sra in sra_list
    )


def loadAccessions(file_sra):
    """
    Loads in list of SRA accession numbers from file_sra
    There is no jason model in ariba docker, otherwise, we can use:
    sra_list = json.load(open(file_sra, "r"))
    """

    sra_list = []
    text = open(file_sra).read()
    tmp = text.split("\n")
    sra_list = [k[(k.index('"') + 1) : k.index('"', 8)] for k in tmp[1:-1]]
    return sra_list


def getArgs():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        prog="runAribaInLoop_withBam.py",
        description="Run Ariba for isolates to output variant report files and intermediate results",
    )
    parser.add_argument("-f", "--fSRAs", dest="fileSRAs")

    parser.add_argument("-i", "--iDir", dest="inDir")

    parser.add_argument("-o", "--oDir", dest="outDir")
    parser.add_argument("-n", "--n_jobs", dest="nJobs")
    args = parser.parse_args()
    f_sra = args.fileSRAs
    i_dir = args.inDir
    o_dir = args.outDir
    n_job = args.nJobs

    return f_sra, i_dir, o_dir, n_job


def runAriba(sra, in_dir, out_dir):
    # print (sra)
    fastq_dir = in_dir + "/"
    reads1 = fastq_dir + sra + "_1.fastq"
    reads2 = fastq_dir + sra + "_2.fastq"
    if os.path.isfile(reads1) and os.path.isfile(reads2):
        out_dir = out_dir + "/outRun_" + sra
        if not (os.path.isfile(out_dir + "/report.tsv")):
            if os.path.isdir(out_dir):
                subprocess.run(["rm", "-r", out_dir])
            cmd = [
                "ariba",
                "run",
                "--noclean",
                "out.card.prepareref",
                reads1,
                reads2,
                out_dir,
            ]
            with open("./aribaRunLog.txt", "a+") as f:
                subprocess.call(cmd, stdout=f)
    else:
        print("UGH! invalid path " + reads1 + " or " + reads2)
        with open("./sra_paired_read_notFound.txt", "a+") as l:
            l.write(sra + "\n")


if __name__ == "__main__":
    main()
