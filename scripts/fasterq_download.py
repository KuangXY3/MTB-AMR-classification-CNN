import subprocess
import json
import os
import time
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter


def main():
    file_sra, out_dir = getArgs()
    # load a list of unique SRA accessions from a jason file, which you want
    # to download their fastq files
    sra_list = json.load(open(file_sra))
    start = time.time()
    for sra in sra_list:
        getfastq(sra, out_dir)
    end = time.time()
    print("Run Duration: {} seconds".format(end - start))


def getArgs():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        prog="fasterq_download.py",
        description="Execute fasterq-dump call for given list of SRAs.",
    )
    parser.add_argument("-f", "--fSRAs", dest="fileSRAs")

    parser.add_argument("-o", "--oDir", dest="outDir")
    args = parser.parse_args()
    f_sra = args.fileSRAs
    o_dir = args.outDir

    return f_sra, o_dir


def getfastq(sra, out_dir):
    # run fasterq-dump from sra toolkit to download fastq files
    cmd = [
        "/home/ubuntu/xk/sratoolkit.2.11.2-ubuntu64/bin/fasterq-dump",  # if you set sra toolkit's path to PATH, call fasterq-dump directly
        sra,
        "--threads",
        "16",
        "-O",
        out_dir,  # directory where you want to save the fastq files
    ]

    # when it is not the first run, it tries to download missing fastq files
    f_file1 = out_dir + "/" + sra + "_1.fastq"
    f_file2 = out_dir + "/" + sra + "_2.fastq"
    if not (os.path.isfile(f_file1) and os.path.isfile(f_file2)):
        print("\n--- API Call for {} ---\n".format(sra))
        with open("fastqDumpLog.txt", "a+") as f:
            subprocess.call(cmd, stdout=f)


if __name__ == "__main__":
    main()
