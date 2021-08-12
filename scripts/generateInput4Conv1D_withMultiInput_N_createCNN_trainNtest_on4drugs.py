""" Based on the number of variants selected by Random Forest feature selection, we generate the same number of
inputs for CNN layers.  Then concatenate flattened output of all the CNN layers and a input layer with gene 
presents and lineage info as a vector. Each input for each Conv1D layer is a 21x4 matrix (use 21 bases window centered by
the variant locus; normalized counts of the 4 bases on each locus based alignment) first part of this script outputs 2 
numpy array with shape (No. of variants, No. of samples,21,4), one for training (80%), one for testing (20%). """
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Conv1D,
    Dropout,
    MaxPooling1D,
    Input,
    Dense,
    Flatten,
    concatenate,
)
from tensorflow.keras.models import Model
import random

# from scipy import stats


class baseHparamsNvars(object):
    """ 
    Provide default hyperparameter
    Feature names are hard coded here, it can be automatically extracted from the output of feature selection step in next version
    """

    def __init__(
        self,
        total_epochs=100,
        learning_rate=0.004,
        l2=0.001,
        batch_size=256,
        window_size=21,  # should be positive odd number
        lineageNgenePresent={
            "rifampicin": [
                "Delhi",
                "Beijing",
                "Haarlem",
                "lineage4",
                "AAC_2___Ic.3002525.AL123456.3.314308_314854.4719",
                "Erm_37_.3000392.AL123456.2231679_2232219.582",
                "efpA.3003955.NC_000962.3.3153038_3154631.4603",
                "mfpA.3003035.AL123456.3773015_3773567.4588",
            ],
            "ethambutol": [
                "AAC_2___Ic.3002525.AL123456.3.314308_314854.4719",
                "Erm_37_.3000392.AL123456.2231679_2232219.582",
                "efpA.3003955.NC_000962.3.3153038_3154631.4603",
                "mfpA.3003035.AL123456.3773015_3773567.4588",
                "TEM_116.3000979.NC_002156.1429_2290.2030",
                "murA.3003784.HE608151.1472985_1474242.3555",
                "LAM",
                "Delhi",
                "Beijing",
                "EAI",
                "X-type",
                "Haarlem",
                "lineage4",
                "S-type",
                "Ural",
                "Cameroon",
                "Tur",
                "Ghana",
            ],
            "isoniazid": [
                "AAC_2___Ic.3002525.AL123456.3.314308_314854.4719",
                "Erm_37_.3000392.AL123456.2231679_2232219.582",
                "efpA.3003955.NC_000962.3.3153038_3154631.4603",
                "mfpA.3003035.AL123456.3773015_3773567.4588",
                "mtrA.3000816.AL123456.3.3626662_3627349.4690",
                "LAM",
                "Delhi",
                "Beijing",
                "EAI",
                "X-type",
                "Haarlem",
                "lineage4",
                "S-type",
                "Ural",
                "Cameroon",
                "Ghana",
            ],
            "pyrazinamide": [
                "AAC_2___Ic.3002525.AL123456.3.314308_314854.4719",
                "Erm_37_.3000392.AL123456.2231679_2232219.582",
                "efpA.3003955.NC_000962.3.3153038_3154631.4603",
                "mfpA.3003035.AL123456.3773015_3773567.4588",
                "mtrA.3000816.AL123456.3.3626662_3627349.4690",
                "LAM",
                "Delhi",
                "Beijing",
                "EAI",
                "X-type",
                "Haarlem",
                "lineage4",
                "S-type",
                "Ural",
                "M.bovis",
                "Cameroon",
                "Tur",
                "BCG",
            ],
        },
        feature_id_path="raw_fList_final.txt",
        # in raw_fList_final.txt, spaces in lineage names in raw_fList.txt were removed for RF feature selection (e.g.'West African 2' to 'WestAfrican2'
        # ariba_result_dir='aribaResult_withBam',
        ariba_result_dir="aribaResult_withBam_new",
        log_path="CNN1D.log",
        # log_path='CNN1D.log'
        model_log_dir="CNN_model_log",
        n_fold=10,
        variants={
            "rifampicin": [
                "katG.3003392.NC_000962.3.2153888_2156111.4732.R463L",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A234G",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.G300W",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A431V",
                "embC.3003327.CP003248.4240040_4243325.2080.R738Q",
                "embC.3003327.CP003248.4240040_4243325.2080.T270I",
                "embB.3003326.AL123456.4246513_4249810.2078.M306I",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315T",
                "rpsL.3003395.AE000516.783533_783908.2087.K43R",
                "embC.3003327.CP003248.4240040_4243325.2080.V981L",
                "rpoB.3003283.AL123456.3.759806_763325.4199.S450L",
                "rpsL.3003395.AE000516.783533_783908.2087.K88R",
                "kasA.3003463.AL123456.2518114_2519365.2084.G269S",
                "gyrA.3003295.AL123456.7301_9818.2055.A90V",
                "embB.3003326.AL123456.4246513_4249810.2078.D1024N",
                "embB.3003326.AL123456.4246513_4249810.2078.G406A",
                "gyrA.3003295.AL123456.7301_9818.2055.S91P",
                "embB.3003326.AL123456.4246513_4249810.2078.M306V",
                "inhA.3003393.AL123456.1674201_1675011.2085.I194T",
                "Mycobacterium_avium_23S.3004164.NG_041979.1.0_3112.4157.A2274G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94N",
                "Mycobacterium_tuberculosis_16S.3003481.AL123456.3.1471846_1473382.3261.A1401G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94Y",
                "embB.3003326.AL123456.4246513_4249810.2078.G406D",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497R",
                "inhA.3003393.AL123456.1674201_1675011.2085.S94A",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315N",
                "embB.3003326.AL123456.4246513_4249810.2078.M306L",
                "gyrA.3003295.AL123456.7301_9818.2055.D94A",
                "gidB.3003470.AL123456.4407527_4408202.4553.A134E",
                "embB.3003326.AL123456.4246513_4249810.2078.G406S",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497K",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21T",
            ],
            "ethambutol": [
                "katG.3003392.NC_000962.3.2153888_2156111.4732.R463L",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A234G",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.G300W",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A431V",
                "rpoB.3003283.AL123456.3.759806_763325.4199.L511R",
                "rpoB.3003283.AL123456.3.759806_763325.4199.D516G",
                "rpoB.3003283.AL123456.3.759806_763325.4199.H526T",
                "embC.3003327.CP003248.4240040_4243325.2080.R738Q",
                "embA.3003453.CP003248.4243410_4246695.2081.P913S",
                "embB.3003326.AL123456.4246513_4249810.2078.E378A",
                "embC.3003327.CP003248.4240040_4243325.2080.T270I",
                "embC.3003327.CP003248.4240040_4243325.2080.N394D",
                "embR.3003455.AL123456.3.1416180_1417347.4712.C110Y",
                "kasA.3003463.AL123456.2518114_2519365.2084.G312S",
                "Mycobacterium_tuberculosis_ndh.3003461.AL123456.3.2101650_2103042.4726.V18A",
                "embB.3003326.AL123456.4246513_4249810.2078.M306I",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315T",
                "rpsL.3003395.AE000516.783533_783908.2087.K43R",
                "embC.3003327.CP003248.4240040_4243325.2080.V981L",
                "rpoB.3003283.AL123456.3.759806_763325.4199.S450L",
                "rpsL.3003395.AE000516.783533_783908.2087.K88R",
                "kasA.3003463.AL123456.2518114_2519365.2084.G269S",
                "thyA.3004153.NC_000962.3.3073679_3074471.4482.T202A",
                "gyrA.3003295.AL123456.7301_9818.2055.A90V",
                "gidB.3003470.AL123456.4407527_4408202.4553.S70R",
                "tlyA.3003445.AE000516.1908741_1909548.2076.N236K",
                "embB.3003326.AL123456.4246513_4249810.2078.D1024N",
                "embB.3003326.AL123456.4246513_4249810.2078.G406A",
                "gyrA.3003295.AL123456.7301_9818.2055.S91P",
                "Mycobacterium_tuberculosis_ndh.3003461.AL123456.3.2101650_2103042.4726.R268H",
                "embB.3003326.AL123456.4246513_4249810.2078.M306V",
                "inhA.3003393.AL123456.1674201_1675011.2085.I194T",
                "Mycobacterium_avium_23S.3004164.NG_041979.1.0_3112.4157.A2274G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94N",
                "Mycobacterium_tuberculosis_16S.3003481.AL123456.3.1471846_1473382.3261.A1401G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94Y",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.A420V",
                "iniA.3003448.AL123456.410837_412760.2099.S501W",
                "embB.3003326.AL123456.4246513_4249810.2078.G406D",
                "pncA.3003394.AL123456.2288680_2289241.4593.Q10R",
                "pncA.3003394.AL123456.2288680_2289241.4593.V7G",
                "pncA.3003394.AL123456.2288680_2289241.4593.L85P",
                "pncA.3003394.AL123456.2288680_2289241.4593.W68G",
                "pncA.3003394.AL123456.2288680_2289241.4593.Q141P",
                "pncA.3003394.AL123456.2288680_2289241.4593.S67P",
                "pncA.3003394.AL123456.2288680_2289241.4593.R154G",
                "pncA.3003394.AL123456.2288680_2289241.4593.L172P",
                "embB.3003326.AL123456.4246513_4249810.2078.Y334H",
                "pncA.3003394.AL123456.2288680_2289241.4593.M175V",
                "gyrA.3003295.AL123456.7301_9818.2055.D89G",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497R",
                "pncA.3003394.AL123456.2288680_2289241.4593.L85R",
                "gidB.3003470.AL123456.4407527_4408202.4553.P75S",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21V",
                "embB.3003326.AL123456.4246513_4249810.2078.D328Y",
                "inhA.3003393.AL123456.1674201_1675011.2085.S94A",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315N",
                "pncA.3003394.AL123456.2288680_2289241.4593.Q10P",
                "pncA.3003394.AL123456.2288680_2289241.4593.P54L",
                "pncA.3003394.AL123456.2288680_2289241.4593.V139A",
                "embB.3003326.AL123456.4246513_4249810.2078.M306L",
                "gyrA.3003295.AL123456.7301_9818.2055.D94A",
                "pncA.3003394.AL123456.2288680_2289241.4593.T47A",
                "pncA.3003394.AL123456.2288680_2289241.4593.T135P",
                "ethA.3003458.AE000516.2.4318327_4319797.4455.A381P",
                "embB.3003326.AL123456.4246513_4249810.2078.G406C",
                "gidB.3003470.AL123456.4407527_4408202.4553.A134E",
                "pncA.3003394.AL123456.2288680_2289241.4593.C14R",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.S150G",
                "gidB.3003470.AL123456.4407527_4408202.4553.A138E",
                "gyrA.3003295.AL123456.7301_9818.2055.G88C",
                "embB.3003326.AL123456.4246513_4249810.2078.G406S",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.I43S",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497K",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21T",
                "embB.3003326.AL123456.4246513_4249810.2078.S297A",
                "thyA.3004153.NC_000962.3.3073679_3074471.4482.H207R",
                "pncA.3003394.AL123456.2288680_2289241.4593.D12A",
                "pncA.3003394.AL123456.2288680_2289241.4593.F58L",
                "pncA.3003394.AL123456.2288680_2289241.4593.G132S",
                "gidB.3003470.AL123456.4407527_4408202.4553.H48Y",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315G",
                "gyrA.3003295.AL123456.7301_9818.2055.D89N",
                "pncA.3003394.AL123456.2288680_2289241.4593.W68R",
                "pncA.3003394.AL123456.2288680_2289241.4593.V155G",
                "embB.3003326.AL123456.4246513_4249810.2078.Y319C",
                "pncA.3003394.AL123456.2288680_2289241.4593.D49A",
                "pncA.3003394.AL123456.2288680_2289241.4593.Y103H",
                "pncA.3003394.AL123456.2288680_2289241.4593.H82R",
                "pncA.3003394.AL123456.2288680_2289241.4593.G97S",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315I",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.I43A",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.Q127P",
                "gyrA.3003295.AL123456.7301_9818.2055.G88A",
                "pncA.3003394.AL123456.2288680_2289241.4593.D12N",
                "pncA.3003394.AL123456.2288680_2289241.4593.A146T",
                "pncA.3003394.AL123456.2288680_2289241.4593.V9G",
                "pncA.3003394.AL123456.2288680_2289241.4593.L4W",
                "pncA.3003394.AL123456.2288680_2289241.4593.L27P",
                "rpsL.3003395.AE000516.783533_783908.2087.K88Q",
                "pncA.3003394.AL123456.2288680_2289241.4593.T160P",
                "pncA.3003394.AL123456.2288680_2289241.4593.T76P",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.Y337C",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.P131Q",
                "embB.3003326.AL123456.4246513_4249810.2078.D328G",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S140N",
            ],
            "isoniazid": [
                "katG.3003392.NC_000962.3.2153888_2156111.4732.R463L",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A234G",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.G300W",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A431V",
                "embC.3003327.CP003248.4240040_4243325.2080.R738Q",
                "embA.3003453.CP003248.4243410_4246695.2081.P913S",
                "embB.3003326.AL123456.4246513_4249810.2078.E378A",
                "embC.3003327.CP003248.4240040_4243325.2080.T270I",
                "embC.3003327.CP003248.4240040_4243325.2080.N394D",
                "embR.3003455.AL123456.3.1416180_1417347.4712.C110Y",
                "Mycobacterium_tuberculosis_ndh.3003461.AL123456.3.2101650_2103042.4726.V18A",
                "embB.3003326.AL123456.4246513_4249810.2078.M306I",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315T",
                "rpsL.3003395.AE000516.783533_783908.2087.K43R",
                "embC.3003327.CP003248.4240040_4243325.2080.V981L",
                "rpoB.3003283.AL123456.3.759806_763325.4199.S450L",
                "rpsL.3003395.AE000516.783533_783908.2087.K88R",
                "kasA.3003463.AL123456.2518114_2519365.2084.G269S",
                "thyA.3004153.NC_000962.3.3073679_3074471.4482.T202A",
                "gyrA.3003295.AL123456.7301_9818.2055.A90V",
                "embB.3003326.AL123456.4246513_4249810.2078.D1024N",
                "embB.3003326.AL123456.4246513_4249810.2078.G406A",
                "gyrA.3003295.AL123456.7301_9818.2055.S91P",
                "Mycobacterium_tuberculosis_ndh.3003461.AL123456.3.2101650_2103042.4726.R268H",
                "embB.3003326.AL123456.4246513_4249810.2078.M306V",
                "inhA.3003393.AL123456.1674201_1675011.2085.I194T",
                "Mycobacterium_avium_23S.3004164.NG_041979.1.0_3112.4157.A2274G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94N",
                "Mycobacterium_tuberculosis_16S.3003481.AL123456.3.1471846_1473382.3261.A1401G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94Y",
                "embB.3003326.AL123456.4246513_4249810.2078.G406D",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497R",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21V",
                "embB.3003326.AL123456.4246513_4249810.2078.D328Y",
                "inhA.3003393.AL123456.1674201_1675011.2085.S94A",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315N",
                "embB.3003326.AL123456.4246513_4249810.2078.M306L",
                "gyrA.3003295.AL123456.7301_9818.2055.D94A",
                "ethA.3003458.AE000516.2.4318327_4319797.4455.A381P",
                "embB.3003326.AL123456.4246513_4249810.2078.G406C",
                "gidB.3003470.AL123456.4407527_4408202.4553.A134E",
                "embB.3003326.AL123456.4246513_4249810.2078.G406S",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497K",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21T",
                "thyA.3004153.NC_000962.3.3073679_3074471.4482.H207R",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315G",
                "pncA.3003394.AL123456.2288680_2289241.4593.H82R",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315I",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.Q127P",
                "gyrA.3003295.AL123456.7301_9818.2055.G88A",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.G279D",
            ],
            "pyrazinamide": [
                "katG.3003392.NC_000962.3.2153888_2156111.4732.R463L",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A234G",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.G300W",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.A431V",
                "embC.3003327.CP003248.4240040_4243325.2080.R738Q",
                "embA.3003453.CP003248.4243410_4246695.2081.P913S",
                "embB.3003326.AL123456.4246513_4249810.2078.E378A",
                "embC.3003327.CP003248.4240040_4243325.2080.T270I",
                "embC.3003327.CP003248.4240040_4243325.2080.N394D",
                "embR.3003455.AL123456.3.1416180_1417347.4712.C110Y",
                "kasA.3003463.AL123456.2518114_2519365.2084.G312S",
                "Mycobacterium_tuberculosis_ndh.3003461.AL123456.3.2101650_2103042.4726.V18A",
                "embB.3003326.AL123456.4246513_4249810.2078.M306I",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315T",
                "rpsL.3003395.AE000516.783533_783908.2087.K43R",
                "embC.3003327.CP003248.4240040_4243325.2080.V981L",
                "rpoB.3003283.AL123456.3.759806_763325.4199.S450L",
                "rpsL.3003395.AE000516.783533_783908.2087.K88R",
                "kasA.3003463.AL123456.2518114_2519365.2084.G269S",
                "thyA.3004153.NC_000962.3.3073679_3074471.4482.T202A",
                "gyrA.3003295.AL123456.7301_9818.2055.A90V",
                "gidB.3003470.AL123456.4407527_4408202.4553.S70R",
                "tlyA.3003445.AE000516.1908741_1909548.2076.N236K",
                "embB.3003326.AL123456.4246513_4249810.2078.D1024N",
                "embB.3003326.AL123456.4246513_4249810.2078.G406A",
                "gyrA.3003295.AL123456.7301_9818.2055.S91P",
                "Mycobacterium_tuberculosis_ndh.3003461.AL123456.3.2101650_2103042.4726.R268H",
                "embB.3003326.AL123456.4246513_4249810.2078.M306V",
                "inhA.3003393.AL123456.1674201_1675011.2085.I194T",
                "Mycobacterium_avium_23S.3004164.NG_041979.1.0_3112.4157.A2274G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94N",
                "Mycobacterium_tuberculosis_16S.3003481.AL123456.3.1471846_1473382.3261.A1401G",
                "gyrA.3003295.AL123456.7301_9818.2055.D94Y",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.A420V",
                "iniA.3003448.AL123456.410837_412760.2099.S501W",
                "embB.3003326.AL123456.4246513_4249810.2078.G406D",
                "pncA.3003394.AL123456.2288680_2289241.4593.Q10R",
                "pncA.3003394.AL123456.2288680_2289241.4593.H57D",
                "pncA.3003394.AL123456.2288680_2289241.4593.V7G",
                "pncA.3003394.AL123456.2288680_2289241.4593.L85P",
                "pncA.3003394.AL123456.2288680_2289241.4593.W68G",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.I43T",
                "pncA.3003394.AL123456.2288680_2289241.4593.Q141P",
                "pncA.3003394.AL123456.2288680_2289241.4593.S67P",
                "pncA.3003394.AL123456.2288680_2289241.4593.L172P",
                "embB.3003326.AL123456.4246513_4249810.2078.Y334H",
                "pncA.3003394.AL123456.2288680_2289241.4593.M175V",
                "gyrA.3003295.AL123456.7301_9818.2055.D89G",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497R",
                "pncA.3003394.AL123456.2288680_2289241.4593.L85R",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21V",
                "pncA.3003394.AL123456.2288680_2289241.4593.V139L",
                "embB.3003326.AL123456.4246513_4249810.2078.D328Y",
                "inhA.3003393.AL123456.1674201_1675011.2085.S94A",
                "katG.3003392.NC_000962.3.2153888_2156111.4732.S315N",
                "pncA.3003394.AL123456.2288680_2289241.4593.Q10P",
                "pncA.3003394.AL123456.2288680_2289241.4593.P54L",
                "pncA.3003394.AL123456.2288680_2289241.4593.V139A",
                "embB.3003326.AL123456.4246513_4249810.2078.M306L",
                "gyrA.3003295.AL123456.7301_9818.2055.D94A",
                "pncA.3003394.AL123456.2288680_2289241.4593.T135P",
                "ethA.3003458.AE000516.2.4318327_4319797.4455.A381P",
                "embB.3003326.AL123456.4246513_4249810.2078.G406C",
                "gidB.3003470.AL123456.4407527_4408202.4553.A134E",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.S150G",
                "gidB.3003470.AL123456.4407527_4408202.4553.A138E",
                "gyrA.3003295.AL123456.7301_9818.2055.G88C",
                "embB.3003326.AL123456.4246513_4249810.2078.G406S",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.I43S",
                "embB.3003326.AL123456.4246513_4249810.2078.Q497K",
                "inhA.3003393.AL123456.1674201_1675011.2085.I21T",
                "embB.3003326.AL123456.4246513_4249810.2078.S297A",
                "thyA.3004153.NC_000962.3.3073679_3074471.4482.H207R",
                "pncA.3003394.AL123456.2288680_2289241.4593.D12A",
                "pncA.3003394.AL123456.2288680_2289241.4593.W119R",
                "pncA.3003394.AL123456.2288680_2289241.4593.F58L",
                "pncA.3003394.AL123456.2288680_2289241.4593.G132S",
                "pncA.3003394.AL123456.2288680_2289241.4593.W68R",
                "pncA.3003394.AL123456.2288680_2289241.4593.A134V",
                "embB.3003326.AL123456.4246513_4249810.2078.Y319C",
                "pncA.3003394.AL123456.2288680_2289241.4593.D49A",
                "pncA.3003394.AL123456.2288680_2289241.4593.H82R",
                "folC.3004157.NC_000962.3.2746134_2747598.4481.I43A",
                "gyrA.3003295.AL123456.7301_9818.2055.G88A",
                "pncA.3003394.AL123456.2288680_2289241.4593.D12N",
                "pncA.3003394.AL123456.2288680_2289241.4593.V9G",
                "pncA.3003394.AL123456.2288680_2289241.4593.L4W",
                "pncA.3003394.AL123456.2288680_2289241.4593.L27P",
                "pncA.3003394.AL123456.2288680_2289241.4593.L159R",
                "rpsL.3003395.AE000516.783533_783908.2087.K88Q",
                "pncA.3003394.AL123456.2288680_2289241.4593.T160P",
                "pncA.3003394.AL123456.2288680_2289241.4593.T76P",
                "pncA.3003394.AL123456.2288680_2289241.4593.H51Q",
                "ethA.3003458.AE000516.2.4318327_4319797.4455.D58A",
            ],
        },
    ):
        self.total_epochs = total_epochs
        self.learning_rate = learning_rate
        self.l2 = l2
        self.batch_size = batch_size
        self.window_size = window_size
        self.lineageNgenePresent = lineageNgenePresent
        self.feature_id_path = feature_id_path
        self.ariba_result_dir = ariba_result_dir
        self.log_path = log_path
        self.variants = variants
        self.model_log_dir = model_log_dir
        self.n_fold = n_fold


_ALLOWED_BASES = ["A", "C", "G", "T"]
_MIN_COV = 1
_MAX_COV = 856
firstLine_TB_4antibio = ["rifampicin", "ethambutol", "isoniazid", "pyrazinamide"]
ext = ".txt"
pre_sra_path = "sra_withFeature_"
pre_label_path = "label_Y_"
pre_feature_path = "featureM_X_"


def generate_multiInputsNlabels4CNN(hparams, drug):
    """ create a list with length (no. of variants + 1(gene presents and lineage in one dimension vector)) and corresonding np 
    arrays as elements, which are inputs of CNN. Loop each sample listed in 'sra_withFeature_(DRUG).txt', extract normalized 
    counts (21x4) for each variant and the additiional 1D features from one sample using generate_inputFromOneIsolate() and 
    append to the arrays into corresponding elements of the list base on the list's index"""
    X = []
    Y = []

    # collect all coverage numbers to find out min and max, then normalize coverages
    cov_l = []

    # The orders in sra_list_file,label_file and feature_file correspond, which were generated for traditional machine learning.
    text = open("".join([pre_sra_path, drug, ext])).read()
    text = text.rstrip()
    sra_list = text.split("\n")

    text = open("".join([pre_label_path, drug, ext])).read()
    text = text.rstrip()
    labels = text.split("\n")
    labels = list(map(int, labels))

    text = open(hparams.feature_id_path).read()
    text = text.rstrip()
    f_ID = text.split("\n")

    df = pd.read_csv(
        "".join([pre_feature_path, drug, ext]), header=None, sep="\s+", dtype=int
    )
    df.columns = f_ID

    l_oneD = len(hparams.lineageNgenePresent[drug])
    l_var = len(hparams.variants[drug])

    for i, sra in enumerate(sra_list):
        report_path = hparams.ariba_result_dir + "/outRun_" + sra + "/report.tsv"
        if not (os.path.isfile(report_path)):
            f.write("Report file for {} does not exist".format(sra))
        else:
            report_df = pd.read_csv(report_path, sep="\t")
            var_input_fromOneSample, cov = generate_var_inputFromOneIsolate(
                i, df, report_df, sra, hparams, drug
            )
            cov_l.extend(cov)

            # prepare lineage and gene present input
            oneD_feature = np.zeros((1, l_oneD))
            for j, fea in enumerate(hparams.lineageNgenePresent[drug]):
                if df.loc[i, fea] == 1:
                    oneD_feature[0, j] = 1

            # prepare variant input
            Y.append(labels[i])
            if len(X) == 0:
                for v in var_input_fromOneSample:
                    X.append(v)
                X.append(oneD_feature)
            else:
                for k, v in enumerate(var_input_fromOneSample):
                    X[k] = np.append(X[k], v, axis=0)
                X[l_var] = np.append(X[l_var], oneD_feature, axis=0)

    cov_l = np.array(cov_l)
    c_min = np.min(cov_l)
    c_max = np.max(cov_l)
    f.write("minimum coverage:{}; maximum coverage:{}".format(c_min, c_max))

    return X, Y


def generate_var_inputFromOneIsolate(i, df, report_df, sra, hparams, drug):
    """ loop the list of variants in one sample and generate a list of np arrays with length no. of variants.
    When ['known_var']=='1' and 'has_known_var'=='1', ref_ctg_change is same to known_var_change.
    So, report_df['ref_name'][i]+'.'+report_df['ref_ctg_change'][i] is how variant ID is composed.
    Create on more column to save the variant IDs"""
    var_input_singleSample = []
    cov = []
    var_join = []

    # Add one additional row of variant IDs to report_df.
    for i_r, r in report_df.iterrows():
        if r["known_var"] == "1" and r["has_known_var"] == "1":
            var_join.append(r["ref_name"] + "." + r["known_var_change"])
        else:
            var_join.append(r["ref_name"] + "." + r["ref_ctg_change"])
    report_df["var_ID"] = var_join

    for var in hparams.variants[drug]:
        # generate a normalized count matrix (21x4) for one variant of one sample
        nor_acc = np.zeros((1, hparams.window_size, len(_ALLOWED_BASES)))

        if (not (var in report_df["var_ID"].values)) and df.loc[i, var] == 1:
            # f_sra_false_vc.write('{} does not exist in {}, however it exists in feature matrix\n'.format(var, sra))
            f_sra_false_vc.write("{}\n".format(sra))
        if (var in report_df["var_ID"].values) and df.loc[i, var] == 1:
            v_index = 10000
            for m, v_name in enumerate(report_df["var_ID"].values):
                if v_name == var:
                    v_index = m
                    break
            # get contig ID and cluster where the var is located
            ctg = report_df.loc[v_index, "ctg"]
            cluster = report_df.loc[v_index, "cluster"]
            # obtain var locus in that contig
            v_locus = int(report_df.loc[v_index, "ctg_start"])

            depth_path = (
                hparams.ariba_result_dir
                + "/outRun_"
                + sra
                + "/clusters/"
                + cluster
                + "/assembly.reads_mapped.bam.read_depths.gz"
            )
            df_depth = pd.read_csv(depth_path, header=None, sep="\t")
            df_depth.columns = ["ctg", "loc", "ctg_base", "alt", "t_read", "c_4base"]
            df_depth["loc"] = df_depth["loc"].astype(int)
            df_depth = df_depth[df_depth["ctg_base"].isin(_ALLOWED_BASES)]
            # Loop each locus in the window

            start = v_locus - (hparams.window_size // 2)
            end = v_locus + (hparams.window_size // 2) + 1
            for locus in range(start, end):
                if locus in df_depth.loc[df_depth["ctg"] == ctg, "loc"]:
                    df_re = df_depth.loc[
                        (df_depth["ctg"] == ctg) & (df_depth["loc"] == locus),
                        ["ctg_base", "alt", "c_4base"],
                    ]
                    # In some case, len(df_re)>1, e.g.: (I only take the first one so far)
                    # 2570        C   .      80
                    # 2571        C  CA    77,1
                    # if len(df_re) != 1:
                    # f.write('duplicated locus for {} in {}, {}'.format(var,sra,df_re))
                    # else:
                    if df_re["alt"].iloc[0] == ".":
                        nor_acc[
                            0,
                            (locus - start),
                            _ALLOWED_BASES.index(df_re["ctg_base"].iloc[0]),
                        ] = 1
                        cov.append(int(df_re["c_4base"].iloc[0]))
                    else:
                        alt_base = df_re["alt"].iloc[0].split(",")
                        alt_base.insert(0, df_re["ctg_base"].iloc[0])
                        base_count = df_re["c_4base"].iloc[0].split(",")
                        base_count = list(map(int, base_count))
                        t_depth = sum(base_count)
                        if len(alt_base) != len(base_count):
                            f.write(
                                "# of based does not match # of counts for {} in {}, {}, {}".format(
                                    var, sra, ctg, v_locus
                                )
                            )
                        else:
                            for i_b, b in enumerate(alt_base):
                                nor_acc[0, (locus - start), _ALLOWED_BASES.index(b)] = (
                                    base_count[i_b] / t_depth
                                )
                                cov.append(base_count[0])

        var_input_singleSample.append(nor_acc)

    return var_input_singleSample, cov


def build_model(hparams, drug):
    """create Conv1D model sets for multi inputs to generate multi flatten outputs, then do binary classification"""
    l_v = len(hparams.variants[drug])
    digit = [None] * l_v
    x = [None] * l_v
    l2_reg = keras.regularizers.l2

    for i in range(l_v):
        digit[i] = Input(shape=(hparams.window_size, len(_ALLOWED_BASES)))
        x[i] = Conv1D(
            filters=32,
            kernel_size=5,
            activation=tf.nn.relu,
            kernel_regularizer=l2_reg(hparams.l2),
        )(digit[i])
        x[i] = Conv1D(
            filters=32,
            kernel_size=3,
            activation=tf.nn.relu,
            kernel_regularizer=l2_reg(hparams.l2),
        )(x[i])
        x[i] = MaxPooling1D(pool_size=3, strides=1)(x[i])
        x[i] = Flatten()(x[i])

    genePresent_lineage = Input(shape=(len(hparams.lineageNgenePresent[drug]),))
    x.append(genePresent_lineage)
    digit.append(genePresent_lineage)
    concatenated = concatenate(x)

    out = Dense(
        units=960, activation=tf.nn.relu, kernel_regularizer=l2_reg(hparams.l2)
    )(concatenated)
    out = Dropout(rate=0.3)(out)

    out = Dense(
        units=640, activation=tf.nn.relu, kernel_regularizer=l2_reg(hparams.l2)
    )(out)
    out = Dropout(rate=0.3)(out)

    out = Dense(
        units=320, activation=tf.nn.relu, kernel_regularizer=l2_reg(hparams.l2)
    )(concatenated)
    out = Dropout(rate=0.3)(out)

    out = Dense(
        units=160, activation=tf.nn.relu, kernel_regularizer=l2_reg(hparams.l2)
    )(out)
    out = Dropout(rate=0.3)(out)

    out = Dense(units=80, activation=tf.nn.relu, kernel_regularizer=l2_reg(hparams.l2))(
        concatenated
    )
    out = Dropout(rate=0.3)(out)

    out = Dense(units=40, activation=tf.nn.relu, kernel_regularizer=l2_reg(hparams.l2))(
        out
    )
    out = Dropout(rate=0.3)(out)

    out = Dense(units=1, activation="sigmoid")(out)

    return digit, out


def run(hparams, trainX, trainY, testX, testY, drug, seed=1):
    """Creates a model, runs training and evaluation on one of n-fold."""
    # Set seed for reproducibility.
    random.seed(seed)
    tf.random.set_seed(seed)

    digit, out = build_model(hparams, drug)
    model = Model(digit, out)
    # print(model.summary())

    optimizer = tf.keras.optimizers.SGD(lr=hparams.learning_rate)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        hparams.model_log_dir, histogram_freq=1
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.FalseNegatives(),
            keras.metrics.FalsePositives(),
            keras.metrics.TrueNegatives(),
            keras.metrics.TruePositives(),
            keras.metrics.AUC(),
        ],
    )

    print("Training the model...")
    model.fit(
        trainX,
        trainY,
        epochs=hparams.total_epochs,
        batch_size=hparams.batch_size,
        callbacks=[tensorboard_callback],
        verbose=0,
    )
    test_metrics = model.evaluate(testX, testY, verbose=0)

    # f.write('Final test metrics - loss: {} - accuracy: {} - FN: {} - FP: {} - TN: {} - TP: {} - AUC: {}'.format(
    # test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3], test_metrics[4], test_metrics[5], test_metrics[6]))

    return test_metrics


def run_nfold_CV(hparams, drug):
    """ run n-fold CV, n is gave in hparams.n_fold"""
    X, Y = generate_multiInputsNlabels4CNN(hparams, drug)
    len_Y = len(Y)
    len_X = len(X)
    f.write(
        "length of X: {}, length of Y: {}, length of X[0]: {}".format(
            len_X, len_Y, len(X[0])
        )
    )

    # Iteratively partition whole dataset into training and test sets for n-fold CV; save results of CV into metrics_nf
    metrics_nf = []
    for i_fold in range(hparams.n_fold):
        n_pos = 0
        n_neg = 0
        trainX = []
        trainY = []
        testX = []
        testY = []
        for i in range(len_Y):
            # make training and test sets have same proportion for negative and positive samples .
            if (Y[i] == 0 and n_neg % hparams.n_fold == i_fold) or (
                Y[i] == 1 and n_pos % hparams.n_fold == i_fold
            ):
                testY.append(Y[i])
                # if it is first sample, initialize every input, else append to corresponding inputs
                if len(testX) == 0:
                    for k in range(len_X):
                        testX.append(np.array([X[k][i]]))
                else:
                    for k in range(len_X):
                        testX[k] = np.append(testX[k], np.array([X[k][i]]), axis=0)
            else:
                trainY.append(Y[i])
                if len(trainX) == 0:
                    for k in range(len_X):
                        trainX.append(np.array([X[k][i]]))
                else:
                    for k in range(len_X):
                        trainX[k] = np.append(trainX[k], np.array([X[k][i]]), axis=0)

            if Y[i] == 0:
                n_neg += 1
            if Y[i] == 1:
                n_pos += 1

        test_m = run(hparams, trainX, trainY, testX, testY, drug)
        metrics_nf.append(test_m)

    metrics_nf = np.array(metrics_nf)
    metrics_mean = metrics_nf.mean(axis=0)
    f.write(
        "Metrics of {}-fold CV - loss: {} - accuracy: {} - FN: {} - FP: {} - TN: {} - TP: {} - AUC: {}\n".format(
            hparams.n_fold,
            metrics_mean[0],
            metrics_mean[1],
            metrics_mean[2],
            metrics_mean[3],
            metrics_mean[4],
            metrics_mean[5],
            metrics_mean[6],
        )
    )

    fn, fp, tn, tp = metrics_mean[2], metrics_mean[3], metrics_mean[4], metrics_mean[5]
    f.write("Accuracy: {}\n".format(str((tp + tn) / float(tp + tn + fp + fn))))
    f.write("Specificity: {}\n".format(str(tn / float(tn + fp))))
    f.write("Sensitivity: {}\n:".format(str(tp / float(tp + fn))))
    Precision = tp / float(tp + fp)
    f.write("Precision: {}\n".format(Precision))
    Recall = tp / float(tp + fn)  # recall is another name of sensitivity
    f.write(
        "F-Measure: {}\n".format(str(2 * (Recall * Precision) / (Recall + Precision)))
    )


def evaluate_models_on4drugs(hparams):
    """ run n-fold CV on each of the models for the 4 first-line TB drugs"""

    for drug in firstLine_TB_4antibio:
        f.write("AMR prediction process on {}\n".format(drug))
        run_nfold_CV(hparams, drug)


hparams = baseHparamsNvars()
f = open(hparams.log_path, "w")
f_sra_false_vc = open("sra_false_vc.txt", "w")
evaluate_models_on4drugs(hparams)
f.close()
f_sra_false_vc.close()
