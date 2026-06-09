import pandas as pd
from ucimlrepo import fetch_ucirepo
from it.combibrivioSLR.dataset.dataset_analisi import DatasetAnalisi
from it.combibrivioSLR.dataset.grafici import Grafici

class DatasetManager:
    def __init__(self):
        self.__dftest = None
        self.__dftrain = None
        self.load()
        self.__data_ana = DatasetAnalisi()
        self.__grafici = Grafici()

    def load(self):
        self.__dftest = pd.read_csv("csvs/test.csv")
        self.__dftrain = pd.read_csv("csvs/train.csv")

    def set_data(self, data):
        self.__dataset = data

    def analisi(self):
        val_nan = self.__data_ana.valori_nulli(self.__dftrain)
        val_strani = self.__data_ana.valori_stringhe(self.__dftrain)
        outliers = self.outlier()
        norm = self.__data_ana.normality(self.__dftrain)
        return {
            "val_nan": val_nan,
            "val_strani": val_strani,
            "outliers": outliers,  # visto che sono tutti categorici fissi (opzioni) non ha senso parlare di outliers
            "test normalità": norm
        }

    def outlier(self):
        outl_iqr = self.__data_ana.outliers_iqr_per_col(self.__dftrain)
        outl_zscore = self.__data_ana.outliers_zscore_per_col(self.__dftrain)
        return {
            "outl_iqr": outl_iqr,
            "outl_zscore": outl_zscore
        }

    def grafici(self):
        correlation = None  # self.__grafici.plot_correlation(self.__dftrain) impossibile fare su categorici
        list_hist = []
        for col in self.__dftrain.columns:
            hist = self.__grafici.plot_hist(self.__dftrain, col)
            list_hist.append(hist)
        return {
            "correlation": correlation,
            "hist": list_hist
        }

    def clean(self):
        self.__dftrain = self.__data_ana.clean_data(self.__dftrain)

    def clean_data(self, dataframe):
        return self.__data_ana.clean_data(dataframe)

    def stampa(self):
        print(self.__dftrain)

    def correlazione(self):
        return self.__data_ana.correlazione(self.__dftrain)

    def get_datatrain(self):
        return self.__dftrain

    def get_datatest(self):
        return self.__dftest