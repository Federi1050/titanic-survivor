import pandas as pd
from ucimlrepo import fetch_ucirepo
from it.combibrivioSLR.dataset.dataset_analisi import DatasetAnalisi
from it.combibrivioSLR.dataset.grafici import Grafici


class DatasetManager:
    def __init__(self):
        self.load()
        self.__data_ana = DatasetAnalisi()
        self.__grafici = Grafici()

    def load(self):
        self.__dftest = pd.read_csv("csvs/test.csv")
        self.__dftrain = pd.read_csv("csvs/train.csv")


    def set_data(self, data):
        self.__dataset = data

    def analisi(self):
        val_nan = self.__data_ana.valori_nulli(self.__dataset)
        val_strani = self.__data_ana.valori_stringhe(self.__dataset)
        outliers = self.outlier()
        return {
            "val_nan": val_nan,
            "val_strani": val_strani,
            "outliers": None  # visto che sono tutti categorici fissi (opzioni) non ha senso parlare di outliers
        }

    def outlier(self):
        outl_iqr = self.__data_ana.outliers_iqr_per_col(self.__dataset)
        outl_zscore = self.__data_ana.outliers_zscore_per_col(self.__dataset)
        return {
            "outl_iqr": outl_iqr,
            "outl_zscore": outl_zscore
        }

    def normality(self):
        # su categorici non ha senso farla
        norm = self.__data_ana.normality(self.__dataset)
        return {
            "normality": norm
        }

    def grafici(self):
        correlation = None  # self.__grafici.plot_correlation(self.__dataset) impossibile fare su categorici
        list_hist = []
        for col in self.__dataset.columns:
            hist = self.__grafici.plot_hist(self.__dataset, col)
            list_hist.append(hist)
        return {
            "correlation": correlation,
            "hist": list_hist
        }

    def clean(self):
        self.__dataset = self.__data_ana.clean_data(self.__dataset)

    def stampa(self):
        print(self.__dataset)

    def correlazione(self):
        return self.__data_ana.correlazione(self.__dataset)

    def get_datatset(self):
        return self.__dataset