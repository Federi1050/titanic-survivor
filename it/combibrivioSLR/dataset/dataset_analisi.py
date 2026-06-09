import numpy as np
from scipy.stats import zscore, chi2_contingency
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DatasetAnalisi:

    def outliers_iqr_per_col(self, data):
        result = {}
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers_count = data[(data[col] < lower) | (data[col] > upper)].shape[0]
            result[col] = outliers_count
        return result
    
    def outliers_zscore_per_col(self, data, threshold=3):
        result = {}
        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if col_data.std() == 0:
                result[col] = 0
                continue
            z_scores = zscore(col_data)
            outliers_count = np.sum(np.abs(z_scores) > threshold)
            result[col] = int(outliers_count)
        return result
    
    def valori_stringhe(self, data):
        cat_cols = data.select_dtypes(include=["object","string"]).columns
        string = ""
        for col in cat_cols:
            string += f"\n{col} ({data[col].nunique()} valori unici): {data[col].unique()}"
        string = string.strip()
        return string

    def valori_nulli(self, data):
        return data.isnull().sum().sort_values(ascending=False)


    def clean_data(self, data):
        data = data.drop_duplicates()
        # gestione outlier   
        # elimina caratteri strani
        # gestire nan 
        #data = data.replace('?', np.nan)
        #data["stalk-root"]=data["stalk-root"].fillna(data["stalk-root"].mode()[0])
        # data.loc[data["poisonous"].isin(["unknown edibility", "not recommended"]), "poisonous"] = "definitely poisonous"

        # colonna con valori tutti iguali per ogni riga
        data = data.drop(columns=["Name","Ticket","Cabin","PassengerId"])
        data["Age"]=data["Age"].fillna(data["Age"].median() + 0.5)
        data["Embarked"] = data["Embarked"].replace("nan", np.nan)
        data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
        data["Embarked"] = data["Embarked"].map({
        "S": "Southampton",
        "C": "Cherbourg",
        "Q": "Queenstown" })
    
        # encoding colonne categoriche:
        le = LabelEncoder()
        
        data["Sex"] = le.fit_transform(data["Sex"].astype(str))

        return data



    def correlazione(self, data):
        risultato = []
        for x in data.columns:
            for y in data.columns:
                risultato.append(f"corr {x} {y} : {self.crames_v(data[x], data[y])}")
        return risultato

    def crames_v(self, x, y):
        # si basa sul test del chi-quadrato
        # misura associazione 2 var categoriche
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        phi2 = chi2 / n
        phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr/ min((kcorr - 1), (rcorr-1)))

    def normality(self, data):
        pass
        # su dataset categorici non ha senso farlo
    