import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
    loaded_data = pd.read_csv(path)
    return loaded_data

def hist_plot(column_data: pd.Series) -> None:
    fig, axis = plt.subplots()
    sns.histplot(column_data)
    

def standardize_data(data: pd.DataFrame) -> np.ndarray:
    std_scaler = StandardScaler()
    standardized_data  = std_scaler.fit_transform(data)
    return standardized_data

if __name__ == "__main__":
    path_to_data = "heart.csv"
    heart_data = load_csv(path_to_data)

    hist_plot(heart_data["age"])

    column_list = ["age", "caa", "chol"]
    stdandardized_columns = standardize_data(heart_data[column_list])

    hist_plot(stdandardized_columns[:, column_list.index("age")])
    
    plt.show()
    




    