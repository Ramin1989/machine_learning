import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



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
    
    feature_columns = [column_name for column_name in heart_data.columns if column_name != 'output']

    stdandardized_columns = standardize_data(heart_data[feature_columns])
    
    X_train, X_test, y_train, y_test = train_test_split(stdandardized_columns, heart_data['output'],
                                                        test_size=0.2, random_state=10)
    

    # print(f'x_train: {np.shape(X_train)}')
    # print(f'x_test: {np.shape(X_test)}')

    classifer = KNeighborsClassifier(n_neighbors=10)
    classifer.fit(X_train, y_train)

    predictions = classifer.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(predictions)
    print(accuracy)







    


    




    