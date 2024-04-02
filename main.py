import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



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
    print(heart_data.head())
    feature_columns = [column_name for column_name in heart_data.columns if column_name != 'output']

    stdandardized_columns = standardize_data(heart_data[feature_columns])
    
    X_train, X_test, y_train, y_test = train_test_split(stdandardized_columns, heart_data['output'],
                                                        test_size=0.2, random_state=10)
    

    # print(f'x_train: {np.shape(X_train)}')
    # print(f'x_test: {np.shape(X_test)}')

    kn_classifer = KNeighborsClassifier(n_neighbors=5, weights='distance', p=1)
    kn_classifer.fit(X_train, y_train)

    kn_predictions = kn_classifer.predict(X_test)

    kn_accuracy = accuracy_score(y_test, kn_predictions)

    print(f'K nearest neighbors classifer accuracy: {kn_accuracy}')

    dt_classifer = DecisionTreeClassifier(random_state=10, max_depth=3)
    dt_classifer.fit(X_train, y_train)

    dt_perdictions = dt_classifer.predict(X_test)

    dt_accuracy = accuracy_score(y_test, dt_perdictions)

    print(f'Decision Tree classifer accuracy: {dt_accuracy}')







    


    




    