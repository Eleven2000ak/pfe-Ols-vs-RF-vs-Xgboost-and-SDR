#import pandas as pd
#from sklearn.model_selection import train_test_split

#df = pd.read_csv(r"C:\Users\hp\Downloads\pfe_comparaison_modeles\data\raw\housing.csv")

# Missing values
#df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Split
#X = df.drop("median_house_value", axis=1)
#y = df["median_house_value"]

# One-hot encoding
#X = pd.get_dummies(X, drop_first=True)

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42
#)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_and_preprocess(path, test_size=0.2, random_state=42):

    
    df = pd.read_csv(r"C:\Users\hp\Downloads\pfe_comparaison_modeles\data\raw\housing.csv")

    # target
    y = df["median_house_value"]
    X = df.drop("median_house_value", axis=1)

    # missing values
    imputer = SimpleImputer(strategy="median")
    X[["total_bedrooms"]] = imputer.fit_transform(X[["total_bedrooms"]])

    # encoding
    X = pd.get_dummies(X, drop_first=True)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # FORCE numeric (critical fix)
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test, y_train, y_test