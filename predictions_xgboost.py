import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

features = pd.read_csv("clean_data/features.csv")
test_features = pd.read_csv("clean_data/test_features.csv")

features['stratify_col'] = features['window'].astype(str) + "_" + features['var_rpta_alt'].astype(str)

X = features.drop(columns=['var_rpta_alt', 'stratify_col'])
y = features['var_rpta_alt']
stratify_col = features['stratify_col']

# First split: Train+Val and Test (stratified by the composite column)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, stratify=stratify_col, test_size=0.2, random_state=42
)

train_val_df = features.loc[X_train_val.index]
train_val_df['stratify_col'] = train_val_df['window'].astype(str) + "_" + train_val_df['var_rpta_alt'].astype(str)

# Second split: Train and Validation (stratified by the composite column in train+val split)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25,
    stratify=train_val_df['stratify_col'], random_state=42
)

id_columns = ["window", "fecha_var_rpta_alt", "num_oblig_orig_enmascarado", "num_oblig_enmascarado", "nit_enmascarado"]
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numerical_cols = numerical_cols.difference(id_columns)
log_cols = ['pago_total_0', 'pago_total_1', 'pago_total_2', 'tot_activos', 'tot_patrimonio', 'total_ing']

X_train.drop(columns=id_columns, inplace=True)
X_val.drop(columns=id_columns, inplace=True)
X_test.drop(columns=id_columns, inplace=True)

imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X_train_after = imp_mode.fit_transform(X_train)
X_train = pd.DataFrame(X_train_after, columns=X_train.columns, index=X_train.index)

X_val_after = imp_mode.transform(X_val)
X_val = pd.DataFrame(X_val_after, columns=X_val.columns, index=X_val.index)

X_test_after = imp_mode.transform(X_test)
X_test = pd.DataFrame(X_test_after, columns=X_test.columns, index=X_test.index)

numerical_transformer = StandardScaler()  # Standardize numerical variables
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
log_transformer = FunctionTransformer(np.log1p, validate=True)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols),
        ('num_log', log_transformer, log_cols),
    ]
)

# Apply the transformations to your datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Define the XGBClassifier
xgb = XGBClassifier(random_state=42, eval_metric='logloss')


# Set up the parameter grid for GridSearch
param_grid = {
    'n_estimators': [300, 400, 500],  # Number of trees
    'max_depth': [6, 8, 10],  # Maximum depth of trees
    'learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'subsample': [0.8, 1.0],  # Subsample ratio
    'colsample_bytree': [0.8, 1.0],  # Feature subsample ratio
    'gamma': [0, 1],  # Minimum loss reduction to make a further partition
    'scale_pos_weight': [1.5],  # Balancing classes
}

# Define the GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, average='macro'),  # Use accuracy as the scoring metric
    cv=3,
    verbose=2,
    n_jobs=-1  # Use all available processors
)

# Fit the grid search to the training data
grid_search.fit(X_train_processed, y_train)

# Best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Validate the model on the validation set
y_val_pred = best_model.predict(X_val_processed)
val_report = classification_report(y_val, y_val_pred)

print("Validation")
print(val_report)

y_test_pred = best_model.predict(X_test_processed)
test_report = classification_report(y_test, y_test_pred)

print("Testing")
print(test_report)

nits_kaggle = list(test_features["nit_enmascarado"])
oblig_kaggle = list(test_features["num_oblig_enmascarado"])
test_features.drop(columns=["nit_enmascarado", "num_oblig_enmascarado", "window"], inplace=True)
X_kaggle_after = imp_mode.transform(test_features)

X_kaggle = pd.DataFrame(X_kaggle_after, columns=test_features.columns, index=test_features.index)
X_kaggle_processed = preprocessor.transform(X_kaggle)

y_predicted = best_model.predict(X_kaggle_processed)

final_data = {
    "nit_enmascarado": nits_kaggle,
    "num_oblig_enmascarado": oblig_kaggle,
    "var_rpta_alt": y_predicted
}
final_data = pd.DataFrame(final_data)

target_clients = pd.read_csv("data/prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv")

answer = target_clients.merge(final_data, on=["nit_enmascarado", "num_oblig_enmascarado"], how="left")
answer["var_rpta_alt"] = answer["var_rpta_alt"].fillna(0)

answer["ID"] = answer['nit_enmascarado'].astype(str) + "#" + answer['num_oblig_orig_enmascarado'].astype(str) + "#" + \
               answer['num_oblig_enmascarado'].astype(str)
answer.drop(columns=["nit_enmascarado", "num_oblig_enmascarado", "num_oblig_orig_enmascarado", "fecha_var_rpta_alt"],
            inplace=True)
answer = answer[['ID', 'var_rpta_alt']]
answer["var_rpta_alt"] = answer["var_rpta_alt"].astype(int)
answer.to_csv("answer_4.csv", index=False)
