# 1. IMPORTING LIBRARIES AND CONFIGURATIONS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")

# 2. LOADING AND CLEANING THE DATASET
df = pd.read_csv('Travel.csv')
df.drop('CustomerID', axis=1, inplace=True)
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

# Handling missing values
num_features = df.select_dtypes(exclude='O').columns
cat_features = df.select_dtypes(include='O').columns

for feature in num_features:
    df[feature] = df[feature].fillna(df[feature].median())
for feature in cat_features:
    df[feature] = df[feature].fillna(df[feature].mode()[0])

# Converting to int
for feature in num_features:
    if feature != 'MonthlyIncome':
        df[feature] = df[feature].astype(int)

# Feature engineering
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns=['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], inplace=True)

# 3. SPLITTING AND PREPROCESSING DATA
X = df.drop('ProdTaken', axis=1)
Y = df['ProdTaken']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=7)

num_features_in_X = X.select_dtypes(exclude='O').columns
cat_features_in_X = X.select_dtypes(include='O').columns

preprocessor = ColumnTransformer([
    ('OneHotEncoder', OneHotEncoder(drop='first'), cat_features_in_X),
    ('StandardScaler', StandardScaler(), num_features_in_X)
])

X_Train = preprocessor.fit_transform(X_Train)
X_Test = preprocessor.transform(X_Test)

# 4. MODEL DEFINITION AND INITIAL EVALUATION
def evaluate_score(true, pred):
    return (
        accuracy_score(true, pred),
        classification_report(true, pred),
        confusion_matrix(true, pred),
        precision_score(true, pred),
        recall_score(true, pred),
        f1_score(true, pred),
        roc_auc_score(true, pred)
    )

models = {
    'LogisticRegression': LogisticRegression(),
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'XGBClassifier': XGBClassifier()
}

threshold = float(input("Enter the Threshold Value: "))
model_selected_list = []

for name, model in models.items():
    model.fit(X_Train, Y_Train)
    Y_Test_Pred = model.predict(X_Test)
    _, _, _, _, _, _, r_a_score_test = evaluate_score(Y_Test, Y_Test_Pred)
    if r_a_score_test > threshold:
        model_selected_list.append(model)

# 5. HYPERPARAMETER TUNING

randomcv_models = []

for model in model_selected_list:
    if isinstance(model, LogisticRegression):
        log_params = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [0, 0.25, 0.5, 0.75, 1]
        }
        randomcv_models.append(('LogisticRegression', LogisticRegression(), log_params))

    elif isinstance(model, SVC):
        svc_params = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'degree': [2, 3, 4, 5]
        }
        randomcv_models.append(('SVC', SVC(), svc_params))

    elif isinstance(model, DecisionTreeClassifier):
        dtc_params = {
            'max_depth': [5, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 10],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_features': ['sqrt', 'log2']
        }
        randomcv_models.append(('DecisionTreeClassifier', DecisionTreeClassifier(), dtc_params))

    elif isinstance(model, RandomForestClassifier):
        rf_params = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False],
            'max_features': ['sqrt', 'log2']
        }
        randomcv_models.append(('RandomForestClassifier', RandomForestClassifier(), rf_params))

    elif isinstance(model, AdaBoostClassifier):
        ada_params = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.5, 1],
            'base_estimator': [DecisionTreeClassifier(max_depth=d) for d in [1, 2, 3]]
        }
        randomcv_models.append(('AdaBoostClassifier', AdaBoostClassifier(), ada_params))

    elif isinstance(model, GradientBoostingClassifier):
        grad_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.6, 0.8, 1],
            'max_features': ['sqrt', 'log2']
        }
        randomcv_models.append(('GradientBoostingClassifier', GradientBoostingClassifier(), grad_params))

    elif isinstance(model, XGBClassifier):
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1],
            'gamma': [0.1, 0.3],
            'reg_alpha': [0.1, 1],
            'reg_lambda': [1.5, 2]
        }
        randomcv_models.append(('XGBClassifier', XGBClassifier(), xgb_params))

# Run Randomized Search and store best models
model_params = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(model, param_distributions=params, n_iter=100, verbose=2, n_jobs=-1)
    random.fit(X_Train, Y_Train)
    model_params[name] = random.best_params_

for model in model_selected_list:
    model_name = type(model).__name__
    if model_name in model_params:
        model.set_params(**model_params[model_name])

model_tuned = {type(model).__name__: model for model in model_selected_list}


# 6. FINAL EVALUATION AND VISUALIZATION
model_scores = {'Model': [], 'Accuracy': [], 'F1 Score': [], 'ROC-AUC': []}

for name, model in model_tuned.items():
    model.fit(X_Train, Y_Train)
    y_pred = model.predict(X_Test)
    model_scores['Model'].append(name)
    model_scores['Accuracy'].append(accuracy_score(Y_Test, y_pred))
    model_scores['F1 Score'].append(f1_score(Y_Test, y_pred))
    model_scores['ROC-AUC'].append(roc_auc_score(Y_Test, y_pred))

# Convert scores to DataFrame and sort by ROC-AUC
scores_df = pd.DataFrame(model_scores).sort_values(by='ROC-AUC', ascending=False)
print(scores_df)

# Plot the model comparison
plt.figure(figsize=(12, 6))
metrics = ['Accuracy', 'F1 Score', 'ROC-AUC']
for i, metric in enumerate(metrics):
    plt.subplot(1, 3, i+1)
    sns.barplot(x='Model', y=metric, data=scores_df, palette='viridis')
    plt.xticks(rotation=45)
    plt.title(f'{metric} Comparison')
    plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("model_comparison_plot.png", dpi=300, bbox_inches='tight')
plt.show()
