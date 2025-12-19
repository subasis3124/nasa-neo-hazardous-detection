from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

SCORING = {
    'f1': 'f1',
    'recall': 'recall',
    'precision': 'precision',
    'roc_auc': 'roc_auc',
    'average_precision': 'average_precision'
}


PARAMS = [
    ('Logistic Regression', LogisticRegression(), {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'C': [0.1, 0.3, 0.5, 0.7, 0.9],
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 300, 500, 700, 900]
    }),

    ('Decision Tree Classifier', DecisionTreeClassifier(), {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [x for x in range(1, 10, 2)],
        'min_samples_split': [x for x in range(2, 10, 2)],
        'min_samples_leaf': [x for x in range(1, 10, 3)],
        'max_features': ['sqrt', 'log2', None]
    }),

    ('Random Forest Classifier', RandomForestClassifier(), {
        'n_estimators': [x*100 for x in range(1, 10, 2)],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [x for x in range(3, 10, 2)],
        'min_samples_split': [x for x in range(2, 10, 2)],
        'min_samples_leaf': [x for x in range(1, 10, 3)],
        'max_features': ['sqrt', 'log2', None]
    }),

    ('Gradient Boosting Classifier', GradientBoostingClassifier(), {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [x/10 for x in range(1, 10, 3)],
        'n_estimators': [x*100 for x in range(1, 10, 2)],
        'subsample': [x/10 for x in range(1, 10, 3)],
        'criterion': ['friedman_mse', 'squared_error'],
        'max_depth': [x for x in range(3, 10, 2)],
        'max_features': ['sqrt', 'log2']
    }),

    ('K Nearest Neighbors Classifier', KNeighborsClassifier(), {
        'n_neighbors': [x for x in range(1, 10, 3)],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [x for x in range(30, 100, 20)],
        'p': [x for x in range(2, 10, 3)]
    }),

    ('Extra Trees Classifier', ExtraTreesClassifier(), {
        'n_estimators': [100, 200, 500, 1000],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2', None]
    }),

    ('AdaBoost Classifier', AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    }),

    ('Catboost Classifier', CatBoostClassifier(
        eval_metric="F1",
        loss_function="Logloss",
        verbose=0
    ), {
        "iterations": [x*100 for x in range(1, 10, 2)],
        "depth": [x for x in range(0, 10, 2)],
        "learning_rate": [0.01, 0.05, 0.1, 0.001, 0.005],
        "l2_leaf_reg": [x for x in range(1, 10, 2)],
        "border_count": [32, 64, 128],
        "bagging_temperature": [0, 1, 0.5, 2],
        "random_strength": [0.5, 1, 2, 5],
        "class_weights": [[1, 5], [1, 10], [1, 20]]
    }
    ),

    ('XGB Classifier', XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    ), {
        "n_estimators": [200, 500, 1000],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 0.1, 0.5, 1],
        "scale_pos_weight": [1, 5, 10, 20]
    }
    )
]
