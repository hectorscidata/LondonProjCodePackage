import pandas as pd
import numpy as np
import scipy.stats as stats
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer

import LondonProject.Pipeline_Processing as pp


def target_classif(df, VarToCat, cat=None):
    if cat is None:
        cat = [210]

    def get_class(VarToCat):
        for i, c in enumerate(cat):
            if i == 0:
                if VarToCat <= c:
                    return f'c{i +1 }'
                elif len(cat) == 1:
                    return f'c{i + 2}'
            elif i+1 == len(cat):
                if VarToCat > c:
                    return f'c{i + 2 }'
                elif VarToCat <= c and VarToCat > cat[i-1]:
                    return f'c{i + 1}'
            else:
                if VarToCat <= c and VarToCat > cat[i-1]:
                    return f'c{i + 1}'

    df['target'] = df[VarToCat].apply(get_class)
    df = df.drop([VarToCat], axis=1)
    df = df.rename(columns={'target': VarToCat})
    return df


class SearchModParaReg:
    def __init__(self, df, y_var='TravelTimeSeconds', covid_out=True, train_size=0.8, scale_datas=True):
        if y_var == 'AttendanceTimeSeconds':
            df = df.assign(AttendanceTimeSeconds=df.TravelTimeSeconds + df.TurnoutTimeSeconds)
            df = df.drop(['TravelTimeSeconds', 'TurnoutTimeSeconds'], axis=1)
        self.datas = df
        self.y_var = y_var
        self.covide_out = covid_out
        self.train_size = train_size
        self.scale_datas = scale_datas
        self.X_train, self.y_train,self.X_test, self.y_test = \
            pp.build_xtrain_from_clean_dataset(self.datas, self.y_var, covid_out, train_size, scale_datas)

        self.regresseur = {'lgbm': [LGBMRegressor(),  {
                                        'num_leaves': [20, 31, 50, 100],
                                        'learning_rate': [0.01, 0.0025, 0.05, 0.0075, 0.1],
                                        'n_estimators': [100, 250, 500, 750, 1000],
                                        'max_depth': [-1, 1, 10, 20]
                                                }
                ],
        'gbr': [GradientBoostingRegressor(), {'n_estimators': [100, 500, 1000],
                                            'learning_rate': [0.01, 0.05, 0.1],
                                            'max_depth': [3, 5, 10],
                                            'subsample': [0.8, 1.0],
                                            'min_samples_split': [2, 5, 10]
                                             }
                ],
        'xgb': [XGBRegressor(objective='reg:squarederror'),{'n_estimators': [100, 500, 1000],
                                                            'learning_rate': [0.01, 0.05, 0.1],
                                                            'max_depth': [3, 5, 10],
                                                            'subsample': [0.8, 1.0],
                                                            'colsample_bytree': [0.8, 1.0]
                                                            }
                ]

        }

    def search_par(self, reg='lgbm', scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1):
        grid_search = GridSearchCV(estimator=self.regresseur[reg][0], param_grid=self.regresseur[reg][1],
                                   scoring=scoring, cv=cv, verbose=verbose, n_jobs=- n_jobs)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_, grid_search.best_estimator_

    def y_pred(self, mod):
        return mod.predict(self.X_test)

    def mse(self, mod):
        return mean_squared_error(self.y_test, self.y_pred(mod))

    def r2(self, mod):
        return r2_score(self.y_test, self.y_pred(mod))

    def rmse(self, mod):
        return root_mean_squared_error(self.y_test, self.y_pred(mod))

    def mae(self, mod):
        return mean_absolute_error(self.y_test, self.y_pred(mod))


class SearchModParaClassif:
    def __init__(self, df, covid_out=True, train_size=0.8, scale_datas=True, classif=None):
        if classif is None:
            classif = {'var':'TravelTimeSeconds', 'cat': [210]}
        if classif['var'] == 'AttendanceTimeSeconds':
            df = df.assign(AttendanceTimeSeconds=df.TravelTimeSeconds + df.TurnoutTimeSeconds)
            df = df.drop(['TravelTimeSeconds', 'TurnoutTimeSeconds'], axis=1)
        self.datas = target_classif(df, classif['var'], classif['cat'])
        self.y_var = classif['var']
        self.covide_out = covid_out
        self.train_size = train_size
        self.scale_datas = scale_datas
        self.X_train, self.y_train, self.X_test, self.y_test = \
            pp.build_xtrain_from_clean_dataset(self.datas, self.y_var, covid_out, train_size, scale_datas)
        self.classif = {'lgbm': [LGBMClassifier(random_state=42),
                                 {'n_estimators': [50, 75, 100, 125, 200],
                                  'learning_rate': [0.01, 0.1, 0.2, 0.3],
                                  'max_depth': [3, 4, 5, 6, 7]
                                  }
                                 ],
                        'logreg': [LogisticRegression(class_weight='balanced', max_iter=1000, solver='saga',
                                                      multi_class='multinomial'),
                                   {'logreg__C': stats.uniform(0.001, 10),  # Inverse of regularization strength
                                    'logreg__penalty': ['l1', 'l2', 'elasticnet', None],
                                    'logreg__l1_ratio': stats.uniform(0, 1)  # Only used when penalty='elasticnet'
                                    }
                                   ]
                        }

    def search_par(self, class_mod='lbgm', scoring='accuracy', cv=5, verbose=1, n_jobs=-1):
        grid_search = GridSearchCV(self.classif[class_mod][0], self.classif[class_mod][1], cv=cv, scoring=scoring,
                                   verbose=verbose, error_score='raise', n_jobs=n_jobs)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_, grid_search.best_estimator_

    def y_pred(self, mod):
        return mod.predict(self.X_test)

    def cm(self, mod):
        return pd.crosstab(self.y_test, self.y_pred(mod), rownames=['Classe réelle'], colnames=['Classe prédite'])


def print_mod_info(s, mod, para, regression=True):
    print(s.X_train.isnull().sum().sum(), s.y_train.isnull().sum())
    print(s.X_train.shape, s.X_test.shape, s.y_train.shape, s.y_test.shape)
    # Affichage des meilleurs paramètres
    print("Meilleurs paramètres :", para)
    # Évaluation sur le jeu de test
    if regression:
        print("MSE sur le jeu de test :", s.mse(mod))
        print('Coefficient de détermination du modèle :', mod.score(s.X_train, s.y_train))
        print(f"R² : {s.r2(mod):.3f}")
        print(f"RMSE : {s.rmse(mod):.3f}")
        print(f"MAE : {s.mae(mod):.3f}")
    else:
        print("Accuracy :", accuracy_score(s.y_test, s.y_pred(mod)))
        print("Rapport de classification :\n", classification_report(s.y_test, s.y_pred(mod)))


def plot_best_reg(s, mod, para, reg):
    fig, axes = plt.subplots(3, 2, figsize=(16, 8))
    fig.suptitle(f'{reg}Regressor, Meilleurs paramètres :, {para} R² {mod.score(s.X_train, s.y_train)}')
    sns.histplot(ax=axes[0,0], x=s.datas[s.y_var], kde=True)
    axes[0, 0].set_title(f"{s.y_var} Distribution")

    axes[0, 1].scatter(s.y_test, s.y_pred(mod), color='#980a10', s=15)
    axes[0, 1].plot((s.y_test.min(), s.y_test.max()), (s.y_test.min(), s.y_test.max()))
    axes[0, 1].set_title("y_test predictions")

    pred_train = mod.predict(s.X_train)
    residus = pred_train-s.y_train
    axes[1, 0].scatter(s.y_train, residus,  color='#980a10', s=15)
    axes[1, 0].plot((s.y_train.min(), s.y_train.max()), (0, 0), lw=3, color='#0a5798')
    axes[1, 0].set_title('Pred_Train Y train')

    residus_norm = (residus-residus.mean())/residus.std()
    stats.probplot(residus_norm, plot=axes[1, 1])
    axes[1, 1].set_title('Residus Norm')

    axes[2, 0].hist(residus, bins=30, edgecolor='k')
    axes[2, 0].set_title("Distribution des résidus")
    axes[2, 0].set_xlabel("Résidu")
    axes[2, 0].set_ylabel("Fréquence")

    importances = mod.feature_importances_
    features = s.X_train.columns  # ou une liste de noms si tu n’utilises pas un DataFrame
    # Tri des features par importance
    indices = np.argsort(importances)[::-1]
    # Plot
    axes[2, 1].set_title(f"Feature importances ({reg})")
    axes[2, 1].bar(range(len(importances)), importances[indices], align="center")
    axes[2, 1].set_xticks(range(len(importances)), [features[i] for i in indices], rotation=45, ha='right')
    #axes[2, 1].set_tight_layout()


def print_mod_prediction(s, mod, para, regression=True, iloc_number=2249):
    single_sample = pd.DataFrame(s.X_test.iloc[iloc_number:iloc_number + 1])
    predicted_class = mod.predict(single_sample)
    inter = s.datas[s.datas.index == single_sample.index[0]]

    if reg:
        print("intervention :", single_sample.index[0], "le ", inter.DateOfCall.iloc[0],
              " temps intervention", inter[s.y_var].iloc[0],
              "Le temps prédit :", predicted_class[0])
    else:
        print("intervention :", single_sample.index[0], "le ", inter.DateOfCall.iloc[0], " temps intervention",
              inter[s.y_var].iloc[0],
              "Classe prédite :", predicted_class[0])


def plot_best_class_explainer(s2, mod_classif):
    # 3. Création d'un explainer SHAP
    explainer = shap.TreeExplainer(mod_classif)
    shap_values = explainer.shap_values(s2.X_test)
    shap.summary_plot(shap_values, s2.X_test, plot_type="bar")
    # 4. Visualisation de l’explication pour un échantillon
    shap.initjs()
    try:
        shap.force_plot(explainer.expected_value[0], shap_values[0][0], s2.X_test.iloc[0])
    except:
        shap.force_plot(explainer.expected_value, shap_values[0], s2.X_test.iloc[0])


def plot_explain_class(s2, mod_classif, iloc_number=2249):
    # 3. Initialiser l'explainer LIME
    explainer = LimeTabularExplainer(s2.X_train,
                                     feature_names=mod_classif.feature_name_,
                                     class_names=mod_classif.classes_,
                                     discretize_continuous=False)

    # 4. Choisir une instance à expliquer
    i = iloc_number
    exp = explainer.explain_instance(s2.X_test.iloc[i], mod_classif.predict_proba, num_features=4)

    # 5. Affichage textuel
    print("Classe prédite :", [mod_classif.predict([s2.X_test.iloc[i]])[0]])
    exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
    plt.show()


