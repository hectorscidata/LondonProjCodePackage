import pandas as pd
import joblib

import Pipeline_Processing as pp
import bilbio_func_class as bb
import Search_Mod_Para as sm
import cartes_incident_Stations as cm

# directory où se trouvent les données sources xslx, les csv et les pickles
directory = ''

#load csv
try:
    df = pd.read_csv(directory + "\\clean_fire_data_set3.csv", sep=';', index_col=0)
except:
    df, df_cord = pp.build_training_data_set(directory,bb.list_of_file,
                         fire_only=True, max_num_call=2, pump_order_max=2, minturnout=10, mintravel=10,
                         minattendance=30, max_speed=90, max_dist=12, min_dist=0.3, drop_grenfell=True,
                            maxturnout=180, frm_date='2015-12-31')

df.info()

#initialisation des données entrainements; test à partir du dataframe df
# choix de la variable TravelTimeSeconds ou AttendanceTimeSeconds; et choix du nmbre de classes pour la classification
s = sm.SearchModParaReg(df, y_var='TravelTimeSeconds')
classif2 = {'var': 'AttendanceTimeSeconds', 'cat': [288]}
s2 = sm.SearchModParaClassif(df, classif={'var': 'TravelTimeSeconds', 'cat': [210]})

reg = list(s.regresseur.keys())[0]
s2.classif.keys()
classif = list(s2.classif.keys())[0]

#Execution du GreadSerach et sauvegarde des données lgbm, xgb ou gbr
try:
    para_reg = joblib.load(directory + f'para_{reg}_reg_{s.y_var}.pkl')
    mod_reg = joblib.load(directory + f'mod_{reg}_reg_{s.y_var}.pkl')
except:
    print(f"Paramètre du modele  {reg} pour la variable {s.y_var} non sauvegardé, patience on les calculs")
    para_reg, mod_reg = s.search_par(reg)
    joblib.dump(para_reg, directory + f'para_{reg}_reg_{s.y_var}.pkl')
    joblib.dump(mod_reg, directory + f'mod_{reg}_reg_{s.y_var}.pkl')

try:
    para_classif = joblib.load(directory + f'para_{classif}_classif_{s.y_var}.pkl')
    mod_classif = joblib.load(directory + f'mod_{classif}_classif_{s.y_var}.pkl')
except:
    print(f"Paramètre du modele  {classif} pour la variable {s.y_var} non sauvegardé, patience on les calculs")
    para_classif, mod_classif = s2.search_par(classif)
    joblib.dump(para_classif, directory + f'para_{classif}_classif_{s.y_var}.pkl')
    joblib.dump(mod_classif, directory +f'mod_{classif}_classif_{s.y_var}.pkl')

sm.print_mod_info(s, mod_reg, para_reg, True)
sm.print_mod_info(s2, mod_classif, para_classif, False)
sm.print_mod_prediction(s, mod_reg, reg=True, iloc_number=2249)
sm.print_mod_prediction(s2, mod_classif,reg=False, iloc_number=2249)

sm.plot_best_reg(s, mod_reg, para_reg, reg)
sm.plot_best_class_explainer(s2, mod_classif, False)
sm.plot_explain_class(s2, mod_classif,  iloc_number=7249, jupyter=False)

df_cord = pd.read_csv(bb.datas_direc + "\\clean_fire_data_set3_cord.csv", sep=';', index_col=0)
dd = s2.datas[['TravelTimeSeconds','DeployedFromStation_Name', 'PropertyCategory']
                ].merge(df_cord, right_index=True, left_index=True)

m = cm.build_maps(dd,'Chelsea','TravelTimeSeconds')
m.show_in_browser()
