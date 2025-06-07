"""on commencera par
    - garder les datas au delà de 31/12/2015
    - eliminer les colonnes incidents inutiles
    - retraiter les colonnes et calculer avant élimination des colonnes intermédiaires
    - Appliquer les filtres sur les colonnes : NumCalls (<3 ?), InicidentGroup FalseAlarm ?

    Puis on traitera de la même manière le fichier mobilisation

    Puis on fera le join des fichiers mob/incident et stations
    On calculera les distances
    On fera un filtre :
        - distance trop petite
        - vitesse incohérente
        - Time of Call de inicident - Time of arrival de mob pas coherent avec tempMob + temps deplacement même après 3600
        """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import bilbio_func_class as te
import holidays

# Preprocessing Hypothesis :
# - Incidents :  frm_date = '2015-12-31',drop_grenfell = True, fire_only=True, max_num_call=2
# - Mobilisation : MinTurnout = 10, MinTravel = 10, MinAttendance =30, pump_order_max=2
# - Mob_Merge_Indicdent = error = 15 seconds spred between Timecall and timerarrive + attendance


# limit incohérence entre heure d'appel d'incident + temps mob+ temps travel = timearrived +ERREUR


def load_incident_training_datas(fire_only=True, max_num_call=2, frm_date='2015-12-31', drop_grenfell=True):
    # Load_datas_incident from '2015-12-31', elimine col inutilue, retraite les colonnes entre elles
    col_to_drop = []
    df = te.load_incident_datas(frm_date, drop_grenfell)  # 1 Million
    nb_raw = len(df)
    """rappel propercase = Borough_Name et Inc_geo_WarName =Ward sous zone du borough"""
    # Filtrage
    # - 1 Numcall<3 (96%)  : pour apprendre pas de cas exceptionnel avec plein d'appels
    # - 2 pas False Alarm - Malicious (off 2%) et peu gener l'eval
    # -3  incident_datas[incident_datas['FirstPumpArriving_DeployedFromStation'].isnull()] # 58K

    df_drop = df[(df.NumCalls > max_num_call) | (df['FirstPumpArriving_DeployedFromStation'].isnull()) |
            (df.IncidentDescription.str.contains('Malicious'))]
    df = df[~ df.index.isin(df_drop.index)]
    if max_num_call < 3:
        col_to_drop = ['NumCalls']
    print("Share of the orignal kept after Call : ", str((len(df) / nb_raw) * 100))

    if fire_only:
        df_fire = df[df.IncidentGroup != 'Special Service']
        print('Share of Fire incidents ', str((len(df_fire)/nb_raw) * 100))
    else:
        df_fire = df

    col_to_drop += ['IncidentGroup']  # on retire l'info False Alarm de l'apprentissage connu après seulement

    # IncidentStationGround != FirstPumpArriving_DeployedFromStation
    # on va mettre un flag 0/1 avant de sortir ces 2 colonnes
    df_fire = df_fire.assign(First_arrive_is_Incident_station=
                                df_fire['FirstPumpArriving_DeployedFromStation'] == df_fire['IncidentStationGround'])  # 300K

    # on retirera après merge avec incident and station
    col_to_drop += ['FirstPumpArriving_DeployedFromStation', 'IncidentStationGround']

    # on va la transformer en residential = dwelling + other residential puis outdoor/outdoorStructure;
    # puis non residential et road vehicule; on vire les autres de l'apprentissage

    propcat = {'Dwelling': 'Residential', 'Other Residential': 'Residential', 'Outdoor Structure': 'Outdoor'}
    df_fire = df_fire.replace({"PropertyCategory": propcat})
    df_fire = df_fire[df_fire.PropertyCategory.isin(['Residential', 'Outdoor', 'Non Residential', 'Road Vehicle'])]

    print("Share of the orignal kept for training : ", str((len(df_fire) / nb_raw) * 100))
    return df_fire, col_to_drop


def load_mob_training_datas(pump_order_max=2, minturnout=10, mintravel=10, minattendance=30, maxturnout=180):
    # Load des mob datas avec incident_number en index et colonne drop et filtre sur delay et homestation
    mob_datas = te.load_mobilisation_datas()
    nb_mob = len(mob_datas)

    # PumpOrder = 1 ou 0 si pas numéro 1
    # pour l'apprentissage on se limite à la première pump arrivée/mobilisée, dans la ligne du 1 call, etc...
    mob_datas = mob_datas[mob_datas.PumpOrder <= pump_order_max]

    # verif la cohérence de temps : time of call - hour arrived = temps mob + temps delay
    # verif la cohérence dans l'espace = FirstPumpArriving_Fromstation = DEployedFromStation =incidentgroundsation
    outlayer_mob = mob_datas[(mob_datas.TurnoutTimeSeconds < minturnout) | (mob_datas.TurnoutTimeSeconds > maxturnout) |
                             (mob_datas.TravelTimeSeconds < mintravel) |
                             (mob_datas.AttendanceTimeSeconds < minattendance)]

    mob_datas = mob_datas[~ mob_datas.index.isin(outlayer_mob.index)]
    print("Share of the orignal mobilisations kept for training : ", str((len(mob_datas) / nb_mob) * 100))
    return mob_datas, outlayer_mob


def merge_incident_mob(incident_datas, mob_datas, error=15):
    # on merge les incidents et les mob en inner sur incidentnumber
    # puis on nettoie à nouveau des outlayers et des invraissemeblances
    # puis on utilisera comme clés : ResourceMobilisationId

    df = incident_datas.join(mob_datas, lsuffix='_mob', rsuffix='_incident', how='inner')

    # prob si firstpump arriving & pas =1 from station et pumporder=1; ça ressemble à un retard ou un pb de saisie
    # on va éviter ce genre de datas pour entrainer le modèle
    anomalie1 = df[(df.FirstPumpArriving_DeployedFromStation != df.DeployedFromStation_Name) & (df.PumpOrder == 1)]

    # on fait ensuite un filtre sur le temps : incoherence entre heure d'appel t heure de mobilised
    # suggère des erreurs de prise de note même après correction des 3600 pour 1 heure
    df = df.assign(TimeCheck=(df.DateAndTimeMobilised - pd.to_datetime(df['DateOfCall'].dt.date.astype(str) + ' '
                                                                       + df['TimeOfCall'])).astype('timedelta64[s]'))
    df['TimeCheck'] = df['TimeCheck'].apply(lambda x: x if x >= 0 else np.abs(x + 3600))
    anomalie2 = df[df.TimeCheck > error]

    df = df[~ df.index.isin(anomalie1.index)]
    df = df[~ df.index.isin(anomalie2.index)]

    # on a l'heure d'appel, les temps en seconds de l'intervention et on a déjà vérifié la cohérence
    df_to_drop = ['DateAndTimeMobilised', 'DateAndTimeMobile', 'DateAndTimeArrived', 'TimeCheck']

    #df['incident_number'] = df.index
    df = df.set_index(['ResourceMobilisationId'])

    return df, df_to_drop, anomalie1, anomalie2


def merge_stations(incident_mob, maxspeed=90, max_dist=12, min_dist=0.3):
    # on merge avec les stations sur le DeployedFromStation de mobilisation
    # on filtre les données invraissemblalbles ou extreme pour l'apprentissage
    # les distances sont KM et les vitesses en KM/H
    stations = te.load_station()
    # nb_station_by_borough = stations[['borough', 'name']].groupby(['borough']).count()
    incident_mob['ResourceMobilisationId'] = incident_mob.index
    df = incident_mob.merge(stations, left_on='DeployedFromStation_Name', right_on='name')
    nb_mob_inc = len(df)
    # Creation d'une variable qui vaut 1 si le Borough de l'incident est celui de la pump Arrive
    df = df.assign(Is_Same_Borough=df['ProperCase'] == df['borough'])
    df_to_drop = ['ProperCase', 'borough', 'IncGeo_WardName']
    # on calcule les distances de haversine
    df = df.assign(deployed_haversine_dist=lambda x: te.haversine_dist([df.latitude, df.longitude],
                                                                       [df.Latitude, df.Longitude]))
    df = df.assign(speed=df.deployed_haversine_dist / df.TravelTimeSeconds * 3600)
    df = df.set_index('ResourceMobilisationId')

    suspect = df[(df.speed > maxspeed) | (df.deployed_haversine_dist > max_dist)
                 | (df.deployed_haversine_dist < min_dist)]  # 42K enregistrement pas bcp surement une erreur



    # une fois les distances calculées et le pb du choix de la station trop loin ; pas le bon borough??
    df_to_drop += ['Latitude', 'Longitude', 'latitude', 'longitude', 'name']

    print("Share of incident mobilisation selected kept for training : ", str((len(df) / nb_mob_inc) * 100))
    return df, df_to_drop, suspect


def build_date_hours_cat(df, col_d='DateOfCall', col_h='HourOfCall'):
    """on va passer les dates en categories: saison, is_week_end uoi/non et séparer les heures en
    22h00/06h00 nuit; 6/10 matin; '10-16 midi; 16/22 soir l'idée est de trouer des cat pertinentes
    pour les calculs de temps de transport sachant que nb_jours * heures * mois = 7 *24*12 soit + de 2000 categories
    là on va etre sur 2 * 4 * 4 = 32"""
    df[col_d] = pd.to_datetime(df[col_d])

    # 1. Déterminer la saison
    def get_saison(date):
        mois = date.month
        jour = date.day
        if (mois == 12 and jour >= 21) or (1 <= mois <= 2) or (mois == 3 and jour < 20):
            return 'hiver'
        elif (mois == 3 and jour >= 20) or (4 <= mois <= 5) or (mois == 6 and jour < 21):
            return 'printemps'
        elif (mois == 6 and jour >= 21) or (7 <= mois <= 8) or (mois == 9 and jour < 22):
            return 'été'
        else:
            return 'automne'

    df['season'] = df[col_d].apply(get_saison)

    # 2. Est-ce le week-end ?
    uk_holidays = holidays.UnitedKingdom()
    df['is_week_end'] = df[col_d].apply(lambda d: 1 if d.weekday() >= 5 or d.date() in uk_holidays else 0).astype(bool)

    # 3. Moment de la journée
    def get_moment_journee(heure):
        if 6 <= heure < 10:
            return 'morning'
        elif 10 <= heure < 16:
            return 'noon'
        elif 16 <= heure < 22:
            return 'evening'
        else:
            return 'night'

    df['day_moment'] = df[col_h].apply(get_moment_journee)
    return df


def build_training_data_set(fire_only=True, max_num_call=2, pump_order_max=2, minturnout=10, mintravel=10,
                            minattendance=30, max_speed=90, max_dist=12, min_dist=0.3, drop_grenfell=True,
                            maxturnout=180, frm_date='2015-12-31'):
    # Time in Seconds, distance ein KM and speed in Kph

    # construction des datasets à parit des données TXT en appliquant les paramètres de la function
    print("Loading & Friltering Incidents datas")
    df_inc, c_inc_drop = load_incident_training_datas(fire_only, max_num_call, frm_date, drop_grenfell)
    print("Loading and Filtering Mobilisations datas")
    df_mob, time_outlayer = load_mob_training_datas(pump_order_max, minturnout, mintravel, minattendance, maxturnout)
    print("Merging Incidents & Mobilisations, checking for artefacts")
    df_all, c_merge_to_drop, pump_an, timecheck = merge_incident_mob(df_inc, df_mob)
    print("Merging with Stations, Calculating distance and Filtering Outlayers")
    df, c_df_to_drop, suspect = merge_stations(df_all, max_speed, max_dist, min_dist)
    df = df[~ df.index.isin(suspect.index)]

    # attendance = travel + mobilised donc compo linéaire
    c_drop = c_inc_drop + c_merge_to_drop + c_df_to_drop + ['AttendanceTimeSeconds']
    c_drop += ['speed']  # ces données ne seront connues qu'après l'intervention
    c_drop += ['TimeOfCall', 'CalYear']  # on a l'heure et on a fait les verif de cohérences entre incident et Mob

    # Enrichissement de la donnée de temps: On garde le mois,le jour de la semaine et l'heure
    df['DateOfCall'] = pd.to_datetime(df['DateOfCall'])
    df['MonthOfCall'] = df['DateOfCall'].dt.month
    df['WeekDayOfCall'] = df['DateOfCall'].dt.dayofweek
    #on ajoute le Timeofcall à dateof call pour synthetiser la donnée
    df['DateOfCall'] = pd.to_datetime(df['DateOfCall'].astype(str) + ' ' + df['TimeOfCall'].astype(str))
    #tri du dataset par date/heure pour la chrono
    df = df.sort_values(by=['DateOfCall'], ascending=True)

    #creation des categories de temps: saison, week_end et moment de la journée
    df = build_date_hours_cat(df)
    #suppresion des données permettant de faire les cat tempo sachant que ces 3 là trop nombreuses
    c_drop +=['MonthOfCall', 'WeekDayOfCall', 'HourOfCall']

    df = df.drop(c_drop, axis=1)
    # reorg colonne
    col_order = ['DateOfCall', 'season', 'is_week_end', 'day_moment',
                 'IncidentDescription', 'PropertyCategory', 'NumCalls',
                 'PumpOrder', 'Is_Same_Borough', 'First_arrive_is_Incident_station', 'DeployedFromStation_Name',
                 'TurnoutTimeSeconds', 'deployed_haversine_dist', 'TravelTimeSeconds']
    if not 'NumCalls' in df.columns:
        col_order.remove('NumCalls')
    df = df.drop_duplicates()
    return df[col_order]
#df.to_csv("clean_fire_data_set3.csv", sep=";")
# to do : date confinement covid et date jour ferie genre paques : change day of week to sunday ?


def build_xtrain_from_clean_dataset(dataset, y_var='TravelTimeSeconds', covid_out=True,
                                    train_size=0.8, scale_datas=True):
    if covid_out:
        datas = dataset[(dataset.DateOfCall < '2020-03-20') | (dataset.DateOfCall > '2021-03-31')]
    else:
        datas = dataset
    if y_var == 'AttendanceTimeSeconds' and 'TravelTimeSeconds' in datas.columns:
        datas = datas.assign(AttendanceTimeSeconds=datas.TravelTimeSeconds + datas.TurnoutTimeSeconds)
        datas = datas.drop(['TravelTimeSeconds', 'TurnoutTimeSeconds'], axis=1)

    # on tri les datas par ordre de date et heure car serie temporelle
    datas = datas.sort_values(by=['DateOfCall'], ascending=True)
    #  on drop les datas inutiles
    datas = datas.drop(['DeployedFromStation_Name', 'DateOfCall'], axis=1)
    datas['PumpOrder'] = datas['PumpOrder'].astype(str)
    # on dummies les variables catégorielles
    data_dummies = pd.get_dummies(datas, prefix=['Incident'], columns=['IncidentDescription'])
    data_dummies = pd.get_dummies(data_dummies, prefix=['PropType'], columns=['PropertyCategory'])
    if 'season' in datas.columns:
        data_dummies = pd.get_dummies(data_dummies, prefix=['s'], columns=['season'])
        data_dummies = pd.get_dummies(data_dummies, prefix=['day'], columns=['day_moment'])
        data_dummies = pd.get_dummies(data_dummies, prefix=['PumpOrder'], columns=['PumpOrder'])

    # on sépare les var explicatives de la variable cible
    X = data_dummies.drop([y_var], axis=1)
    y = data_dummies[y_var]

    # Comme on a une serie temporelle on fait un split temporel; on a mis les données en ordre de temps
    offset = int(X.shape[0] * train_size)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    # on normalise les datas numériques, pas les dummies
    if scale_datas:
        scaler = StandardScaler()
        col_norm = list(datas.drop([y_var], axis=1).select_dtypes('number').columns)
        X_train[col_norm] = pd.DataFrame(scaler.fit_transform(X_train[col_norm]), index=X_train.index)
        X_test[col_norm] = pd.DataFrame(scaler.transform(X_test[col_norm]), index=X_test.index)

    return X_train, y_train, X_test, y_test