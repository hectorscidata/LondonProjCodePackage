"""Importation des données xlsx et définition des fonctions de claculs de distance et
de transformation de coordonnées"""

import pandas as pd
import openpyxl
import csv
import os
import numpy as np
#import pyproj
from pyproj import Transformer
"""Dans cette library on loge les functions de loading et preprocessing de datas:
        - il y a l'adresse et la localisation de la raw data : to do passer au dowload direct
        - La transforamtion de la data XLSX en CSV
        - Le liste des colonnes à éliminer pour les appels et les interventions
        - La transformation des coordonnées eAnglaise en systeme international
        - L'élimination des colonnes concernent celles non pertinentes et celles dont l'information ne sera disponible
                que post interbvention donc pas intéresssante en entrainement"""

#adresse des datas xls
""" https://data.london.gov.uk/dataset/london-fire-brigade-incident-records"""
""" https://data.london.gov.uk/dataset/london-fire-brigade-mobilisation-records"""


datas_direc = r'C:\Python\LondonProject\LondonDatas'
list_of_file = ['\\LFB Incident data from 2018 - November 2024', '\\LFB Incident data from 2009 - 2017',
                '\\LFB Mobilisation data 2021 - 2024', '\\LFB Mobilisation data from 2015 - 2020',
                '\\LFB Mobilisation data from January 2009 - 2014']
datas_store = datas_direc + """\\LFB Incident data from 2018 - November 2024.xlsx"""


#pas discriminant ou missing
incident_col_to_drop = ['FRS','Easting_m','Northing_m','FRS','Postcode_full', 'Notional Cost (£)']
#les datas post Call pas pertinente
incident_col_to_drop += ['Postcode_district','PropertyType','AddressQualifier','PumpCount'
                           ,'UPRN','USRN','IncGeo_WardCode', 'IncGeo_WardNameNew','IncGeo_BoroughCode'
                            , 'IncGeo_BoroughName']
incident_col_to_drop +=['PumpMinutesRounded','NumPumpsAttending','NumStationsWithPumpsAttending',
                'SecondPumpArriving_AttendanceTime','SecondPumpArriving_DeployedFromStation',
                'FirstPumpArriving_AttendanceTime']
#sans interet ou postecall
mobilsation_col_to_drop = ['PerformanceReporting',  'CalYear','HourOfCall',
                          'DeployedFromStation_Code','DateAndTimeReturned','DateAndTimeLeft','Resource_Code',
                           'PlusCode_Description', 'DelayCode_Description']
#trop de missing
mobilsation_col_to_drop += ['BoroughName','WardName']

#les évenements de la tour grenfell par leur ampleur ont perturbé les datas et l'organisation des secours
grenfell_start='2017-06-14'
grenfell_end='2017-06-30'

def xlsx_to_csv(directory, files_to_convert):
    wb = openpyxl.load_workbook(directory + files_to_convert + '.xlsx', data_only=True)
    sh = wb.active  # was .get_active_sheet()
    with open(directory + files_to_convert + """.csv""", 'w', newline="", encoding='utf-8') as f:
        c = csv.writer(f, dialect='excel')
        for row in sh:
           c.writerow([cell.value for cell in row])


def convert_datas_to_csv(directory, files_to_convert):
    for ff in files_to_convert:
        if not ff.replace('\\', '') + '.csv' in os.listdir(directory):
            print(ff + '.xlsx', ' to convert')
            xlsx_to_csv(directory, ff)

def haversine_dist(source, target):
    #calcul en KM du de la distance entre source et target
    dlat = np.radians(target[0]) - np.radians(source[0])
    dlon = np.radians(target[1]) - np.radians(source[1])
    a = np.sin(dlat / 2) ** 2 + np.cos( np.radians(source[0])) * np.cos(np.radians(target[0])) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 6373

    return c * r

class DatasQuality:
    #créée les fonctions d'examen d'un pd dataframe en data_quality

    def __init__(self, datas, metadatas=None):
        if type(datas) is str:
            self.datas = pd.read_csv(datas, sep=',', low_memory=False)
        else:
            self.datas = datas
        if type(metadatas) is str:
            self.metadatas = pd.read_excel(datas_direc + '\\' + metadatas.replace('.xlsx','') + '.xlsx', index_col=0)
        else:
            self.metadatas = None

    def first_des(self):
        first_ana = pd.DataFrame([self.datas.isnull().sum(axis=0)/len(self.datas)*100], ['Pct_Null']).T
        first_ana['Nb_Values'] = 0
        first_ana['Data_Type'] = 'Object'
        for i in range(len(first_ana.index)):
            first_ana.loc[first_ana.index[i], 'Nb_Values'] = len(self.datas[first_ana.index[i]].value_counts())
            first_ana.loc[first_ana.index[i], 'Data_Type'] = self.datas[first_ana.index[i]].dtype.name
        if len(self.metadatas) > 0:
            first_ana = first_ana.join(self.metadatas, how='outer')
        return first_ana

    def concat(self, datas_to_add):
        self.datas = pd.concat([self.datas, datas_to_add.datas], ignore_index=True)
        return self

    def prop_cat_var(self, variables):
        return self.datas[variables].value_counts(normalize=True)

    def drop_col(self, col_to_drop):
        return self.datas.drop(col_to_drop, axis=1, inplace=True)

#from pyproj constant to calculate from English to international coordianate
#bng = pyproj.Proj(init='epsg:27700')
#wgs84 = pyproj.Proj(init='epsg:4326')

def rounded_easting_to_lat_lon(df_inc):
    """prend un pandas transorm les easting_rounded et northing_rounded
       replace les nan lont/lat, ajoute une col flag rouned et drop les col rounded"""

    if not 'Easting_rounded' in df_inc.columns or not 'Northing_rounded' in df_inc.columns:
        print("Please make sure your dataframe has a colum Northing_rounded and a column Easting_rounded")
        return df_inc
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326")
    #df_inc['roundedLon'], df_inc['roundedLat'] = pyproj.transform(bng, wgs84, df_inc['Easting_rounded'].tolist(),
    #                                                              df_inc['Northing_rounded'].tolist())
    #par rapport à l'ancienne version inversion de coordonnées
    df_inc['roundedLat'], df_inc['roundedLon'] = transformer.transform(df_inc['Easting_rounded'].tolist(),
                                                                  df_inc['Northing_rounded'].tolist())

    # les lat/long à 0 sont mis à null
    df_inc['Latitude'].replace(0, np.nan, inplace=True)
    df_inc['Longitude'].replace(0, np.nan, inplace=True)
    # une colonne is_rouned est ajoutée
    df_inc['IsRounded'] = df_inc.Latitude.isnull()
    # les null sont remplacés par les lat/lon calculées à partir des rounded
    df_inc['Latitude'] = df_inc['Latitude'].fillna(df_inc['roundedLat'])
    df_inc['Longitude'] = df_inc['Longitude'].fillna(df_inc['roundedLon'])

    return df_inc.drop(['Easting_rounded', 'Northing_rounded', 'roundedLat', 'roundedLon'], axis=1)

def load_all_raw_incident_datas():
    # load excel datas
    dd_incident = DatasQuality(datas_direc + list_of_file[0] + '.csv', 'Metadata')
    dd_incident2 = DatasQuality(datas_direc + list_of_file[1] + '.csv', 'Metadata')
    return dd_incident.concat(dd_incident2)

def load_all_raw_mobilisation_datas():
    dd_mob = DatasQuality(datas_direc + list_of_file[2] + '.csv', 'Mobilisations Metadata')
    dd_mob2 = DatasQuality(datas_direc + list_of_file[3] + '.csv', 'Mobilisations Metadata')
    dd_mob3 = DatasQuality(datas_direc + list_of_file[4] + '.csv', 'Mobilisations Metadata')

    dd_mob = dd_mob.concat(dd_mob2)
    return dd_mob.concat(dd_mob3)

def load_incident_datas(frm_date='2015-12-31', grenfell_drop = True):

    #recup datas dans un df
    df = load_all_raw_incident_datas().datas

    #drop des colonnes
    df = df.drop(incident_col_to_drop, axis=1)
    # passage de la date en date_time
    df['DateOfCall'] = pd.to_datetime(df['DateOfCall'])
    #selection de la satrt_date
    df = df[df.DateOfCall > pd.to_datetime(frm_date)]
    if grenfell_drop:
        df = df[(df.DateOfCall < pd.to_datetime(grenfell_start)) |
                     (df.DateOfCall > pd.to_datetime(grenfell_end))]

    #defintion incident_number en index
    df = df.set_index('IncidentNumber')

    #transformation des données de easting en lat lont et drop des colonnes
    df = rounded_easting_to_lat_lon(df)

    #fusion des colonnes special_service et stopcodedes
    df['IncidentDescription'] = df['SpecialServiceType'].fillna(
        df['StopCodeDescription'])
    df.drop(['SpecialServiceType', 'StopCodeDescription'], axis=1, inplace=True)
    return df

def load_mobilisation_datas():
    # recup datas dans un df
    df = load_all_raw_mobilisation_datas().datas
    # drop des colonnes
    df = df.drop(mobilsation_col_to_drop, axis=1)

    #drop des mob ou TrunoutTimesSeconds or TravelTimeSeconds is null
    df = df.dropna(subset=['TurnoutTimeSeconds', 'TravelTimeSeconds'], axis=0)

    # defintion incident_number en index
    df = df.set_index('IncidentNumber')

    #filtrage de lignes
    df = df[df.PlusCode_Code == 'Initial']  #99.6%
    # DeployedFromLocation on garde Home et on vire la colonne
    df = df[df.DeployedFromLocation == 'Home Station']  # 96.6%
    # on garde les mobilisations where delaycode is NULL pour apprentissage après on drop la colon
    df = df[df.DelayCodeId.isnull()]
    # on drop les colonnes qui ont permis la selection des lignes
    df = df.drop(['PlusCode_Code', 'DelayCodeId', 'DeployedFromLocation'], axis=1)
    #passage des date en datetime pandas
    df['DateAndTimeArrived'] = pd.to_datetime(df['DateAndTimeArrived'])
    df['DateAndTimeMobilised'] = pd.to_datetime(df['DateAndTimeMobilised'])
    df['DateAndTimeMobile'] = pd.to_datetime(df['DateAndTimeMobile'])

    return df

def load_station():
    stations = pd.read_csv(datas_direc + '\\stations.csv')
    stations['borough'] = stations.borough.str.replace('&nbsp;', '').str.replace('<br>', '')
    return stations[['name', 'latitude', 'longitude', 'borough']]

"""Comme les fichiers xlslx sont très long à ouvrir à pandas, on va les passer en csv avant de commencer"""

