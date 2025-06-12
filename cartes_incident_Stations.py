# Coordonnées des points (latitude, longitude) et leurs labels

import folium
from folium.plugins import BeautifyIcon
import pandas as pd

cat_colors = {'Residential':'blue', 'Outdoor':'green', 'Road Vehicle':'purple', 'Non Residential':'black'}
legend_html = """
<div style="
    position: fixed;
    bottom: 50px;
    left: 50px;
    width: 150px;
    background-color: white;
    border:2px solid grey;
    z-index:9999;
    font-size:14px;
    padding: 10px;
    ">
<b>Légende</b><br>
● = Rapide<br>
■ = Lent<br><br>
<span style="color:black;">■</span> Non Residential<br>
<span style="color:blue;">■</span> Residential<br>
<span style="color:green;">■</span> Outdoor<br>
<span style="color:purple;">■</span>Road Vehicle
</div>
"""


def build_maps(dd, borough='Chelsea', y_var='TravelTimeSeconds'):
    # datas doit contenir les noms des stations et leuurs latitude/longitude
    # il doit egalement constenir les incidents que l'on veut montrer avec leur Latitude/Longitude
    # attention minuscuule pour les stations et majuscules pour les lieux

    # 1. Carte centrée sur Londres
    m = folium.Map(location=[51.50, -0.10], zoom_start=10)

    # 2. Ajout des Brigades
    data = dd[['DeployedFromStation_Name','latitude','longitude']].drop_duplicates()
    # Ajouter les points sur la carte
    for i in range(0,len(data)):
        folium.Marker(
            location=[data.iloc[i]['latitude'], data.iloc[i]['longitude']],
            popup=data.iloc[i]['DeployedFromStation_Name'],  # Affiche le label lorsque tu cliques sur un point
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)


    if 'PropertyCategory' in dd.columns:
        data_inc = dd[dd.DeployedFromStation_Name==borough][
                        [y_var,'PropertyCategory','Longitude', 'Latitude']]

        def add_marker(row):
            color = cat_colors.get(row['PropertyCategory'], 'gray')
            loc = [row['Latitude'], row['Longitude']]
            #popup = f"{row['PropertyCategory']} – {row[y_var]}"
            if row[y_var] == 'c1':
                folium.CircleMarker(
                    location=loc,
                    radius=6,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.9,
                    popup=f"{row['PropertyCategory']} - {row[y_var]}"
                ).add_to(m)
            else:  # c2
                folium.Marker(
                    location=loc,
                    icon=BeautifyIcon(
                        icon='■',
                        icon_shape='rectangle',
                        border_color=color,
                        background_color=color,
                        text_color='white'
                    ),
                    #popup=f"{row['PropertyCategory']} - {row[y_var]}"
                    popup=folium.Popup(f"{row['PropertyCategory']} - {row[y_var]}", parse_html=True)
                ).add_to(m)

        data_inc.apply(add_marker, axis=1)
        #m.get_root().header.add_child(folium.Element(font_awesome))
        m.get_root().html.add_child(folium.Element(legend_html))
        folium.LayerControl().add_to(m)
    else:
        data_inc = dd[dd.DeployedFromStation_Name == 'Chelsea'][
            [y_var, 'Longitude', 'Latitude']]
        for i in range(0,len(data)):
            folium.Marker(
                location=[data_inc.iloc[i]['Latitude'], data_inc.iloc[i]['Longitude']],
                popup=data_inc.iloc[i]['TravelTimeSeconds'],  # Affiche le label lorsque tu cliques sur un point
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(m)

    return m

# Sauvegarder la carte sous forme de fichier HTML
#m.save("carte_classes.html")
#m.show_in_browser()




#Légende & contrôle de calques
#folium.LayerControl().add_to(m)
#m.show_in_browser()