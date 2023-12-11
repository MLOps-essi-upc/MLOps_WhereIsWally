import pandas as pd

emissions = pd.read_csv('./emissions.csv')
total_emissions = emissions['emissions'].sum() #In kg eq. CO2
with open('total_emission.txt', 'w',  encoding='utf-8') as f:
    f.write(str(total_emissions)+' kg eq. Co2')
