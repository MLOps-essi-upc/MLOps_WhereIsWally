import pandas as pd

def main():
    emissions = pd.read_csv('./emissions.csv')
    total_emissions = emissions['emissions'].sum() # In kg eq. CO2
    km_equiv = total_emissions*0.196974607 # Equivalent driven km of diesel family car

    with open('total_emission.txt', 'w') as f:
        f.write(f'{total_emissions:.2f} kg eq. Co2 \n')
        f.write(f'which is equivalent to driving {km_equiv:.2f} kilometers with a family sized diesel car.')

if __name__ == "__main__":
    main()
