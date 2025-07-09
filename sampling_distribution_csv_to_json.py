import pandas as pd
import json

# Load CSV file to be converted here
df = pd.read_csv('sampling distribusi.csv', skiprows=1)

df = df.rename(columns={df.columns[0]: 'Provinsi' })

# Translate provinces from Indonesian to English
province_translation = {
    'Maluku': 'Maluku',
    'Lampung': 'Lampung',
    'Jawa Barat': 'West Java',
    'Banten': 'Banten',
    'Kalimantan Selatan': 'South Kalimantan',
    'Bali': 'Bali',
    'DKI Jakarta': 'Jakarta',
    'Sulawesi Selatan': 'South Sulawesi',
    'Jawa Timur': 'East Java',
    'Sulawesi Utara': 'North Sulawesi',
    'Nusa Tenggara Barat': 'West Nusa Tenggara',
    'Sumatera Utara': 'North Sumatra',
    'Sumatera Barat': 'West Sumatra',
    'Sumatera Selatan': 'South Sumatra',
    'Kep. Bangka Belitung': 'Bangka Belitung',
    'Kalimantan Barat': 'West Kalimantan',
    'Kalimantan Timur': 'East Kalimantan',
    'Jawa Tengah': 'Central Java',
    'Riau': 'Riau',
    'DI Yogyakarta': 'Yogyakarta',
    'Kalimantan Tengah': 'Central Kalimantan',
    'Gorontalo': 'Gorontalo',
    'Sulawesi Tengah': 'Central Sulawesi',
    'Sulawesi Tenggara': 'Southeast Sulawesi',
    'Papua Barat': 'West Papua',
    'Nusa Tenggara Timur': 'East Nusa Tenggara',
    'Kepulauan Riau': 'Riau Islands',
    'Jambi': 'Jambi',
    'Bengkulu': 'Bengkulu',
    'Maluku Utara': 'North Maluku',
    'Sulawesi Barat': 'West Sulawesi',
    'Aceh': 'Aceh',
    'Nasional': 'National'
}

df['Provinsi'] = df['Provinsi'].map(province_translation)

# Define mapping for region
region_map = {
    'Java': ['Jakarta', 'West Java', 'Central Java', 'East Java', 'Yogyakarta', 'Banten'],
    'Sumatra': ['Aceh', 'North Sumatra', 'West Sumatra', 'Riau', 'Jambi', 'South Sumatra', 
                'Bengkulu', 'Lampung', 'Bangka Belitung', 'Riau Islands'],
    'Kalimantan': ['West Kalimantan', 'Central Kalimantan', 'South Kalimantan', 'East Kalimantan', 
                          'North Kalimantan'],
    'Bali-Nusa': ['Bali', 'West Nusa Tenggara', 'East Nusa Tenggara'],
    'Sulampua': ['North Sulawesi', 'Central Sulawesi', 'South Sulawesi', 'Southeast Sulawesi', 'Gorontalo', 
                 'West Sulawesi','Maluku', 'North Maluku', 'Papua', 'West Papua']
}

# Impute provinces with missing values with the national value
for col in df.columns:
    df[col] = df[col].fillna(df[col].iloc[0])

# Drop national row
df = df.drop(0)

# Define categories
expenditure_cols = ['1-2 Juta', '2.1-3 Juta', '3.1-4 Juta', '4.1-5 Juta', '5.1-6 Juta', '6.1-7 Juta', '7.1-8 Juta', '> 8 Juta']
age_cols = ['20-30 thn', '31-40 thn', '41-50 thn', '51-60 thn', '> 60 thn']
education_cols = ['SMA', 'Diploma', 'Sarjana', 'Pasca Sarjana']
education_labels = ['Senior High School', 'Diploma', "Bachelor's Degree", "Master's Degree"]

# Clean columns of whitespace
df.columns = df.columns.str.strip()

result = {}

for _, row in df.iterrows():
    prov = row['Provinsi']

    for key, val in region_map.items():
        if prov in val:
            region = key
    
    result[prov] = {
        'region': region_map.get(prov, region),
        'expenditure': [(row[col], col) for col in expenditure_cols],
        'age': [(row[col], col.replace(' thn', '')) for col in age_cols],
        'education': [(row[col], label) for col, label in zip(education_cols, education_labels)]
    }

# Save as JSON
with open("converted_distribution.json", "w", encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

# Print results
print(json.dumps(result, indent=4, ensure_ascii=False))
