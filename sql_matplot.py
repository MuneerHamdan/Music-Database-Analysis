import sqlite3

conn = sqlite3.connect('test.db')
print("Opened database successfully");

conn.execute('''
CREATE TABLE IF NOT EXISTS student_info(first_name text,
                      last_name text,
                      credit integer,
                      gpa float);''')
print("Table created successfully");
# Clearing the table
conn.execute('DELETE FROM student_info;',);

# Inserting values
conn.execute("INSERT INTO student_info VALUES('Kate', 'Perry', 120, 3.3);")
conn.execute("INSERT INTO student_info VALUES('Kelvin', 'Harris', 50, 3.0);")
conn.execute("INSERT INTO student_info VALUES('Bin', 'Diesel', 250, 3.5);")
conn.execute("INSERT INTO student_info VALUES('nick', 'Cage', 22, 2.8);")
conn.execute("INSERT INTO student_info VALUES('Shawn', 'Carter', 100, 3.7);")
conn.execute("INSERT INTO student_info VALUES('Lucy', 'Lu', 200, 3.8 );")
conn.execute("INSERT INTO student_info VALUES('John', 'Senna', 0, 0.0 );")
conn.execute("INSERT INTO student_info VALUES('Syd', 'Barrett', 183, 2.8 );")
conn.execute("INSERT INTO student_info VALUES('Peter', 'Chao', 111, 2.3 );")
conn.execute("INSERT INTO student_info VALUES('Shang', 'abi', 64, 3.1 );")

conn.commit()
conn.close()

conn = sqlite3.connect('test.db')


# cursor = conn.execute(''' Your SQL Query''')


cursor = conn.execute(''' SELECT *
                          FROM student_info;''')


for row in cursor:
  print(row)
conn.close()

def retrieve_students():
    conn = sqlite3.connect('test.db')
    cursor = conn.execute('''SELECT * FROM student_info;''')

    for row in cursor:
        if row[2] < 150 and row[3] > 3.0:
            print(row)
    conn.close()

retrieve_students()

email_list = ['John[dot]Wick[at]rutgers[dot]edu',
              'Nancy@rutgers.edu.com',
              'Toby.Chavez.edu',
              'dfe.edu'
              'Steve[at]Peterson[at]rutgers[dot]edu',
              'Sydney[at]Lucas[at]rutgers[dot]edu',
              'Sydney[at][at]rutgers[dot]edu',
              'Byron.Dennis@umd.edu',
              'Nancy.Ruell@rutgers.edu',
              'Benjamin[dot]Conner[at]rutgers[dot]edu',
              'Nancy@rutgersedu',
              'dfe.edu.com',
              'dfe.edu.[]',
            ]

def valid_email_list(email_list):
    valid_emails = []

    for address in email_list:
        # Replace [dot] and [at] with . and @, then split using '.'
        email = address.replace('[dot]', '.').replace('[at]', '@')

        period = email.find('.')
        if (period != -1):
            at = email.find('@')
            if (at != -1 and at > period and email[period+1].isupper()):
                firstname = email[:period]
                lastname = email[period+1:at]
                other = email[at+1:]
                full = firstname + '.' + lastname + '@' + other
                valid_emails.append(full)
    
    return valid_emails

import pandas as pd

file_path = 'EuCitiesTemperatures.csv'
df = pd.read_csv(file_path)


averages = df.groupby('country')[['latitude', 'longitude']].mean()

averages = averages.round(2)

df['latitude'] = df['latitude'].fillna(df['country'].map(averages['latitude']))
df['longitude'] = df['longitude'].fillna(df['country'].map(averages['longitude']))

print(df)

subset_df = df[(df['latitude'] >= 40) & (df['latitude'] <= 60) & (df['longitude'] >= 15) & (df['longitude'] <= 30)]

country_counts = subset_df['country'].value_counts()

max_city_count = country_counts.max()

countries_with_max_cities = country_counts[country_counts == max_city_count].index.tolist()

for country in countries_with_max_cities:
    print(country)

grouped = df.groupby(['EU', 'coastline'])

for index, row in df.iterrows():
    if pd.isna(row['temperature']):
        region_type = (row['EU'], row['coastline'])
        average_temperature = grouped.get_group(region_type)['temperature'].mean()
        df.at[index, 'temperature'] = average_temperature


print(df)

import matplotlib.pyplot as plt

region_counts = df.groupby(['EU', 'coastline']).size().reset_index(name='count')

fig, ax = plt.subplots(figsize=(10, 6))
region_counts.plot(kind='bar', ax=ax, color=['blue', 'orange'], legend=False)

ax.set_xlabel('Region Type (EU, Coastline)')
ax.set_ylabel('Number of Cities')
ax.set_title('Number of Cities in Each Region Type')

region_labels = [f"{eu}-{coast}" for eu, coast in zip(region_counts['EU'], region_counts['coastline'])]
ax.set_xticklabels(region_labels, rotation=45, ha='right')

plt.show()

import matplotlib

unique_countries = df['country'].unique()
country_colors = matplotlib.pyplot.get_cmap('viridis', len(unique_countries))

plt.figure(figsize=(10, 6))
for country, color in zip(unique_countries, country_colors(range(len(unique_countries)))):
    country_df = df[df['country'] == country]
    plt.scatter(country_df['longitude'], country_df['latitude'], label=country, color=color, s=10)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Latitude vs Longitude for Cities')

plt.legend(bbox_to_anchor=(1, 1), loc='upper left')

plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['population'], bins=5, color='skyblue', edgecolor='black')

plt.xlabel('Population')
plt.ylabel('Number of Countries')
plt.title('Number of Countries Belonging to Each Population Group')

plt.show()

import numpy as np

grouped = df.groupby(['EU', 'coastline'])

def get_temperature_color(temperature):
    if temperature > 10:
        return 'red'
    elif temperature < 6:
        return 'blue'
    else:
        return 'orange'

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, (region_type, region_df) in zip(axs.flatten(), grouped):
    colors = region_df['temperature'].apply(get_temperature_color)

    scatter = ax.scatter(np.arange(len(region_df)), region_df['latitude'], c=colors)

    ax.set_xticks(np.arange(len(region_df)))
    ax.set_xticklabels([])

    ax.set_xlabel('City')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Scatter Plot for {region_type} Region Type')


plt.tight_layout()
plt.show()

