import pandas as pd

# 1. Load your raw dataset
# Replace 'your_dataset.csv' with the actual filename of your dataset
file_path = 'steam_games_2026.csv' 
df = pd.read_csv(file_path)

print(f"Starting data cleanup pipeline...")
print(f"Initial row count: {len(df)}")

# 2. Drop rows where the game Name is missing
df = df.dropna(subset=['Name'])

# 3. Drop duplicate games based on the Name column
df = df.drop_duplicates(subset=['Name'])

# 4. Handle missing values in text features
# If a game has no genre or tags listed, fill it with a blank space
df['Primary_Genre'] = df['Primary_Genre'].fillna('')
df['All_Tags'] = df['All_Tags'].fillna('')

# 5. Build the Metadata Soup
# Combine genre and tags into a single descriptive text block for each game
df['metadata_soup'] = df['Primary_Genre'] + ' ' + df['All_Tags']

# 6. Normalize text (Convert everything to lowercase)
df['metadata_soup'] = df['metadata_soup'].str.lower()

# 7. Export the polished dataframe to a new file
output_file = 'clean_games.csv'
df.to_csv(output_file, index=False)

print("\n--- Pipeline Success ---")
print(f"Cleaned data rows retained: {len(df)}")
print(f"Successfully generated clean file: '{output_file}'")