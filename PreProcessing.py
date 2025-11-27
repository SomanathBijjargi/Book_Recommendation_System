
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# --- 1. Load the Raw Data ---
# Note: You might need to adjust the file paths.
# The dataset often has encoding issues, so we use 'latin-1'.
# The separator is ';', not the usual ','.

print("Loading data...")
try:
    users = pd.read_csv('BX-Users.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
    books = pd.read_csv('BX-Books.csv', sep=';', on_bad_lines='skip', encoding='latin-1', low_memory=False)
    ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
except FileNotFoundError:
    print("Error: Make sure the CSV files (BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv) are in the same directory.")
    exit()

# Rename columns for easier access
books.columns = ['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
users.columns = ['User-ID', 'Location', 'Age']
ratings.columns = ['User-ID', 'ISBN', 'Book-Rating']

print("Data loaded successfully.")
print(f"Original ratings shape: {ratings.shape}")

# --- 2. Handle Sparsity: Filter users and books ---
print("\nFiltering data for sparsity...")

# Get user and book interaction counts
user_counts = ratings['User-ID'].value_counts()
book_counts = ratings['ISBN'].value_counts()

# Filter out users with less than 10 ratings and books with less than 5 ratings
ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= 10].index)]
ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= 5].index)]

print(f"Shape after filtering: {ratings.shape}")


# --- 3. Merge DataFrames to get a complete dataset ---
print("\nMerging dataframes...")

# Merge ratings with book information
df = pd.merge(ratings, books, on='ISBN')

# We need User-ID, ISBN, Book-Rating, Book-Author, AND Book-Title
df = df[['User-ID', 'ISBN', 'Book-Rating', 'Book-Author', 'Book-Title']]


# --- 4. Prepare Data for the Model ---

# a) Create the binary interaction label
print("\nCreating binary interaction label...")
df['interaction'] = df['Book-Rating'].apply(lambda x: 1 if x > 0 else 0)

# We only want positive interactions for this step
df_pos = df[df['interaction'] == 1].copy()

# b) Encode categorical IDs to integer indices
print("Encoding IDs to integer indices...")
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()
author_encoder = LabelEncoder()

df_pos['user_idx'] = user_encoder.fit_transform(df_pos['User-ID'])
df_pos['book_idx'] = book_encoder.fit_transform(df_pos['ISBN'])
df_pos['author_idx'] = author_encoder.fit_transform(df_pos['Book-Author'])

# Get the number of unique entities
n_users = df_pos['user_idx'].nunique()
n_books = df_pos['book_idx'].nunique()
print(f"Number of unique users: {n_users}")
print(f"Number of unique books: {n_books}")

# c) Generate Negative Samples
print("\nGenerating negative samples...")

# Create a set of all positive (user, book) interactions for quick lookup
positive_interactions = set(zip(df_pos['user_idx'], df_pos['book_idx']))

negative_samples = []
num_neg_samples_per_pos = 4 # Ratio of negative to positive samples

# Loop until we have enough negative samples
while len(negative_samples) < len(df_pos) * num_neg_samples_per_pos:
    user_idx = np.random.randint(0, n_users)
    book_idx = np.random.randint(0, n_books)
    
    # If the (user, book) pair is NOT a positive interaction, add it as a negative sample
    if (user_idx, book_idx) not in positive_interactions:
        negative_samples.append([user_idx, book_idx, 0]) # [user, book, interaction=0]

# Convert to DataFrame
neg_df = pd.DataFrame(negative_samples, columns=['user_idx', 'book_idx', 'interaction'])

# We need to add author information to the negative samples
# Create a mapping from book_idx to author_idx
book_to_author_map = df_pos.drop_duplicates(subset='book_idx').set_index('book_idx')['author_idx']
neg_df['author_idx'] = neg_df['book_idx'].map(book_to_author_map)

# Drop any negative samples where the author couldn't be found (if any)
neg_df.dropna(inplace=True)
neg_df['author_idx'] = neg_df['author_idx'].astype(int)


# --- 5. Finalize the Dataset ---
print("\nFinalizing the dataset...")

# Select only the columns needed for the model from the positive dataframe
model_df_pos = df_pos[['user_idx', 'book_idx', 'author_idx', 'interaction']]

# Concatenate positive and negative samples
final_df = pd.concat([model_df_pos, neg_df], ignore_index=True)

# Shuffle the dataset for randomness during training
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nPreprocessing complete!")
print("Final model-ready dataset head:")
print(final_df.head())
print(f"\nFinal dataset shape: {final_df.shape}")
print("\nThis 'final_df' DataFrame is now ready to be used to train your NCF model.")
print("\nSaving the final model-ready dataset to a CSV file...")

n_users = df_pos['user_idx'].nunique()
n_books = df_pos['book_idx'].nunique()
n_authors = df_pos['author_idx'].nunique() # Make sure to calculate this too
print(f"Number of unique users: {n_users}")
print(f"Number of unique books: {n_books}")
print(f"Number of unique authors: {n_authors}")

final_df.to_csv('model_ready_data.csv', index=False)
print("File 'model_ready_data.csv' saved successfully! ✅")

# --- NEW CELL TO CREATE THE LOOKUP FILE ---

print("\nCreating the book title lookup file...")

# 1. Use the 'df_pos' DataFrame which already has all the mapped info
#    (It was created in Step 4b)
try:
    # 2. Select the columns we need for our "bridge"
    lookup_df = df_pos[['book_idx', 'Book-Title', 'Book-Author']]
    
    # 3. Drop duplicates to get one row per 'book_idx'
    lookup_df = lookup_df.drop_duplicates(subset=['book_idx'])
    
    # 4. Save the file
    lookup_df.to_csv('book_title_lookup.csv', index=False, encoding='latin1')
    
    print("✅ Successfully created 'book_title_lookup.csv'.")
    print("You can now run your training notebook, then your recommendation script.")

except NameError:
    print("Error: 'df_pos' is not defined.")
    print("Please make sure you have run all the cells above this one.")
except KeyError:
    print("Error: Your 'df_pos' DataFrame is missing 'book_idx', 'Book-Title', or 'Book-Author'.")