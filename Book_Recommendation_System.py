import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import warnings

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')

# --- 1. Load All Artifacts ---
print("Loading model and artifacts...")
books_df = None # Initialize books_df
try:
    # Load the trained Keras model
    model = tf.keras.models.load_model('best_enhanced_ncf_model.h5')
    
    # Load the saved scaler
    scaler = joblib.load('standard_scaler.joblib')
    
    # Load the metadata CSVs
    book_metadata_df = pd.read_csv('book_metadata.csv', index_col='book_idx', encoding='latin1', on_bad_lines='skip')
    user_metadata_df = pd.read_csv('user_metadata.csv', index_col='user_idx', encoding='latin1', on_bad_lines='skip')
    
    # Load the original interactions to know what to exclude
    final_df = pd.read_csv('model_ready_data.csv', encoding='latin1', on_bad_lines='skip')
    
    
    # --- NEW: Load the 'book_title_lookup.csv' file ---
    # This file maps 'book_idx' to 'Book-Title' and 'Book-Author'.
    try:
        books_df = pd.read_csv('book_title_lookup.csv', encoding='latin1')
        
        # Check for the required columns
        if 'book_idx' not in books_df.columns or \
           'Book-Title' not in books_df.columns or \
           'Book-Author' not in books_df.columns:
            
            print("\n[Warning] 'book_title_lookup.csv' is missing required columns.")
            books_df = None
        else:
            print("✅ Original book titles loaded.")
            
    except FileNotFoundError:
        print("\n[Warning] 'book_title_lookup.csv' not found.")
        print(">>> Please run 'Cell 8' in your training notebook to create this file.")
        print("Recommendations will be shown with IDs only.")
        books_df = None
    
    print("✅ All artifacts loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: Could not find a necessary file. {e}")
    print("Please make sure all .h5, .joblib, and .csv files are in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading artifacts: {e}")
    exit()


# --- 2. The Recommendation Function (No changes) ---

def get_recommendations(user_idx, top_n=10):
    """
    Generates book recommendations for a given user_idx.
    """
    print(f"\nGenerating recommendations for user_idx: {user_idx}")
    
    # --- Step A: Get User's Features ---
    try:
        user_activity = user_metadata_df.loc[user_idx]['user_activity']
    except KeyError:
        print(f"Warning: user_idx {user_idx} not found in metadata. "
              "Using average activity for prediction.")
        user_activity = user_metadata_df['user_activity'].mean()

    # --- Step B: Generate "Candidate" Books ---
    all_book_indices = set(book_metadata_df.index)
    interacted_books = set(final_df[final_df['user_idx'] == user_idx]['book_idx'])
    candidate_books_indices = list(all_book_indices - interacted_books)
    
    if not candidate_books_indices:
        print("User has already interacted with all known books.")
        return pd.DataFrame(columns=['book_idx', 'predicted_score'])
        
    print(f"Found {len(candidate_books_indices)} candidate books to score.")
    
    # --- Step C: Prepare Model Inputs ---
    n_candidates = len(candidate_books_indices)
    user_input_array = np.full(n_candidates, user_idx)
    book_input_array = np.array(candidate_books_indices)
    
    candidate_metadata = book_metadata_df.loc[candidate_books_indices]
    author_input_array = candidate_metadata['author_idx'].values
    
    numeric_features_df = candidate_metadata[['book_popularity', 'book_avg_rating']].copy()
    numeric_features_df['user_activity'] = user_activity 
    
    expected_feature_order = ['user_activity', 'book_popularity', 'book_avg_rating']
    numeric_features_df = numeric_features_df[expected_feature_order]
    scaled_numeric_features = scaler.transform(numeric_features_df)
    
    # --- Step D: Predict Scores ---
    model_inputs = [
        user_input_array,
        book_input_array,
        author_input_array,
        scaled_numeric_features
    ]
    
    print("Predicting scores for all candidate books...")
    predicted_scores = model.predict(model_inputs, batch_size=1024, verbose=0).flatten()
    
    # --- Step E: Rank and Return Top N ---
    results_df = pd.DataFrame({
        'book_idx': candidate_books_indices,
        'predicted_score': predicted_scores
    })
    
    top_recommendations = results_df.sort_values(by='predicted_score', 
                                                 ascending=False).head(top_n)
    
    top_recommendations = top_recommendations.reset_index(drop=True)
    
    return top_recommendations

# --- 3. Example Usage (Now Corrected) ---
if __name__ == "__main__":
    
    test_user_id = int(input("Enter a user ID to get recommendations: "))
    
    # Get the recommendations (this is the DataFrame with 'book_idx')
    recommendations = get_recommendations(test_user_id, top_n=10)

    print(f"\n--- Top 10 Recommendations for User {test_user_id} ---")
    
    # --- FINAL: Make the output human-readable ---
    if books_df is not None:
        
        # --- This merge is now simple and correct ---
        # It merges recommendations['book_idx'] with books_df['book_idx']
        readable_recommendations = pd.merge(
            recommendations,
            books_df,
            on='book_idx' # Both files share this key
        )
        
        # Select and reorder the columns for a clean print
        final_output = readable_recommendations[[
            'Book-Title',
            'Book-Author',
            'predicted_score'
        ]]
        
        # Set the index to start from 1 for a "Top 10" list
        final_output.index = final_output.index + 1
        
        print(final_output)
        
    else:
        # Fallback if we couldn't load the book titles
        print(recommendations)