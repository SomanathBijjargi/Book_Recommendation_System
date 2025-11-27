import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Multiply
from sklearn.model_selection import train_test_split

# --- 0. Load Your Preprocessed Data ---
# In a real script, you would load 'final_df.csv' if you saved it.
# For this example, let's assume 'final_df' and the counts are in memory
# from the previous step.

# Replace with your actual counts from the preprocessing script
final_df = pd.read_csv('model_ready_data.csv') # Example of loading
n_users = 11977   
n_books = 36539  
n_authors = 12080 

# --- 1. Define Model Hyperparameters ---
embedding_size = 32
mlp_layers = [64, 32, 16] # Sizes of the dense layers in the MLP path


# --- 2. Build the Hybrid NCF (NeuMF) Model ---

# Input Layers
user_input = Input(shape=(1,), name='user_input')
book_input = Input(shape=(1,), name='book_input')
author_input = Input(shape=(1,), name='author_input')

# Embedding Layers
# We create separate embeddings for the GMF path and MLP path for more flexibility
gmf_user_embedding = Embedding(input_dim=n_users, output_dim=embedding_size, name='gmf_user_embedding')(user_input)
gmf_book_embedding = Embedding(input_dim=n_books, output_dim=embedding_size, name='gmf_book_embedding')(book_input)

mlp_user_embedding = Embedding(input_dim=n_users, output_dim=embedding_size, name='mlp_user_embedding')(user_input)
mlp_book_embedding = Embedding(input_dim=n_books, output_dim=embedding_size, name='mlp_book_embedding')(book_input)
mlp_author_embedding = Embedding(input_dim=n_authors, output_dim=embedding_size, name='mlp_author_embedding')(author_input)


# Flatten the embedding outputs to vectors
gmf_user_vector = Flatten()(gmf_user_embedding)
gmf_book_vector = Flatten()(gmf_book_embedding)

mlp_user_vector = Flatten()(mlp_user_embedding)
mlp_book_vector = Flatten()(mlp_book_embedding)
mlp_author_vector = Flatten()(mlp_author_embedding)


# GMF (Generalized Matrix Factorization) Path
gmf_output = Multiply()([gmf_user_vector, gmf_book_vector])

# MLP (Multi-Layer Perceptron) Path
mlp_concat = Concatenate()([mlp_user_vector, mlp_book_vector, mlp_author_vector])

# Build MLP layers dynamically
mlp = mlp_concat
for layer_size in mlp_layers:
    mlp = Dense(layer_size, activation='relu')(mlp)
mlp_output = mlp


# Fusion (NeuMF) Layer
neumf_concat = Concatenate()([gmf_output, mlp_output])
output = Dense(1, activation='sigmoid', name='output')(neumf_concat)


# Create the Model
model = Model(
    inputs=[user_input, book_input, author_input],
    outputs=output,
    name='Hybrid_NCF_Model'
)

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')] # AUC is a great metric for this
)

model.summary()


# --- 3. Prepare Data for Training ---

# Input data is a list of arrays for each input layer
X = [
    final_df['user_idx'].values,
    final_df['book_idx'].values,
    final_df['author_idx'].values
]
# Target data
y = final_df['interaction'].values

# Split into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Unpack the list of arrays for training and validation
X_train_unpacked = [X_train[0], X_train[1], X_train[2]]
X_val_unpacked = [X_val[0], X_val[1], X_val[2]]


# --- 4. Train the Model ---
print("\nStarting model training...")

history = model.fit(
    x=X_train_unpacked,
    y=y_train,
    batch_size=256,
    epochs=10, # Start with 10 epochs, you can increase this
    verbose=1,
    validation_data=(X_val_unpacked, y_val)
)

print("\nTraining complete!")
# --- 5. Evaluate Model Performance ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
print("\nEvaluating model performance on validation set...")

# Predict probabilities
y_pred_prob = model.predict(X_val_unpacked)
# Convert to binary (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

# Compute Metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
cm = confusion_matrix(y_val, y_pred)

# Print all metrics
print("\n--- Performance Metrics ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_val, y_pred, zero_division=0))

