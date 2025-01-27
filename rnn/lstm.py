import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os  # Make sure this is at the top with other imports

# Check if GPU is available
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Read data from slot files
def read_slot_file(slot_number):
    filename = f'slot{slot_number}'  # Using @ prefix for the slot files
    abs_path = os.path.abspath(filename)
    print(f"Attempting to open file: {abs_path}")
    try:
        with open(filename, 'r') as f:
            numbers = [int(line.strip()) for line in f if line.strip()]
        print(f"Successfully read {len(numbers)} numbers from {filename}")
        return np.array(numbers)
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
        raise
    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        raise

# Load data for all slots
slot_data = {}
for slot in range(1, 7):
    slot_data[slot] = read_slot_file(slot)

print(f"\nUsing {len(slot_data[1])} total draws for training")

# Function to create sequences for training
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Function to create and train model for each slot
def train_slot_model(slot_data, slot_number, max_value):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    slot_scaled = scaler.fit_transform(slot_data.reshape(-1, 1))
    
    # Create sequences
    sequence_length = 10  # Increased sequence length for better long-term memory
    X, y = create_sequences(slot_scaled, sequence_length)
    
    # Display first 5 sequences and their targets
    print(f"\nFirst 5 sequences for Slot {slot_number}:")
    print("----------------------------------------")
    for i in range(5):
        original_sequence = scaler.inverse_transform(X[i].reshape(-1, 1)).flatten()
        original_target = scaler.inverse_transform(y[i].reshape(-1, 1)).flatten()[0]
        print(f"Sequence {i+1}: {original_sequence} → Target: {original_target}")
    print("----------------------------------------")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape input data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build the LTMS model with attention mechanism
    inputs = tf.keras.Input(shape=(sequence_length, 1))
    
    # First LSTM layer with return sequences
    lstm1 = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    attention1 = tf.keras.layers.Attention()([lstm1, lstm1])
    
    # Second LSTM layer
    lstm2 = tf.keras.layers.LSTM(64, return_sequences=True)(attention1)
    attention2 = tf.keras.layers.Attention()([lstm2, lstm2])
    
    # Third LSTM layer
    lstm3 = tf.keras.layers.LSTM(32)(attention2)
    
    # Dense layers with dropouts
    dense1 = tf.keras.layers.Dense(64, activation='relu')(lstm3)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
    
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1)(dropout2)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])
    
    # Train the model with increased epochs and patience
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            )
        ],
        verbose=1
    )
    
    # Make prediction for the next number
    last_sequence = slot_scaled[-sequence_length:]
    last_sequence = last_sequence.reshape((1, sequence_length, 1))
    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted_number = scaler.inverse_transform(predicted_scaled)[0][0]
    
    # Round and constrain prediction to valid range
    predicted_number = round(predicted_number)
    predicted_number = max(1, min(predicted_number, max_value))
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    
    # Save the model
    model.save(f'ltms_slot{slot_number}_predictor.h5')
    
    return predicted_number, test_loss, test_mae, history

# Define max values for each slot
max_values = {1: 29, 2: 38, 3: 49, 4: 49, 5: 49, 6: 49}

# Train models and make predictions for all slots
predictions = []
print("\nTraining models and making predictions for all slots...")

for slot_num in range(1, 7):
    predicted_number, test_loss, test_mae, history = train_slot_model(
        slot_data[slot_num], 
        slot_num,
        max_values[slot_num]
    )
    predictions.append(predicted_number)
    
    print(f"\nSlot {slot_num} Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Predicted number: {predicted_number}")
    
    # Plot training history for each slot
    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Training History for Slot {slot_num}')
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Display final predictions for all slots
print("\nFinal Predictions for Next Draw:")
print("--------------------------------")
print(f"Slot 1: {predictions[0]}")
print(f"Slot 2: {predictions[1]}")
print(f"Slot 3: {predictions[2]}")
print(f"Slot 4: {predictions[3]}")
print(f"Slot 5: {predictions[4]}")
print(f"Slot 6: {predictions[5]}")
print("--------------------------------")
