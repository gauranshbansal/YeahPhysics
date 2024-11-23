import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data using Newtonian mechanics
def generate_data(num_samples=1000):
    G = 6.67430e-11  # Gravitational constant
    m1 = 5.0         # Mass of body 1
    m2 = 3.0         # Mass of body 2
    t = np.linspace(0, 10, num_samples)  # Time range (0 to 10 seconds)

    # Initial positions and velocities (2D)
    r1 = np.array([1.0, 0.0])  # Body 1 position
    r2 = np.array([0.0, 1.0])  # Body 2 position
    v1 = np.array([0.0, 0.2])  # Body 1 velocity
    v2 = np.array([-0.2, 0.0]) # Body 2 velocity

    positions = []

    for _ in t:
        # Compute gravitational force
        r = r2 - r1
        distance = np.linalg.norm(r)
        force = G * m1 * m2 / distance**2
        direction = r / distance  # Unit vector

        # Update velocities and positions using Newton's laws
        a1 = force / m1 * direction
        a2 = -force / m2 * direction

        v1 += a1 * (t[1] - t[0])
        v2 += a2 * (t[1] - t[0])

        r1 += v1 * (t[1] - t[0])
        r2 += v2 * (t[1] - t[0])

        positions.append([r1[0], r1[1]])

    return np.array(t), np.array(positions)

# Generate data
time, positions = generate_data()

# Step 2: Split data into training and testing
split_index = int(0.6 * len(time))  # 80% training, 20% testing
train_x, test_x = time[:split_index], time[split_index:]
train_y, test_y = positions[:split_index], positions[split_index:]

# Step 3: Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(2)  # Output 2D position
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Step 4: Train the model
model.fit(train_x, train_y, epochs=1000, batch_size=32, verbose=1)

# Step 5: Evaluate the model
loss, mae = model.evaluate(test_x, test_y, verbose=0)
print(f"Mean Absolute Error on Test Data: {mae:.4f}")

# Step 6: Make predictions
predictions = model.predict(test_x)

# Step 7: Plot results
plt.figure(figsize=(10, 6))
plt.plot(test_x, test_y[:, 0], label='True X', color='blue')
plt.plot(test_x, predictions[:, 0], label='Predicted X', color='red', linestyle='--')
plt.plot(test_x, test_y[:, 1], label='True Y', color='green')
plt.plot(test_x, predictions[:, 1], label='Predicted Y', color='orange', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('True vs Predicted Positions')
plt.show()
