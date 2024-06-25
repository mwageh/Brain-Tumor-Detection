import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
import time

def calculate_fitness(X, y, feature_mask):
    selected_features = tf.boolean_mask(X, feature_mask, axis=1)
    scores = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(tf.keras.backend.sum(tf.keras.backend.log(x + 1e-8) * x, axis=0), axis=-1))(selected_features)
    return scores.numpy()

# Function to create initial population
def create_population(population_size, num_features):
    population = []
    for _ in range(population_size):
        feature_mask = [bool(random.getrandbits(1)) for _ in range(num_features)]
        population.append(feature_mask)
    return population

# Function to perform genetic feature selection
def genetic_feature_selection(X, y, num_features, population_size, generations):
    population = create_population(population_size, num_features)
    best_fitness = 0
    best_features = None

    for _ in tqdm(range(generations), desc="Genetic Feature Selection"):
        fitness_scores = []

        for feature_mask in population:
            fitness = calculate_fitness(X, y, feature_mask)
            fitness_scores.append((feature_mask, fitness))

            if fitness > best_fitness:
                best_fitness = fitness
                best_features = feature_mask

        # Select parents for crossover
        selected_parents = sorted(fitness_scores, key=lambda x: x[1], reverse=True)[:population_size // 2]
        parents = [parent[0] for parent in selected_parents]

        # Perform crossover
        offspring = []
        for _ in range(population_size - len(parents)):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            offspring_mask = []
            for bit1, bit2 in zip(parent1, parent2):
                if random.random() < 0.5:
                    offspring_mask.append(bit1)
                else:
                    offspring_mask.append(bit2)

            offspring.append(offspring_mask)

        population = parents + offspring

    return best_features

# Path to input CSV file
input_file = 'input feature file'

# Path to output CSV file
output_file = 'output selected features.csv'

# Read CSV file
data = pd.read_csv(input_file)

# Create a mapping dictionary for labels
label_map = {"no": 0, "yes": 1}

# Replace string labels with numerical labels
data['Label'] = data['Label'].replace(label_map)

# Separate features and target
X = tf.constant(data.drop('Label', axis=1).values, dtype=tf.float32)
y = tf.constant(data['Label'].values, dtype=tf.float32)

start_time = time.time()

# Apply genetic feature selection
with tf.device('/GPU:0'):
    selected_features = genetic_feature_selection(X, y, num_features=X.shape[1], population_size=50, generations=100)

end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

# Convert selected features to a NumPy array
selected_features = np.array(selected_features)

# Save selected features to a new CSV file
selected_data = pd.DataFrame(X.numpy()[:, selected_features])
selected_data.to_csv(output_file, index=False)

print("Selected Features:")
print(selected_data.head())