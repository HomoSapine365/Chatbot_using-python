import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential  
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON file
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Tokenizing and processing patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)  # Tokenize each word
        words.extend(w)
        documents.append((w, intent['tag']))  # Store word-pattern relationships

        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add unique classes

# Lemmatization & Sorting
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))  # Remove duplicates
classes = sorted(set(classes))

print(f"{len(documents)} documents")
print(f"{len(classes)} classes: {classes}")
print(f"{len(words)} unique lemmatized words: {words}")

# Save data to disk
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]

    # Create bag of words
    for w in words:
        bag.append(1 if w in pattern_words else 0)
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into input (X) and output (Y)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

print("Training data created!")

# Build Neural Network Model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))  # Output layer

# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save trained model
model.save('chatbot_model.h5')
print("Model loaded successfully!")