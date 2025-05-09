# Poetic-Text-Generation-with-LSTM
This project implements a text generation model using an LSTM neural network in TensorFlow/Keras. It generates poetic text based on a training dataset, such as Shakespeare's plays, and allows you to experiment with different "temperature" values to control the randomness of the generated text.

Features
Preprocessing of a text dataset.
Training an LSTM-based model for sequence prediction.
Generating text with adjustable creativity (temperature).
Saving and reusing the trained model.
Dataset
The dataset used is Shakespeare's text, downloaded directly from TensorFlow's storage: Shakespeare Dataset

Installation and Setup
Clone this repository:

git clone https://github.com/yourusername/poetic-text-generation.git
cd poetic-text-generation
Install required libraries:

pip install tensorflow numpy
Run the script to train the model:

python Poetic_text_generation.py
File Structure
Poetic_text_generation.py: Main script for preprocessing, training, and generating text.
textgenerator.keras: Trained model saved in Keras format.
README.md: Project documentation.
Usage
1. Training the Model
The script preprocesses the dataset, trains an LSTM model, and saves the trained model as textgenerator.keras.

2. Generating Text
The script provides a function to generate text:

generate_text(length, temperature)
length: Number of characters to generate.
temperature: Controls randomness in text generation. A lower temperature generates more predictable text, while a higher temperature generates more random text.
Example outputs for different temperatures:

Temperature 0.2:
thou art a villain, a coward, a slave, and a
Temperature 1.0:
thou hrtnpotie - deliebly ur' heawife, thou art thou!
Code Highlights
Model Architecture
A simple sequential LSTM model:
model = Sequential([
    LSTM(128, input_shape=(SEQ_LENGTH, len(characters))),
    Dense(len(characters)),
    Activation('softmax')
])
Sampling Function
Adjusts randomness in text generation:
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))
Results
After training, the model can generate coherent poetic text similar to Shakespeare's style. Experiment with the temperature value to observe changes in creativity.
