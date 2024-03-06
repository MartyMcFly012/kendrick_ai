from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.models import model_from_json
import json
import pandas as pd

app = Flask(__name__)


text = []
with open("/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/kendrick.txt", "r", encoding="utf-8") as file:
    text += file.readlines()

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1

# Step 3: Sequencing
input_sequences = []
output_words = []
sequence_length = 5

csv_path = '/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/pos_tags_data.csv'
df = pd.read_csv(csv_path)
# Load the model
# model = load_model("/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/models/20_bidirectional.h5")
model_paths = ['/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/models/5_bidirectional.h5', '/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/models/10_bidirectional.h5', '/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/models/20_bidirectional.h5', '/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/models/50_bidirectional_earlystop_lrscheduler.h5', '/Users/marty/Desktop/Kendrick_Lamar_Lyric_Generator/models/50_bidirectional_earlystop.h5']

models = []
model_architectures = []

for model_path in model_paths:
    model = load_model(model_path)
    
    # Save model architecture to JSON
    model_json = model.to_json()
    model_architectures.append(json.loads(model_json))
    
    models.append(model)

# Function to sample the next word
def sample_next_word(seed_sequence, model, temperature=0.5):
    seed_sequence = pad_sequences([seed_sequence], maxlen=sequence_length)
    predicted_probs = model.predict(seed_sequence)[0]
    
    # Adjust the temperature for more controlled randomness
    predicted_probs = np.power(predicted_probs, 1.0 / temperature)
    predicted_probs /= predicted_probs.sum()

    next_word_id = np.random.choice(total_words, p=predicted_probs)
    return next_word_id

# Function to generate text
def generate_text(seed_text, model, num_words=50, temperature=0.5):
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_sequence = seed_sequence[:]

    for _ in range(num_words):
        next_word_id = sample_next_word(generated_sequence[-sequence_length:], model, temperature)
        generated_sequence.append(next_word_id)

    generated_lyrics = tokenizer.sequences_to_texts([generated_sequence])[0]
    
    return generated_lyrics[0].upper() + generated_lyrics[1:]

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/understanding')
def understanding():
    return render_template('understanding_the_data.html', table=df.to_html())

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user inputs from the form
        seed_text = request.form['seed_text']
        num_words = int(request.form['num_words'])
        diversity = float(request.form['diversity'])
        # model_index = int(request.form.getlist('model_index'))
        
        selected = request.form.getlist('model_index')
        # Call the text generation function with the selected model
        model_indices = [int(index) for index in selected]
        model_names = ['5 epochs', '10 epochs', '20 epochs', '50* epochs (with an early stop at 22 and learning rate scheduler)', '50 epochs']

        # Call the text generation function for each selected model
        generated_texts = {}
        for model_index in model_indices:
            model = models[model_index]
            generated_text = generate_text(seed_text, model, num_words, diversity)
            generated_texts[model_index] = generated_text
        return render_template('result_compare.html', generated_texts=generated_texts, model_names=model_names)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
