from flask import Flask, request, render_template
from Model import SpellCheckerModule
import string
from spellchecker import SpellChecker
import nltk
import pickle
import os
import re
from io import BytesIO
import base64
app = Flask(__name__)
spell_checker_module = SpellCheckerModule()

# routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spell', methods=['POST', 'GET']) # methods is mandatory here
def spell():
    if request.method == 'POST': 
        text = request.form['text']
        corrected_text = spell_checker_module.correct_spell(text)
        corrected_grammar, _ = spell_checker_module.correct_grammar(text)
        return render_template('index.html', corrected_text=corrected_text, corrected_grammar=corrected_grammar)

@app.route('/grammar', methods=['POST', 'GET'])
def grammar():
    if request.method == 'POST':
        file = request.files['file']
        readable_file = file.read().decode('utf-8', errors='ignore')
        corrected_file_text = spell_checker_module.correct_spell(readable_file)
        corrected_file_grammar, _ = spell_checker_module.correct_grammar(readable_file)
        return render_template('index.html', corrected_file_text=corrected_file_text, corrected_file_grammar=corrected_file_grammar)



spell = SpellChecker()

# Load the trained SVR model
with open('mod_svr', 'rb') as file:
    svr_model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    paragraph = request.form['paragraph']

    # Tokenize the paragraph into sentences and words
    sentences = nltk.sent_tokenize(paragraph)
    words = nltk.word_tokenize(paragraph)

    # Count the number of sentences, words, and characters
    num_sentences = len(sentences)
    num_words = len(words)
    num_chars = sum(len(word) for word in words if word not in string.punctuation)

    # Find misspelled words
    misspelled = spell.unknown(words)
    num_misspelled = len(misspelled)

    # Predict score using the SVR model
    X_predict = [[num_chars, num_words, num_sentences, num_misspelled]]
    score = svr_model.predict(X_predict)[0]
    return render_template('result.html', score=score)

import csv

@app.route('/upload', methods=['POST'])
def upload():
    # Get uploaded files
    files = request.files.getlist('file')

    # List to store grades for each file along with filenames
    file_grades = []

    for file in files:
        # Read the file content
        
        paragraph = file.read().decode("utf-8")

        # Tokenize the paragraph into sentences and words
        sentences = nltk.sent_tokenize(paragraph)
        words = nltk.word_tokenize(paragraph)

        # Count the number of sentences, words, and characters
        num_sentences = len(sentences)
        num_words = len(words)
        num_chars = sum(len(word) for word in words if word not in string.punctuation)

        # Calculate the number of spelling mistakes (you can use your SpellCheckerModule here)
        num_misspelled = 0  # Replace this with your logic to count spelling mistakes

        # Predict score using the SVR model
        X_predict = [[num_chars, num_words, num_sentences, num_misspelled]]
        score = svr_model.predict(X_predict)[0]
        score = round(score, 2)
        if score < 2 :
            f = 'Fail'
        elif score > 2 and score < 5:
            f = 'Can do better'
        elif score > 5 and score < 8:
            f = 'Good'
        else:
            f = 'Excellent'
        # Add file name and grade to the list
        file_grades.append({'filename': file.filename, 'grade': score , 'Feedback':f})

    # Append data to the CSV  file
    csv_filename = 'file_grades.csv'
    file_exists = os.path.exists(csv_filename)
    with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['filename', 'grade','Feedback']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for item in file_grades:
            writer.writerow({'filename': re.sub('[^\w\s]', '', item['filename']), 'grade': item['grade'], 'Feedback':item['Feedback']})

    return render_template('result.html', file_grades=file_grades)
'''
# pie chart
@app.route('/pie', methods=['POST'])
def pie():
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv('file_grades.csv')

    # Count the number of students based on feedback
    feedback_counts = df['feedback'].value_counts()

    # Plotting the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(feedback_counts, labels=feedback_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title('Feedback Distribution')

    # Save the plot as a PNG file
    filename = 'pie_chart.png'
    plt.savefig(filename)
    
    return render_template('result.html', pie_chart_url=filename)'''
# python main
if __name__ == "__main__":
    app.run(debug=True)
