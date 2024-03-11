#######################################################
# Import necessary libraries                          #
#######################################################
import wikipedia
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from nltk.sem import Expression
from nltk.inference import ResolutionProver


#######################################################
#  Initialise AIML agent
#######################################################
import aiml
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

########################################################
# Load the trained model                               #
########################################################
model = load_model('fruit_classifier_model.h5')

# Define a mapping from predicted class indices to human-readable labels
label_map = {
    0: 'Apple Braeburn',
    1: 'Apple Granny Smith',
    2: 'Apricot',
    3: 'Avocado',
    4: 'Banana',
    5: 'Blueberry',
    6: 'Cactus fruit',
    7: 'Cantaloupe',
    8: 'Cherry',
    9: 'Clementine',
    10: 'Corn',
    11: 'Cucumber Ripe',
    12: 'Grape Blue',
    13: 'Kiwi',
    14: 'Lemon',
    15: 'Limes',
    16: 'Mango', # Mapping indices to fruit names
    17: 'Onion White',
    18: 'Orange',
    19: 'Papaya',
    20: 'Passion Fruit',
    21: 'Peach',
    22: 'Pear',
    23: 'Pepper Green',
    24: 'Pepper Red',
    25: 'Pineapple',
    26: 'Plum',
    27: 'Pomegranate',
    28: 'Potato Red',
    29: 'Raspberry',
    30: 'Strawberry',
    31: 'Tomato',
    32: 'Watermelon'
}

# Function to preprocess images
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = np.array(img)
    if img.shape[2] == 4:  # Check for RGBA and convert to RGB
        img = img[..., :3]
    img = img/255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img



######################################################
# Loading the Q/A csv file - task a                  #
######################################################

def load_qa_kb(filepath):
    qa_kb = []
    with open(filepath, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            qa_kb.append((row['Question'].lower(), row['Answer']))  # Convert questions to lowercase for case-insensitive matching
    return qa_kb

# Load the Q/A kb
qa_kb = load_qa_kb('mybot-basicExtra.csv')

######################################################
# Initialise TF-IDF Vectorizer                       #
######################################################

vectorizer = TfidfVectorizer()
question_texts = [q for q, _ in qa_kb]
vectorizer.fit(question_texts)  # Train the vectorizer on your questions

######################################################
# Finding the most similar question                  #
######################################################

# Function to find the most similar question in the kb to the user's query
def find_most_similar_question(query, qa_kb, vectorizer):
    query_vec = vectorizer.transform([query.lower()])
    kb_vecs = vectorizer.transform(question_texts)
    similarities = cosine_similarity(query_vec, kb_vecs).flatten()
    most_similar_idx = similarities.argmax()
    return qa_kb[most_similar_idx][1]  # Return the answer of the most similar question


# Initialize NLTK's logic parser
read_expr = Expression.fromstring


######################################################
# Task B                                             #
######################################################

# Function to load a logical kb from a CSV file
def load_kb(file_path):
    kb = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            # Assuming each row has a single column with the FOL expression
            kb.append(read_expr(row[0]))
    return kb

# Load the logical kb
kb = load_kb("logical-kb.csv")
csv_file_path = "logical-kb.csv"

# Function to check the integrity of the kb
def check_kb_integrity(kb):
    for statement in kb:
        if ResolutionProver().prove(statement, kb[:kb.index(statement)] + kb[kb.index(statement) + 1:]):
            print("Contradiction found in KB with statement:", statement)
            return False
    return True

# Function to add a new fact to the kb and save it to the CSV file
def add_fact(kb, fact, file_path):
    if not check_kb_integrity(kb + [fact]):  # Check for contradictions with the new fact
        print("Cannot add due to contradiction.")
    else:
        kb.append(fact)
        print("Ok, I will remember that.")
        save_fact_to_csv(fact, file_path)

# Function to save a new fact to the CSV file
def save_fact_to_csv(fact, file_path):
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([fact])

# Function to check a fact against the kb
def check_fact(kb, fact):
    if ResolutionProver().prove(fact, kb):
        return "Correct"
    elif ResolutionProver().prove(fact.negate(), kb):
        return "Incorrect"
    else:
        return "Sorry, I don't know"


#######################################################
# Welcome user
#######################################################


print("                                           ")
print("  ______          _ _     ____        _               .:'")
print(" |  ____|        (_) |   |  _ \      | |          __ :'__")
print(" | |__ _ __ _   _ _| |_  | |_) | ___ | |_      .'`  `-'  ``.")
print(" |  __| '__| | | | | __| |  _ < / _ \| __|    :             :")
print(" | |  | |  | |_| | | |_  | |_) | (_) | |_     :             :")
print(" |_|  |_|   \__,_|_|\__| |____/ \___/ \__|     :           :")
print("                                                `.__.-.__.'")
print("Welcome to this chat bot based on Fruits! Please feel free to ask questions from me!")
print("Enter 'What can you do?' to gain more information about me!")


######################################################
# Main loop                                          #
######################################################
while True:
    try:
        userInput = input("> ")
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break

    answer = kern.respond(userInput)

    # Check if the AIML response is a command and execute accordingly
    if answer == "#SPEAK#":
        print("Voice recognition activated. Please speak now...")
        userInput = recognize_speech_from_mic()  # Trigger speech recognition
        if userInput:
            # Process the recognized speech through the AIML engine again
            answer = kern.respond(userInput)
        else:
            continue  # If speech recognition did not work, prompt for input again

    elif answer == "#UPLOAD#":
        # Open a file dialog for the user to select an image
        root = tk.Tk()
        root.withdraw()
        root.call('wm', 'attributes', '.', '-topmost', True)

        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")]
        )
        root.destroy()

        if filepath:
            image = preprocess_image(filepath)
            prediction = model.predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class_label = label_map[predicted_class_index]
            print(f"Image selected: {filepath}")
            print(f"Predicted class: {predicted_class_label}")
        else:
            print("No image selected.")
        continue

    elif answer.startswith('#'):
        # Process special commands encoded in AIML responses
        params = answer[1:].split('$')
        if len(params) >= 2:
            cmd = params[0]
            fol_str = params[1]  # Directly using the FOL string from the AIML response

            if cmd == "31":  # Add fact to KB
                try:
                    fact = read_expr(fol_str)
                    add_fact(kb, fact, csv_file_path)  # Add a new fact to the kb
                except Exception as e:
                    print(f"Error processing logical expression: {e}")

            elif cmd == "32":  # Check a fact against the kb
                try:
                    fact = read_expr(fol_str)
                    result = check_fact(kb, fact)
                    print(result)
                except Exception as e:
                    print(f"Error processing logical expression: {e}")

            elif cmd == "0":
                # Exit the program
                print("Bye! Nice talking to you.")
                break

            elif cmd == "1":
                # Fetch and print a summary from Wikipedia
                try:
                    wSummary = wikipedia.summary(params[1], sentences=3, auto_suggest=False)
                    print(wSummary)
                except Exception as e:
                    print("Sorry, I do not know that. Be more specific!")

            elif cmd == "99":
                # Find and print the most similar question from the kb
                similar_answer = find_most_similar_question(userInput, qa_kb, vectorizer)
                if similar_answer:
                    print(similar_answer)
                else:
                    # Handle unknown commands
                    print("I'm not sure how to respond to that.")
        else:
            print(answer)  # Handle responses without a command prefix
    else:
        print(answer)  # Handle normal AIML responses
