import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
#loading the model
model=pickle.load(open('savedmodel.sav','rb'))

# Ensure the CSV file is correctly loaded
data = pd.read_csv('Cleaned_data.csv')
# Strip leading and trailing spaces from the 'location' column
data['location'] = data['location'].apply(lambda x: x.strip() if isinstance(x, str) else x)



# Print the columns to ensure 'location' exists
print(data.columns)

@app.route('/')
def index():
   locations = pd.DataFrame(data)['location'].unique()

# Define a custom sorting function
   def custom_sort(location):
      # Convert float values to strings
      location = str(location)
      # Split the location name into text and numerical parts
      parts = location.split()
      text_part = ''.join(filter(str.isalpha, parts[0]))  # Extract alphabetic characters
      num_part = ''.join(filter(str.isdigit, parts[0]))  # Extract numerical characters
      return (text_part, int(num_part or 0))  # Convert numerical part to integer, handle empty strings

# Sort the locations using the custom sorting function
   locations = sorted(locations, key=custom_sort)
   prediction=''
   return render_template('index.html', **locals())


def preprocess_input(location):
    if location not in data['location'].unique():
        location = 'other'
    return location

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
       location = request.form['location']
       bhk = int(request.form['bhk'])
       bath = int(request.form['bath'])
       total_sqft = float(request.form['total_sqft'])
       # Dummy prediction logic, replace with actual model
       location = preprocess_input(location)
       data = pd.DataFrame([[location,total_sqft,bath,bhk]], columns=["location", "total_sqft", "bath", "bhk"])
       prediction = model.predict(data) # Replace with model prediction logic
       return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)