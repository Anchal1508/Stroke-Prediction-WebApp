from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = open('stroke-model.pkl', 'rb')
classifier = pickle.load(filename)
filename.close()
app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    if request.method == 'POST':

        myDict = request.form
        gender = int(myDict['gender'])
        age = int(myDict['age'])
        hypertension = int(myDict['hypertension'])
        heart = int(myDict['heart'])
        married = int(myDict['married'])
        work = int(myDict['work'])
        residence = int(myDict['residence'])
        glucose = float(myDict['glucose'])
        bmi = float(myDict['bmi'])
        smoking = int(myDict['smoking'])
    
        data = [gender,age,hypertension,heart,married,work,residence,glucose,bmi,smoking]
        my_prediction = classifier.predict([data])
    
        return render_template('result.html',prediction=my_prediction)



if __name__ == "__main__":
    app.run(debug=True)
