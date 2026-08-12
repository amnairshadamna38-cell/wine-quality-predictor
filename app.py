from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load Dataset & Train Model
def train_model():
    df = pd.read_csv('wine_data.csv')
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [
            float(data['fixed_acidity']),
            float(data['volatile_acidity']),
            float(data['citric_acid']),
            float(data['residual_sugar']),
            float(data['chlorides']),
            float(data['free_sulfur_dioxide']),
            float(data['total_sulfur_dioxide']),
            float(data['density']),
            float(data['pH']),
            float(data['sulphates']),
            float(data['alcohol'])
        ]
        
        prediction = model.predict([features])[0]
        result = "Good Quality Wine 🍷" if prediction == 1 else "Bad Quality Wine ⚠️"
        
        return jsonify({'status': 'success', 'prediction': result})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)