from flask import Flask, request, render_template
import joblib
import json

app = Flask(__name__, template_folder='templates')

# Load model and features list
model = joblib.load('../Data/churn_model_top5.pkl')
with open('../Data/top_features.json') as f:
    FEATURES = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    if request.method == 'POST':
        input_data = []
        missing_features = []
        for feat in FEATURES:
            val = request.form.get(feat)
            if val is None or val.strip() == '':
                missing_features.append(feat)
            else:
                try:
                    input_data.append(float(val))
                except ValueError:
                    prediction_text = f"Invalid input for {feat}. Please enter a number."
                    return render_template('index.html', prediction_text=prediction_text, features=FEATURES)
        if missing_features:
            prediction_text = f"Please enter values for all fields: {', '.join(missing_features)}"
        else:
            pred = model.predict([input_data])
            result = 'Churn' if int(pred[0]) == 1 else 'No Churn'
            prediction_text = f'Prediction: {result}'
    return render_template('index.html', prediction_text=prediction_text, features=FEATURES)

if __name__ == '__main__':
    app.run(debug=True)
