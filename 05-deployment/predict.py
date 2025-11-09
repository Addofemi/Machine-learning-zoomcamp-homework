import pickle

# Load the model
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)

# Data to predict
client = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Get probability
X = [client]
pred = model.predict_proba(X)[0, 1]
print(pred)
