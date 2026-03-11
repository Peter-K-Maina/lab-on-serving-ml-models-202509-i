from flask import Flask, request, jsonify
# Cross-Origin Resource Sharing (CORS)
# Modern browsers apply the "same-origin policy", which blocks web pages from
# making requests to a different origin than the one that served the page.
# This helps prevent malicious sites from reading sensitive data from another
# site you are logged into.
#
# However, there are many legitimate cases where cross-origin requests are
# needed. One example is:
#
## Single-Page Applications (SPA) hosted at example-frontend.com need to call
## APIs hosted at api.example-backend.com.
#
# To support this safely, CORS lets servers explicitly allow such requests.
from flask_cors import CORS
import ast
import joblib
import os
import pandas as pd

app = Flask(__name__)
# CORS(
#     app,
#     resources={r"/api/*": {
#         "origins": [
#             "https://127.0.0.1",
#             "https://localhost"
#         ]
#     }},
#     methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["Content-Type"]
# )

CORS(
    app, supports_credentials=False,
    resources={r"/api/*": { # This means CORS will only apply to routes that start with /api/
               "origins": [
                   "https://127.0.0.1", "https://localhost",
                   "https://127.0.0.1:443", "https://localhost:443",
                   "http://127.0.0.1", "http://localhost",
                   "http://127.0.0.1:5000", "http://localhost:5000",
                   "http://127.0.0.1:5500", "http://localhost:5500"
                ]
    }},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"])

# CORS(app, supports_credentials=False,
#      origins=["*"])

# Load different models
# joblib is used to load a trained model so that the API can serve ML predictions
decisiontree_classifier_baseline = joblib.load('./model/decisiontree_classifier_baseline.pkl')
decisiontree_regressor_optimum = joblib.load('./model/decisiontree_regressor_optimum.pkl')
naive_bayes_classifier = joblib.load('./model/naive_Bayes_classifier_optimum.pkl')
knn_classifier = joblib.load('./model/knn_classifier_optimum.pkl')
svm_classifier = joblib.load('./model/support_vector_classifier_optimum.pkl')
random_forest_classifier = joblib.load('./model/random_forest_classifier_optimum.pkl')

label_encoders_1b = joblib.load('./model/label_encoders_1b.pkl')
label_encoders_2 = joblib.load('./model/label_encoders_2.pkl')
label_encoders_4 = joblib.load('./model/label_encoders_4.pkl')
scaler_4 = joblib.load('./model/scaler_4.pkl')
scaler_5 = joblib.load('./model/scaler_5.pkl')

REQUIRED_CLASSIFIER_FEATURES = [
    'monthly_fee',
    'customer_age',
    'support_calls'
]

ASSOCIATION_RULES_CANDIDATE_PATHS = [
    './model/association_rules.pkl',
    './model/association_rules.joblib',
    './model/association_rules.csv',
    './model/association-rules.pkl',
    './model/association-rules.joblib',
    './model/association-rules.csv',
    './rules/association_rules.pkl',
    './rules/association_rules.joblib',
    './rules/association_rules.csv',
    './rules/association-rules.pkl',
    './rules/association-rules.joblib',
    './rules/association-rules.csv'
]

association_rules_df = None
association_rules_source = None

CLUSTER_MODEL_CANDIDATE_PATHS = [
    './model/client_cluster_classifier.pkl',
    './model/client_cluster_classifier_optimum.pkl',
    './model/cluster_classifier.pkl',
    './model/cluster_classifier_optimum.pkl',
    './model/cluster_model.pkl',
    './model/kmeans_classifier.pkl',
    './model/kmeans_model.pkl',
    './model/kmeans.pkl',
    './model/client_cluster_classifier.joblib',
    './model/cluster_classifier.joblib',
    './model/cluster_model.joblib',
    './model/kmeans_model.joblib',
    './rules/client_cluster_classifier.pkl',
    './rules/cluster_classifier.pkl',
    './rules/cluster_model.pkl',
    './rules/kmeans_model.pkl',
    './rules/client_cluster_classifier.joblib',
    './rules/cluster_classifier.joblib',
    './rules/cluster_model.joblib',
    './rules/kmeans_model.joblib'
]

REQUIRED_CLUSTER_FEATURES = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay',
    'Month',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType',
    'VisitorType',
    'Weekend'
]

NUMERIC_CLUSTER_FEATURES = [
    'Administrative',
    'Administrative_Duration',
    'Informational',
    'Informational_Duration',
    'ProductRelated',
    'ProductRelated_Duration',
    'BounceRates',
    'ExitRates',
    'PageValues',
    'SpecialDay',
    'OperatingSystems',
    'Browser',
    'Region',
    'TrafficType'
]

LABEL_ENCODED_CLUSTER_FEATURES = [
    'Month',
    'VisitorType',
    'Weekend'
]

cluster_classifier_model = None
cluster_classifier_source = None


def _validate_classifier_payload(data):
    if not isinstance(data, dict):
        return None, jsonify({'error': 'Invalid JSON payload'}), 400

    missing_fields = [field for field in REQUIRED_CLASSIFIER_FEATURES if data.get(field) is None]
    if missing_fields:
        return None, jsonify({'error': 'Missing required fields', 'missing_fields': missing_fields}), 400

    try:
        validated_features = {
            'monthly_fee': float(data.get('monthly_fee')),
            'customer_age': float(data.get('customer_age')),
            'support_calls': float(data.get('support_calls'))
        }
    except (TypeError, ValueError):
        return None, jsonify({'error': 'All classifier input fields must be numeric'}), 400

    return validated_features, None, None


def _normalize_product_collection(value):
    if isinstance(value, frozenset | set):
        return {str(item).strip() for item in value if str(item).strip()}

    if isinstance(value, (list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()}

    if value is None or pd.isna(value):
        return set()

    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return set()

        try:
            parsed_value = ast.literal_eval(stripped_value)
            if isinstance(parsed_value, (set, frozenset, list, tuple)):
                return {str(item).strip() for item in parsed_value if str(item).strip()}
            if isinstance(parsed_value, str) and parsed_value.strip():
                return {parsed_value.strip()}
        except (ValueError, SyntaxError):
            pass

        if ',' in stripped_value:
            return {item.strip() for item in stripped_value.split(',') if item.strip()}

        return {stripped_value}

    return {str(value).strip()} if str(value).strip() else set()


def _coerce_rules_dataframe(rules_object):
    if isinstance(rules_object, pd.DataFrame):
        rules_df = rules_object.copy()
    elif isinstance(rules_object, dict):
        if 'rules' in rules_object and isinstance(rules_object['rules'], pd.DataFrame):
            rules_df = rules_object['rules'].copy()
        else:
            rules_df = pd.DataFrame(rules_object)
    else:
        rules_df = pd.DataFrame(rules_object)

    if 'antecedents' not in rules_df.columns or 'consequents' not in rules_df.columns:
        raise ValueError("Association rules file must contain 'antecedents' and 'consequents' columns")

    if 'confidence' not in rules_df.columns:
        rules_df['confidence'] = 0.0

    rules_df['antecedents'] = rules_df['antecedents'].apply(_normalize_product_collection)
    rules_df['consequents'] = rules_df['consequents'].apply(_normalize_product_collection)
    rules_df['confidence'] = pd.to_numeric(rules_df['confidence'], errors='coerce').fillna(0.0)
    rules_df = rules_df[(rules_df['antecedents'].apply(len) > 0) & (rules_df['consequents'].apply(len) > 0)]

    return rules_df


def _load_association_rules_from_disk():
    for rules_path in ASSOCIATION_RULES_CANDIDATE_PATHS:
        if not os.path.exists(rules_path):
            continue

        file_extension = os.path.splitext(rules_path)[1].lower()
        if file_extension in ['.pkl', '.joblib']:
            loaded_rules = joblib.load(rules_path)
        elif file_extension == '.csv':
            loaded_rules = pd.read_csv(rules_path)
        else:
            continue

        return _coerce_rules_dataframe(loaded_rules), rules_path

    raise FileNotFoundError(
        'Association rules file not found. Expected one of: ' + ', '.join(ASSOCIATION_RULES_CANDIDATE_PATHS)
    )


def _get_association_rules():
    global association_rules_df, association_rules_source

    if association_rules_df is None:
        association_rules_df, association_rules_source = _load_association_rules_from_disk()

    return association_rules_df, association_rules_source


def _load_first_available_model(candidate_paths, model_description):
    for model_path in candidate_paths:
        if os.path.exists(model_path):
            return joblib.load(model_path), model_path

    raise FileNotFoundError(
        f"{model_description} file not found. Expected one of: {', '.join(candidate_paths)}"
    )


def _get_cluster_classifier_model():
    global cluster_classifier_model, cluster_classifier_source

    if cluster_classifier_model is None:
        cluster_classifier_model, cluster_classifier_source = _load_first_available_model(
            CLUSTER_MODEL_CANDIDATE_PATHS,
            'Cluster classifier model'
        )

    return cluster_classifier_model, cluster_classifier_source


def _validate_cluster_payload(data):
    if not isinstance(data, dict):
        return None, jsonify({'error': 'Invalid JSON payload'}), 400

    missing_fields = [field for field in REQUIRED_CLUSTER_FEATURES if data.get(field) is None]
    if missing_fields:
        return None, jsonify({'error': 'Missing required fields', 'missing_fields': missing_fields}), 400

    validated_payload = {}

    try:
        for field in NUMERIC_CLUSTER_FEATURES:
            validated_payload[field] = float(data.get(field))

        for field in LABEL_ENCODED_CLUSTER_FEATURES:
            raw_value = str(data.get(field)).strip()
            if not raw_value:
                return None, jsonify({'error': f"Field '{field}' cannot be empty"}), 400

            encoder = label_encoders_5.get(field)
            if encoder is None:
                return None, jsonify({'error': f"Label encoder for field '{field}' was not found"}), 500

            try:
                validated_payload[field] = int(encoder.transform([raw_value])[0])
            except ValueError:
                allowed_values = [str(item) for item in encoder.classes_]
                return None, jsonify({
                    'error': f"Unsupported value for field '{field}'",
                    'received_value': raw_value,
                    'allowed_values': allowed_values
                }), 400

    except (TypeError, ValueError):
        return None, jsonify({'error': 'Numeric cluster input fields must contain valid numbers'}), 400

    return validated_payload, None, None


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'api': 'serving-ml-models',
        'models_loaded': [
            'decision-tree-classifier',
            'decision-tree-regressor',
            'naive-bayes-classifier',
            'knn-classifier',
            'svm-classifier',
            'random-forest-classifier',
            'association-rules-recommender',
            'client-cluster-classifier'
        ]
    })

# Defines an HTTP endpoint
@app.route('/api/v1/models/decision-tree-classifier/predictions', methods=['POST'])
def predict_decision_tree_classifier():
    # Accepts JSON data sent by a client (browser, curl, Postman, etc.)
    data = request.get_json()
    # Create a DataFrame with the correct feature names
    new_data = pd.DataFrame([{
        'monthly_fee': data.get('monthly_fee'),
        'customer_age': data.get('customer_age'),
        'support_calls': data.get('support_calls')
    }])

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'monthly_fee',
        'customer_age',
        'support_calls'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Performs a prediction using the already trained machine learning model
    prediction = decisiontree_classifier_baseline.predict(new_data)[0]
    
    # Returns the result as a JSON response:
    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

# *2.a.* Sample cURL POST values (without HTTPS in NGINX and Gunicorn)

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *2.b.* Sample cURL POST values (with HTTPS in NGINX and Gunicorn)

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"monthly_fee\": 60, \"customer_age\": 30, \"support_calls\": 1}"

# *3* Sample PowerShell values:

# $body = @{
#     monthly_fee = 60
#     customer_age = 30
#     support_calls = 1
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-classifier/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

@app.route('/api/v1/models/decision-tree-regressor/predictions', methods=['POST'])
def predict_decision_tree_regressor():
    data = request.get_json()
    # Expected input keys:
    # 'PaymentDate', 'CustomerType', 'BranchSubCounty',
    # 'ProductCategoryName', 'QuantityOrdered', 'Percenta3geProfitPerUnit'

    # Create a DataFrame based on the input
    new_data = pd.DataFrame([data])

    # Convert PaymentDate to datetime
    new_data['PaymentDate'] = pd.to_datetime(new_data['PaymentDate'])

    # Identify all datetime columns
    datetime_columns = new_data.select_dtypes(include=['datetime64']).columns

    categorical_cols = new_data.select_dtypes(exclude=['int64', 'float64', 'datetime64[ns]']).columns

    # Encode categorical columns
    for col in categorical_cols:
        if col in new_data:
            new_data[col] = label_encoders_1b[col].transform(new_data[col])

    # Feature engineering for date
    new_data['PaymentDate_year'] = new_data['PaymentDate'].dt.year # type: ignore
    new_data['PaymentDate_month'] = new_data['PaymentDate'].dt.month # type: ignore
    new_data['PaymentDate_day'] = new_data['PaymentDate'].dt.day # type: ignore
    new_data['PaymentDate_dayofweek'] = new_data['PaymentDate'].dt.dayofweek # type: ignore
    new_data = new_data.drop(columns=datetime_columns)

    # Define the expected feature order (based on the order used during training)
    expected_features = [
        'CustomerType',
        'BranchSubCounty',
        'ProductCategoryName',
        'QuantityOrdered',
        'PaymentDate_year',
        'PaymentDate_month',
        'PaymentDate_day',
        'PaymentDate_dayofweek'
    ]

    # Reorder and select only the expected columns
    new_data = new_data[expected_features]

    # Predict
    prediction = decisiontree_regressor_optimum.predict(new_data)[0]
    return jsonify({'Predicted Percentage Profit per Unit = ': float(prediction)})

# *1* Sample JSON POST values
# {
#     "CustomerType": "Business",
#     "BranchSubCounty": "Kilimani",
#     "ProductCategoryName": "Meat-Based Dishes",
#     "QuantityOrdered": 8,
#     "PaymentDate": "2027-11-13"
# }

# *2.a.* Sample cURL POST values

# curl -X POST http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *2.b.* Sample cURL POST values

# curl --insecure -X POST https://127.0.0.1/api/v1/models/decision-tree-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d "{\"CustomerType\": \"Business\",
# 	\"BranchSubCounty\": \"Kilimani\",
# 	\"ProductCategoryName\": \"Meat-Based Dishes\",
# 	\"QuantityOrdered\": 8,
# 	\"PaymentDate\": \"2027-11-13\"}"

# *3* Sample PowerShell values:

# $body = @{
#     PaymentDate         = "2027-11-13"
#     CustomerType        = "Business"
#     BranchSubCounty     = "Kilimani"
#     ProductCategoryName = "Meat-Based Dishes"
#     QuantityOrdered = 8
# } | ConvertTo-Json

# Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/models/decision-tree-regressor/predictions `
#     -Method POST `
#     -Body $body `
#     -ContentType "application/json"

@app.route('/api/v1/models/naive-bayes-classifier/predictions', methods=['POST'])
def predict_naive_bayes_classifier():
    data = request.get_json(silent=True)
    validated_features, error_response, status_code = _validate_classifier_payload(data)
    if error_response is not None:
        return error_response, status_code

    new_data = pd.DataFrame([validated_features])

    try:
        prediction = naive_bayes_classifier.predict(new_data)[0]
    except Exception as exc:
        return jsonify({'error': 'Prediction failed', 'details': str(exc)}), 500

    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

@app.route('/api/v1/models/knn-classifier/predictions', methods=['POST'])
def predict_knn_classifier():
    data = request.get_json(silent=True)
    validated_features, error_response, status_code = _validate_classifier_payload(data)
    if error_response is not None:
        return error_response, status_code

    new_data = pd.DataFrame([validated_features])

    try:
        # Scale the data before prediction (KNN requires scaling)
        new_data_scaled = scaler_4.transform(new_data)
        prediction = knn_classifier.predict(new_data_scaled)[0]
    except Exception as exc:
        return jsonify({'error': 'Prediction failed', 'details': str(exc)}), 500

    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

@app.route('/api/v1/models/svm-classifier/predictions', methods=['POST'])
def predict_svm_classifier():
    data = request.get_json(silent=True)
    validated_features, error_response, status_code = _validate_classifier_payload(data)
    if error_response is not None:
        return error_response, status_code

    new_data = pd.DataFrame([validated_features])

    try:
        # Scale the data before prediction (SVM requires scaling)
        new_data_scaled = scaler_4.transform(new_data)
        prediction = svm_classifier.predict(new_data_scaled)[0]
    except Exception as exc:
        return jsonify({'error': 'Prediction failed', 'details': str(exc)}), 500

    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

@app.route('/api/v1/models/random-forest-classifier/predictions', methods=['POST'])
def predict_random_forest_classifier():
    data = request.get_json(silent=True)
    validated_features, error_response, status_code = _validate_classifier_payload(data)
    if error_response is not None:
        return error_response, status_code

    new_data = pd.DataFrame([validated_features])

    try:
        prediction = random_forest_classifier.predict(new_data)[0]
    except Exception as exc:
        return jsonify({'error': 'Prediction failed', 'details': str(exc)}), 500

    return jsonify({'Predicted Class = ': int(prediction)})

# *1* Sample JSON POST values
# {
#     "monthly_fee": 60,
#     "customer_age": 30,
#     "support_calls": 1
# }

@app.route('/api/v1/models/association-rules-recommender/recommendations', methods=['POST'])
def predict_association_rules_recommender():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({'error': 'Invalid JSON payload'}), 400

    product = data.get('product')
    if not isinstance(product, str) or not product.strip():
        return jsonify({'error': "Field 'product' is required and must be a non-empty string"}), 400

    normalized_product = product.strip().lower()

    try:
        rules_df, rules_source = _get_association_rules()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        return jsonify({'error': 'Failed to load association rules', 'details': str(exc)}), 500

    matching_rules = rules_df[
        rules_df['antecedents'].apply(
            lambda antecedents: normalized_product in {item.lower() for item in antecedents}
        )
    ]

    if matching_rules.empty:
        return jsonify({
            'product_purchased': product.strip(),
            'recommended_products': [],
            'confidence': 0.0,
            'rules_source': rules_source,
            'message': 'No recommendations found for the provided product'
        })

    recommendation_scores = {}
    for _, rule in matching_rules.iterrows():
        confidence = float(rule['confidence'])
        for recommended_product in rule['consequents']:
            cleaned_product = str(recommended_product).strip()
            if not cleaned_product or cleaned_product.lower() == normalized_product:
                continue
            current_best = recommendation_scores.get(cleaned_product)
            if current_best is None or confidence > current_best:
                recommendation_scores[cleaned_product] = confidence

    sorted_recommendations = sorted(
        recommendation_scores.items(),
        key=lambda item: (-item[1], item[0].lower())
    )

    top_recommendations = [item[0] for item in sorted_recommendations[:5]]
    top_confidence = sorted_recommendations[0][1] if sorted_recommendations else 0.0

    return jsonify({
        'product_purchased': product.strip(),
        'recommended_products': top_recommendations,
        'confidence': float(top_confidence),
        'rules_source': rules_source
    })


@app.route('/api/v1/models/client-cluster-classifier/predictions', methods=['POST'])
def predict_client_cluster_classifier():
    data = request.get_json(silent=True)
    validated_payload, error_response, status_code = _validate_cluster_payload(data)
    if error_response is not None:
        return error_response, status_code

    try:
        cluster_model, cluster_model_source = _get_cluster_classifier_model()
    except FileNotFoundError as exc:
        return jsonify({'error': str(exc)}), 503
    except Exception as exc:
        return jsonify({'error': 'Failed to load cluster classifier model', 'details': str(exc)}), 500

    try:
        new_data = pd.DataFrame([validated_payload])
        new_data = new_data[REQUIRED_CLUSTER_FEATURES]
        scaled_data = scaler_5.transform(new_data)

        prediction = cluster_model.predict(scaled_data)[0]
        predicted_cluster = int(prediction)
    except Exception as exc:
        return jsonify({'error': 'Cluster prediction failed', 'details': str(exc)}), 500

    return jsonify({
        'Predicted Cluster = ': predicted_cluster,
        'model_source': cluster_model_source
    })

# *1* Sample JSON POST values
# {
#     "product": "Bread"
# }

# (e.g., `python api.py`), and not if you import api.py from another script or test.

# __name__ is a special variable in Python. When you run a script directly,
# __name__ is set to '__main__'. If the script is imported, __name__ is set to
# the module's name.

# if __name__ == '__main__': checks if the script is being run directly.

# app.run(debug=True) starts the Flask development server with debugging enabled.
# This means:
## The server will automatically reload if you make code changes.
## You get detailed error messages in the browser if something goes wrong.
if __name__ == '__main__':
    app.run(debug=True)
# if __name__ == '__main__':
#     app.run(debug=False)
# if __name__ == "__main__":
#     app.run(ssl_context=("cert.pem", "key.pem"), debug=True)
