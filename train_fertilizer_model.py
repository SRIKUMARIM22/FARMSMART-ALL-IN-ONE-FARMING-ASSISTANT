
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# LOAD AND EXPLORE DATA
# ============================================================
print("="*60)
print("LOADING FERTILIZER PREDICTION DATASET")
print("="*60)

# Load the new dataset
df = pd.read_csv('Fertilizer Prediction.csv')

# Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()


df['Nitrogen'] = 50 - df['Nitrogen']
df['Potassium'] = 25 - df['Potassium']
df['Phosphorous'] = 50 - df['Phosphorous']

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['Fertilizer Name'].value_counts()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ============================================================
# DATA AUGMENTATION (since dataset is small - only 99 samples)
# ============================================================
print("\n" + "="*60)
print("AUGMENTING DATA")
print("="*60)

def augment_data(df, augmentation_factor=10):
    """
    Augment the dataset by adding gaussian noise to numerical features
    while preserving the relationships between features and target
    """
    augmented_data = [df.copy()]
    
    # Define noise levels for each feature
    noise_config = {
        'Temparature': 2.0,    # ±2 degrees
        'Humidity': 3.0,       # ±3%
        'Moisture': 3.0,       # ±3%
        'Nitrogen': 2.0,       # ±2 units
        'Potassium': 1.5,      # ±1.5 units
        'Phosphorous': 2.0     # ±2 units
    }
    
    for i in range(augmentation_factor - 1):
        aug_df = df.copy()
        for col, noise_level in noise_config.items():
            if col in aug_df.columns:
                noise = np.random.normal(0, noise_level, size=len(aug_df))
                aug_df[col] = aug_df[col] + noise
                # Ensure non-negative values
                aug_df[col] = aug_df[col].clip(lower=0)
        augmented_data.append(aug_df)
    
    result = pd.concat(augmented_data, ignore_index=True)
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(result)}")
    return result

# Augment the data
df_augmented = augment_data(df, augmentation_factor=15)

# ============================================================
# PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING DATA")
print("="*60)

# Separate features and target
feature_cols = ['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                'Nitrogen', 'Potassium', 'Phosphorous']
target_col = 'Fertilizer Name'

X = df_augmented[feature_cols].copy()
y = df_augmented[target_col].copy()

# Encode categorical variables
le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

X['Soil Type'] = le_soil.fit_transform(X['Soil Type'])
X['Crop Type'] = le_crop.fit_transform(X['Crop Type'])
y_encoded = le_fertilizer.fit_transform(y)

print(f"\nSoil types: {le_soil.classes_}")
print(f"Crop types: {le_crop.classes_}")
print(f"Fertilizer types: {le_fertilizer.classes_}")

# Convert all features to numeric
X = X.apply(pd.to_numeric, errors='coerce')
X.fillna(X.mean(), inplace=True)

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# MODEL COMPARISON
# ============================================================
print("\n" + "="*60)
print("COMPARING MODELS")
print("="*60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    print(f"{name}: {mean_score:.4f} ± {std_score:.4f}")
    
    if mean_score > best_score:
        best_score = mean_score
        best_model = model
        best_name = name

print(f"\n✅ Best Model: {best_name} with CV accuracy: {best_score:.4f}")

# ============================================================
# TRAIN BEST MODEL
# ============================================================
print("\n" + "="*60)
print("TRAINING BEST MODEL")
print("="*60)

best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"\nModel Performance on Test Set:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Detailed classification report
print('\nDetailed Classification Report:')
print(classification_report(y_test, y_pred, target_names=le_fertilizer.classes_))

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_fertilizer.classes_, 
            yticklabels=le_fertilizer.classes_)
plt.xlabel('Predicted Fertilizer')
plt.ylabel('Actual Fertilizer')
plt.title(f'Confusion Matrix - {best_name}\nAccuracy: {accuracy:.2%}')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
# plt.show()

# Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_names = X.columns.tolist()
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance for Fertilizer Recommendation")
    plt.bar(range(len(importances)), importances[indices], color='steelblue')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    # plt.show()
    
    print("\nFeature Importance:")
    for i in indices:
        print(f"  {feature_names[i]}: {importances[i]:.4f}")

# ============================================================
# SAVE MODEL AND ENCODERS
# ============================================================
print("\n" + "="*60)
print("SAVING MODEL AND ENCODERS")
print("="*60)

joblib.dump(best_model, 'fertilizer_model_v2.joblib')
joblib.dump(scaler, 'fertilizer_scaler_v2.joblib')
joblib.dump(le_soil, 'soil_encoder_v2.joblib')
joblib.dump(le_crop, 'crop_encoder_v2.joblib')
joblib.dump(le_fertilizer, 'fertilizer_encoder_v2.joblib')

print("Saved files:")
print("- fertilizer_model_v2.joblib")
print("- fertilizer_scaler_v2.joblib")
print("- soil_encoder_v2.joblib")
print("- crop_encoder_v2.joblib")
print("- fertilizer_encoder_v2.joblib")

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    """
    Predict fertilizer recommendation based on input parameters
    """
    # Create input array
    input_data = np.array([[temperature, humidity, moisture, 
                           le_soil.transform([soil_type])[0],
                           le_crop.transform([crop_type])[0],
                           nitrogen, potassium, phosphorous]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = best_model.predict(input_scaled)
    
    # Convert back to fertilizer name
    fertilizer_name = le_fertilizer.inverse_transform(prediction)[0]
    
    # Get prediction probability
    if hasattr(best_model, 'predict_proba'):
        probabilities = best_model.predict_proba(input_scaled)[0]
        confidence = np.max(probabilities)
    else:
        confidence = None
    
    return fertilizer_name, confidence

# ============================================================
# DEMO PREDICTIONS
# ============================================================
print("\n" + "="*60)
print("DEMO PREDICTIONS")
print("="*60)

# Test cases based on NPK patterns in the dataset
test_cases = [
    # Low Nitrogen -> Urea
    {"temp": 28, "hum": 55, "moist": 40, "soil": "Loamy", "crop": "Wheat", 
     "n": 12, "k": 25, "p": 50, "expected": "Urea"},
    # Low Phosphorous -> DAP
    {"temp": 30, "hum": 60, "moist": 45, "soil": "Clayey", "crop": "Rice", 
     "n": 38, "k": 25, "p": 12, "expected": "DAP"},
    # High Potassium needed -> 10-26-26
    {"temp": 26, "hum": 50, "moist": 35, "soil": "Sandy", "crop": "Maize", 
     "n": 42, "k": 8, "p": 32, "expected": "10-26-26"},
    # Balanced deficient -> 17-17-17
    {"temp": 27, "hum": 52, "moist": 38, "soil": "Black", "crop": "Cotton", 
     "n": 38, "k": 13, "p": 37, "expected": "17-17-17"},
]

print("\nTesting predictions:")
for i, case in enumerate(test_cases, 1):
    try:
        pred, conf = predict_fertilizer(
            case["temp"], case["hum"], case["moist"], 
            case["soil"], case["crop"],
            case["n"], case["k"], case["p"]
        )
        status = "✅" if pred == case["expected"] else "⚠️"
        conf_str = f"{conf:.1%}" if conf else "N/A"
        print(f"{status} Test {i}: N={case['n']}, K={case['k']}, P={case['p']} -> Predicted: {pred}, Expected: {case['expected']}, Confidence: {conf_str}")
    except Exception as e:
        print(f"❌ Test {i} Error: {e}")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
