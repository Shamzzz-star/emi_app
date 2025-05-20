import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pandas as pd


# Example Dataset

df = pd.read_csv("spam.csv", encoding="latin-1")
df = df.dropna()
labels= df['v2'].map({'ham': 0, 'spam': 1})  # Convert labels to binary
texts = df['v1'].values
# Split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Save Vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Models
models = {
    "logistic_regression": LogisticRegression(),
    "naive_bayes": MultinomialNB(),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "svc": SVC(probability=True)
}

# Train and Save
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)

print("✅ All models trained and saved in /models folder")
