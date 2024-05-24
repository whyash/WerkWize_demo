import nltk
nltk.download('movie_reviews')
import nltk
from nltk.corpus import movie_reviews
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

# Load the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Convert to DataFrame
data = pd.DataFrame(documents, columns=['words', 'category'])

# Join words into a single string for each document
data['text'] = data['words'].apply(lambda x: ' '.join(x))

# Remove unnecessary columns
data = data.drop(columns=['words'])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess_text)

# Encode the target labels
data['label'] = data['category'].apply(lambda x: 1 if x == 'pos' else 0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['label'], test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print classification report
print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))
# Example new text
new_text = ["I am driven to achieve and often take the lead in organizing group efforts.", "The movie was average and a pne time watch."]
new_text_clean = [preprocess_text(text) for text in new_text]
new_text_tfidf = tfidf_vectorizer.transform(new_text_clean)

# Predict sentiment
predictions = model.predict(new_text_tfidf)
predicted_labels = ['pos' if pred == 1 else 'neg' for pred in predictions]
print(predicted_labels)
