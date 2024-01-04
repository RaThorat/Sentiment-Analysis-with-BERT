import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load pre-trained model and tokenizer for multilingual support
model_name = 'distilbert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Load your Excel file
file_path = '/content/drive/MyDrive/Colab Notebooks/KTO resultaten open reacties.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Define a function to perform sentiment analysis and return sentiment scores
def get_sentiment_score(text):
    if pd.isnull(text) or isinstance(text, (int, float)):
        return np.nan
    else:
        encoded_input = tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs.logits

        # Applying a softmax activation function to obtain a probability distribution
        probabilities = torch.softmax(logits, dim=1).squeeze()

        # Calculate a single sentiment score based on the probability distribution
        sentiment_score = (probabilities[0] - probabilities[1]).item()
        return sentiment_score

# Define the columns to be used for sentiment analysis and their corresponding new column names
feedback_columns = {
    'Feedback website': 'Website_score',
    'Feedback telefonisch contact': 'Telefonisch_score',
    'Feedback emailcontact': 'Email_score',
    'Feedback aanvraagformulier': 'Aanvraagformulier_score',
    'Feedback regeling': 'Regeling_score',
    'Feedback Dienstverlening': 'Dienstverlening_score'
}

# Apply sentiment analysis function to the feedback columns and store the results in new columns
for col, new_col in feedback_columns.items():
    df[new_col] = df[col].apply(get_sentiment_score)

# Drop the original feedback columns from the DataFrame
df.drop(columns=feedback_columns.keys(), inplace=True)

# Save the modified DataFrame to a new Excel file
output_file_path = '/content/drive/MyDrive/Colab Notebooks/KTO resultaten open reacties sentiment analysis.xlsx'  # Replace with desired output file path
df.to_excel(output_file_path, index=False)
