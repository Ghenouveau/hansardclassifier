import io
import streamlit as st
from process_pdf import process_pdf
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from nltk.tokenize import sent_tokenize
import nltk
import re
import json
from fuzzywuzzy import process
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelEncoder

def initialize_nltk():
    """Initialize NLTK by downloading required data"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            st.error("Could not download required NLTK data. Tokenization will be disabled.")
            return False
    return True

on = False
def correct_names(correct_names_path, threshold=90):
    # Function to replace names based on fuzzy matching ###overkill
    #this is not best way to do it. this is slow and not efficient
    def replace_names(original_name, choices):
        best_match, score = process.extractOne(original_name, choices)
        return best_match if score >= threshold else original_name

    #check if it a dataframe
    # check if it is a dataframe
    if isinstance(correct_names_path, pd.DataFrame):
        correct_names = correct_names_path
    else:
        correct_names = pd.read_csv(correct_names_path)
    names_to_update = pd.read_csv('mps_name.csv')

    # Get list of correct names
    choices = correct_names['name'].tolist()

    # Apply fuzzy matching to update names
    names_to_update['Speaker'] = names_to_update['Speaker'].apply(lambda x: replace_names(x, choices))

    return names_to_update   
def predict_labels(dataframe,vectorizer_path='default_vectorizer.pkl',classifier_path='default_classifier.pkl',keywords=None, use_keyword_frequency=True):
    """
    Predict labels for the given dataframe.
    """
    if keywords is None:
        keywords = ['minta', '?']
    # Load pre-trained vectorizer and classifier
    vectorizer = joblib.load(vectorizer_path)
    classifier = joblib.load(classifier_path)

    if use_keyword_frequency:
        for keyword in keywords:
            dataframe[keyword] = dataframe['Dialogue'].apply(lambda x: x.count(keyword))

        # Combine keyword frequencies with TF-IDF features
        X_combined = hstack([vectorizer.transform(dataframe['Dialogue']), dataframe[keywords]])
        predictions = classifier.predict(X_combined)
    else:
        # Use only TF-IDF features for prediction
        predictions = classifier.predict(vectorizer.transform(dataframe['Dialogue']))

    # Add predictions to the DataFrames
    dataframe['Label'] = predictions
    return dataframe
def predict_labels_nn(dataframe, model_path='default_model.h5', tokenizer_path='default_tokenizer.json', label_encoder_path='default_label_encoder.pkl', maxlen=280):
    """
    Predict labels for the given dataframe using a Neural Network model.
    """
    # Load the pre-trained Neural Network model
    loaded_model = load_model(model_path)
    
    # Load the tokenizer
    with open(tokenizer_path) as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)

    # Load the label encoder
    label_encoder = joblib.load(label_encoder_path)

    # Ensure the Dialogue column is string type
    dataframe['Dialogue'] = dataframe['Dialogue'].astype(str)

    # Tokenize and pad the dialogues
    dialogues_seq = tokenizer.texts_to_sequences(dataframe['Dialogue'])
    dialogues_padded = pad_sequences(dialogues_seq, maxlen=maxlen)

    # Make predictions using the Neural Network
    predictions = loaded_model.predict(dialogues_padded)
    predicted_label_indexes = np.argmax(predictions, axis=1)

    # Transform predicted label indexes back to original labels
    predicted_labels = label_encoder.inverse_transform(predicted_label_indexes)

    # Add predicted labels to the DataFrame
    dataframe['Label'] = predicted_labels

    # Optionally, add probabilities for each class to the DataFrame
    for i, class_label in enumerate(label_encoder.classes_):
        dataframe[f'Probability_{class_label}'] = predictions[:, i]

    return dataframe
def tokenize_dialogues(df):
    """
    Tokenize dialogues into sentences using NLTK instead of Malaya.
    Falls back to simple splitting if NLTK is unavailable.
    """
    if 'Dialogue' not in df.columns:
        return df

    # Initialize NLTK if not already done
    nltk_available = initialize_nltk()

    all_rows = []
    for _, row in df.iterrows():
        try:
            if nltk_available:
                # Use NLTK's sent_tokenize
                sentences = sent_tokenize(row['Dialogue'])
            else:
                # Fallback to simple splitting on periods
                text = row['Dialogue']
                sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            
            for sentence in sentences:
                new_row = row.copy()
                new_row['Dialogue'] = sentence
                all_rows.append(new_row)
        except Exception as e:
            # If tokenization fails, keep the original text as is
            all_rows.append(row)
            
    return pd.DataFrame(all_rows)


def extract_speakers_and_dialogues(extracted_texts):
    all_data = []

    for content, filename in extracted_texts:
        pattern = r"(.+?):\s*([\s\S]+?)(?=(\n[A-Z].+?:|\d+\.\s*[^\n:]+?\s(?:minta*[\s\S]+?)(?=(\n[^\n:]+:|\Z))|\Z))"
        matches = re.findall(pattern, content)

        for match in matches:
            speaker, dialogue = match[0], match[1]
            all_data.append({"Filename": filename, "Speaker": speaker, "Dialogue": dialogue})

    return pd.DataFrame(all_data)
def process_uploaded_pdfs(uploaded_files, start_page=6, top_margin=64, bottom_margin=25, left_margin=25, right_margin=25):
    extracted_texts = []

    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            file_stream = io.BytesIO(uploaded_file.getvalue())
            processed_text = process_pdf(
                file_stream, start_page,
                top_margin, bottom_margin,
                left_margin, right_margin
            )
            extracted_texts.append((processed_text, uploaded_file.name))

    return extracted_texts
def filter_menteri(df):
    """
    Filter out rows based on specific patterns in the 'Speaker' column.
    """
    df = df[~df['Speaker'].str.contains(r'(Timbalan|Pertua|Menteri|Pengerusi|tugas Khas)|^(?!.*(?<!\[)\])(?=.*\[[^\]]+\]).', regex=True)]
    df = df[df['Speaker'].str.contains(r'\[.*?\]', regex=True)]
    return df
def remove_titles(df, titles_to_remove):
    if 'Speaker' in df.columns:
        regex_pattern = '|'.join(titles_to_remove)
        df['Speaker'] = df['Speaker'].astype(str).str.replace(regex_pattern, '', regex=True).str.strip()
    return df
def remove_seats_from_all_names(df):
    if 'Speaker' in df.columns:
        regex_pattern = r'\[.*\]'
        df['Speaker'] = df['Speaker'].astype(str).str.replace(regex_pattern, '', regex=True).str.strip()
    return df
def preprocess_dialogue(df, newline_handling=True):
    def preprocess_text(text):
        if newline_handling:
            text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        text = re.sub(r'[^\s\w.,!?:;"\'@#$%^&*()\-=+\[\]{}|<>/`~]', '', text)
        return text
    if 'Dialogue' in df.columns:
        df['Dialogue'] = df['Dialogue'].astype(str).apply(preprocess_text)
    return df
def merge_dialogues(df):
    grouped_texts = []
    current_group = {}

    for _, row in df.iterrows():
        # Check for change in speaker or label
        if row['Speaker'] != current_group.get('Speaker') or row['Label'] != current_group.get('Label'):
            if current_group:
                # Append the current group to the result
                current_group['Dialogue'] = current_group['Dialogue'].strip()
                grouped_texts.append(current_group.copy())

            # Start a new group
            current_group = row.to_dict()
        else:
            # Concatenate dialogue for the same speaker and label
            current_group['Dialogue'] += " " + row['Dialogue']

    # Add the last group
    if current_group:
        current_group['Dialogue'] = current_group['Dialogue'].strip()
        grouped_texts.append(current_group)

    return pd.DataFrame(grouped_texts)

def main():
    titles_to_remove = ['Ts','Reformasi','Setiausaha','Institusi','Pengerusi','Dato\'','\' ','Sri','Yang di-Pertua Dewan Rakyat',',','\‚Äô','\'','Timbalan Perdana Menteri', 'Menteri', 'Timbalan Menteri', 'Dato Sri', 'Datuk Seri', 'Datuk', 'Dato', 'Tuan', 'Puan', 'Dato Haji', "Dato' Haji", 'Datuk Haji', 'Datuk Seri Dr.', "Dato' Seri Dr.", 'Ir.', 'Tan Sri', 'Kapten', 'Dr.', 'Datuk Seri Utama', 'Datuk Seri Panglima', 'Tuan Haji', 'Puan Haji', "Tuan Syed", "Dato' Sri Haji", 'Yang Berhormat', 'Senator', 'Ahli Yang Berhormat']
    st.title("Automatic Hansard Classifier")
    st.caption("By Ghakindye")
    auto, manual = st.tabs(["Automatic", "Manual"])

    if 'processed_text' not in st.session_state:
        st.session_state['processed_text'] = None

    with auto:
        uploaded_files = st.file_uploader("Choose a PDF file", type=["csv","pdf"] ,accept_multiple_files=True, key="auto_file_uploader")
        advanced_options = st.expander("Advanced Options", expanded=False)
        with advanced_options:
            extract_text = st.checkbox("Just extract the text", value=False)
            name_correction = st.checkbox("Standardize names", disabled=True,value=False)
            split_dialogue = st.checkbox("Split long dialogues into small pieces (more accurate result)", value=True)
            merge_dialogue = st.checkbox("Merge similar results together", value=True)
            custom_model = st.checkbox("Use Custom Model", value=False)
        rf_or_nn = st.selectbox("Model", ["Random Forest", "Neural Network"])
        custom_vec = st.file_uploader("Use Custom Vectorizer",disabled=not custom_model, type="pkl")
        custom_clf = st.file_uploader("Use Custom Classifier",disabled=not custom_model, type="pkl")

        if st.button("Run",type="primary"):
            if not uploaded_files:
                st.error("Please upload at least one file")
                return

            try:
                # Check 1: PDF extraction
                extracted_texts = process_uploaded_pdfs(uploaded_files)
                if not extracted_texts:
                    st.error("No text could be extracted from the uploaded files")
                    return

                # Check 2: Speaker/Dialogue extraction
                final_texts = extract_speakers_and_dialogues(extracted_texts)
                if final_texts.empty:
                    st.error("No speakers or dialogues could be identified in the text")
                    return

                if 'Speaker' not in final_texts.columns or 'Dialogue' not in final_texts.columns:
                    st.error("Required columns 'Speaker' and 'Dialogue' not found in extracted text")
                    return

                # If just extracting text, validate before output
                if extract_text:
                    if len(final_texts) == 0:
                        st.error("No valid text found to extract")
                        return
                    st.dataframe(final_texts[['Filename','Speaker','Dialogue']])
                    st.download_button('Download CSV', final_texts[['Filename','Speaker','Dialogue']].to_csv(index=False).encode('utf-8'), 'predicted.csv', 'text/csv', key='download-csv')
                    return

                # Check 3: Filtering results
                final_texts = filter_menteri(final_texts)
                if final_texts.empty:
                    st.error("No data remained after filtering ministers")
                    return

                final_texts = remove_titles(final_texts, titles_to_remove)
                final_texts = remove_seats_from_all_names(final_texts)

                # Check 4: Preprocessing results
                final_texts = preprocess_dialogue(final_texts)
                if final_texts['Dialogue'].str.len().max() == 0:
                    st.error("All dialogues were empty after preprocessing")
                    return

                # Check 5: Tokenization
                if split_dialogue:
                    if not initialize_nltk():
                        st.warning("NLTK initialization failed. Proceeding without splitting dialogues.")
                    else:
                        final_texts = tokenize_dialogues(final_texts)
                        if final_texts.empty:
                            st.error("No valid dialogues after splitting")
                            return

                # Check 6: Model prediction
                if rf_or_nn == "Random Forest":
                    if custom_model:
                        if not (custom_vec and custom_clf):
                            st.error("Please upload both vectorizer and classifier for custom model")
                            return
                        final_texts = predict_labels(final_texts, custom_vec, custom_clf, keywords=['minta', '?'])
                    else:    
                        final_texts = predict_labels(final_texts)
                elif rf_or_nn == "Neural Network":
                    final_texts = predict_labels_nn(final_texts)

                # Check 7: Prediction results
                if 'Label' not in final_texts.columns:
                    st.error("Classification failed - no labels generated")
                    return

                # Check 8: Merging results
                if merge_dialogue:
                    final_texts = merge_dialogues(final_texts)
                    if final_texts.empty:
                        st.error("No data remained after merging dialogues")
                        return

                # Check 9: Name correction
                if name_correction:
                    final_texts = correct_names(final_texts, threshold=90)
                    if final_texts.empty:
                        st.error("Name correction resulted in empty dataset")
                        return

                # Final validation before output
                if len(final_texts) == 0:
                    st.error("No valid results to display")
                    return

                required_columns = ['Filename','Speaker','Dialogue','Label']
                if not all(col in final_texts.columns for col in required_columns):
                    st.error("Missing required columns in final output")
                    return

                st.dataframe(final_texts[required_columns])
                st.download_button('Download CSV', final_texts[required_columns].to_csv(index=False).encode('utf-8'), 'predicted.csv', 'text/csv', key='download-csv')

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.write("Please try again with a different file or check the file format")

if __name__ == "__main__":
    main()
