import io
import streamlit as st
from process_pdf import process_pdf
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import malaya
import re
from fuzzywuzzy import process

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
def tokenize_dialogues(df):
    tokenizer = malaya.tokenizer.SentenceTokenizer()
    if 'Dialogue' not in df.columns:
        return df

    # Tokenize dialogues and expand the DataFrame
    all_rows = []
    for _, row in df.iterrows():
        sentences = tokenizer.tokenize(row['Dialogue'])
        for sentence in sentences:
            new_row = row.copy()
            new_row['Dialogue'] = sentence
            all_rows.append(new_row)

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
    titles_to_remove = ['Ts','Reformasi','Setiausaha','Institusi','Pengerusi','Dato\'','\’ ','Sri','Yang di-Pertua Dewan Rakyat',',','\‚Äô','\'','Timbalan Perdana Menteri', 'Menteri', 'Timbalan Menteri', 'Dato Sri', 'Datuk Seri', 'Datuk', 'Dato', 'Tuan', 'Puan', 'Dato Haji', "Dato' Haji", 'Datuk Haji', 'Datuk Seri Dr.', "Dato' Seri Dr.", 'Ir.', 'Tan Sri', 'Kapten', 'Dr.', 'Datuk Seri Utama', 'Datuk Seri Panglima', 'Tuan Haji', 'Puan Haji', "Tuan Syed", "Dato' Sri Haji", 'Yang Berhormat', 'Senator', 'Ahli Yang Berhormat']
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
        custom_vec = st.file_uploader("Use Custom Vectorizer",disabled=not custom_model, type="pkl")
        custom_clf = st.file_uploader("Use Custom Classifier",disabled=not custom_model, type="pkl")
        if st.button("Run",type="primary"):
            if process_uploaded_pdfs(uploaded_files) is not None:

                extracted_texts = process_uploaded_pdfs(uploaded_files)

                final_texts = extract_speakers_and_dialogues(extracted_texts)
                final_texts = filter_menteri(final_texts)
                final_texts = remove_titles(final_texts, titles_to_remove)
                final_texts = remove_seats_from_all_names(final_texts)

                final_texts = preprocess_dialogue(final_texts)
                if split_dialogue:
                    final_texts = tokenize_dialogues(final_texts)

                if extract_text:
                    st.dataframe(final_texts[['Filename','Speaker','Dialogue']])
                    st.download_button('Download CSV', final_texts[['Filename','Speaker','Dialogue']].to_csv(index=False).encode('utf-8'), 'predicted.csv', 'text/csv', key='download-csv')
                    return

                if custom_model:
                    final_texts = predict_labels(final_texts, custom_vec, custom_clf, keywords=['minta', '?'])
                else:    
                    final_texts = predict_labels(final_texts)
                
                if merge_dialogue:
                    final_texts = merge_dialogues(final_texts)

                if name_correction:
                    final_texts = correct_names(final_texts, threshold=90)

                st.dataframe(final_texts[['Filename','Speaker','Dialogue','Label']])

                st.download_button('Download CSV', final_texts[['Filename','Speaker','Dialogue','Label']].to_csv(index=False).encode('utf-8'), 'predicted.csv', 'text/csv', key='download-csv')

    with manual:
        st.warning("[IN DEVELOPMENT] This features is coming soon!")

if __name__ == "__main__":
    main()
