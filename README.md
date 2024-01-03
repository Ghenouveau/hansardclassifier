# Malaysian Parliament Hansard Text Classification

## Introduction
This Streamlit app is designed to automatically classify segments of Hansard text from the Malaysian Parliament into categories: 'Question', 'Motion', or 'None'. It utilizes a combination of TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and a Random Forest classifier to analyze and categorize the text efficiently.

## Features
- Text classification into 'Question', 'Motion', or 'None'.
- Utilizes TF-IDF for feature extraction from text.
- Employs a Random Forest classifier for the prediction.
- Interactive Streamlit interface for easy use.

## Installation
To run this app, you'll need to install the required Python packages. Follow these steps:

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/ghaffur/hansardclassifier.git
   ```
2. Navigate to the app's directory.
   ```bash
   cd hansardclassifier
   ```
3. Install the required packages.
   ```bash
   pip install -r requirements.txt
   ```

## Running the App
After installation, you can run the app using Streamlit:

```bash
streamlit run hansardclassifier.py
```

Open your web browser and go to `http://localhost:8501` to view the app.

## Usage
- On the app's interface, upload a Hansard pdf file.
- The app will process the text and automatically classify each sentences for you.
- Use the default model or your own custom model.

## Train custom model using colab
- You must check "just extract text" then go to colab for training.
https://colab.research.google.com/drive/17jFvLk04el440-FBjBaxq1NDrIlAMwdy?usp=sharing

## License
This project is licensed under MIT License.