# Topic Modeling on News Articles

Discover hidden topics or themes in a collection of news articles (BBC News Dataset).

## Features

- Preprocess text: tokenization, lowercasing, stopword removal, lemmatization
- Train LDA topic model and save artifacts
- Compare LDA vs NMF
- Visualize topics using WordClouds
- Optional: Streamlit app for interactive topic exploration

## Folder Structure

topic_modeling_app/
│
├── data/
│   └── bbc-text.csv             # dataset

│
├── artifacts/                   # هيتولد بعد التدريب

│   ├── vectorizer.joblib

│   ├── lda_model.joblib

│   ├── topic_top_words.json

│   └── wordclouds/              # الصور الناتجة عن WordCloud

│
├── code/

│   ├── train_topic_model.py     # تدريب LDA وحفظ artifacts

│   ├── compare_lda_nmf.py      # مقارنة LDA vs NMF

│   ├── topic_visualizations.py  # WordCloud لكل topic

│   └── app.py                   # واجهة Streamlit 
│

├── requirements.txt

└── README.md

# Topic Modeling on News Articles


![Topic Modeling on News Articles](https://github.com/maskar122/Elevvo_Internship/blob/d23513cce56056de02e20b1926d1b5a013f2d7a6/tasks/Screenshot%20(762).png)![Topic Modeling on News Articles](https://github.com/maskar122/Elevvo_Internship/blob/d23513cce56056de02e20b1926d1b5a013f2d7a6/tasks/Screenshot%20(763).png)
![Topic Modeling on News Articles](https://github.com/maskar122/Elevvo_Internship/blob/d23513cce56056de02e20b1926d1b5a013f2d7a6/tasks/Screenshot%20(764).png)

![Topic Modeling on News Articles](https://github.com/maskar122/Elevvo_Internship/blob/d23513cce56056de02e20b1926d1b5a013f2d7a6/tasks/Screenshot%20(767).png)
![Topic Modeling on News Articles](https://github.com/maskar122/Elevvo_Internship/blob/d23513cce56056de02e20b1926d1b5a013f2d7a6/tasks/Screenshot%20(769).png)
![Topic Modeling on News Articles](https://github.com/maskar122/Elevvo_Internship/blob/d23513cce56056de02e20b1926d1b5a013f2d7a6/tasks/Screenshot%20(772).png)





















# Text Summarization App

Generate concise summaries from long articles using pre-trained NLP models (BART).

## Features

- Clean and tokenize CNN-DailyMail dataset
- Generate summaries using BART
- Evaluate summaries with ROUGE scores
- Streamlit app with:
  - Textbox for article input
  - Textbox for reference summary
  - PDF upload support
  - ROUGE evaluation display

## Folder Structure
text_summarization_app/
│
├── data/ # Dataset (optional, cleaned small subset)
│ ├── load_data.py
│ ├── summarize.py
│ ├── evaluate_rouge.py
│ └── app.py


# Text Summarization App
![Text Summarization App](https://github.com/maskar122/Elevvo_Internship/blob/76550f2774aee975b2812dd8e061bf122a1bb798/tasks/Screenshot%20(756).png)
![Text Summarization App](https://github.com/maskar122/Elevvo_Internship/blob/502174ecd4a05bbc371ee51b2b7b234b08408da5/tasks/Screenshot%20(758).png)
