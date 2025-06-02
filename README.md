# Sarcasm detection in Customer-Feedback using NLP 

Problem Statement 


This project aims to enhance sentiment analysis using advanced NLP techniques by capturing sarcastic sentiments, focusing on reviews.
The goal is to develop a model that captures Sarcastic nuances more accurately than traditional methods.
The outcome will help businesses identify customer satisfaction levels effectively, improving product development and customer engagement. 
![image](https://github.com/user-attachments/assets/4088f64f-6eb1-4de8-8f5c-b5dc4d1b6598)
 By capturing subtle linguistic nuances often overlooked by existing tools, such as sarcasm and mixed emotions, the model will provide businesses with deeper insights into customer satisfaction and product performance. Success will be measured by the model's accuracy and its impact on enhancing business decisions and customer engagement strategies.
<center><img src = "https://memesbams.com/wp-content/uploads/2017/11/sheldon-sarcasm-meme.jpg" width="300"</center>

Here 's the link to the Dataset
https://storage.googleapis.com/retail-analytics-data/reviews_us_Electronics_v1_00.tsv
## Dataset
Source: Publicly available dataset from a major e-commerce platform

Domain: Electronics product reviews

Size: Sampled 10,000 reviews

Fields Used: review_body for sentiment analysis

## Methodology
Preprocessing

Lowercasing, contraction normalization, removal of punctuation, HTML tags, stop words

Lemmatization and tokenization

Feature Engineering

Polarity (–1 to +1)

Subjectivity (0 to 1)

Sarcasm labeling using heuristic rules based on polarity, subjectivity, and keywords

Modeling

Classical: TF-IDF + Multinomial Naïve Bayes

Deep Learning: BERT (bert-base-uncased)

## Preprocessing
Cleaned and normalized the text

Removed stop words and non-alphabetic characters

Converted text to lowercase

Applied lemmatization

Engineered sarcasm labels using heuristic rules

## Feature Engineering
Polarity and subjectivity scores were extracted using TextBlob

Sarcasm labeled using high subjectivity, low polarity, and presence of sarcastic keywords


## Modeling
TF-IDF + Naïve Bayes
Vectorizer: TfidfVectorizer(max_features=5000)

Classifier: MultinomialNB()

Train-Test Split: 80–20 stratified

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

 Accuracy Achieved: 79.3%

BERT Fine-Tuning
Model: bert-base-uncased

Tokenizer: Hugging Face AutoTokenizer

Hyperparameters:

Learning Rate: 2e-5

Batch Size: 16

Epochs: 3

Dropout Rate: 0.1 – 0.4 (tuned)

Weight Decay: 0.01 – 0.03

Framework: Hugging Face Trainer API
Accuracy Achieved: 91.0%

## Results
Model	Accuracy	
TF-IDF + Naïve Bayes	79.3%	
BERT (fine-tuned)	91.0%	

## Visualizations
Countplot for sentiment distribution

Word Clouds per sentiment class

Correlation Heatmap of polarity vs. subjectivity

Training Curves: Loss, learning rate, gradient norm, accuracy per epoch

## Limitations
Sarcasm detection was based on heuristic rules due to lack of labeled data

Class imbalance impacted performance in sarcasm and neutral categories

Training large models like BERT is resource-intensive

## Future Work
Collect or build a labeled sarcasm dataset

Try alternative transformer models like RoBERTa or DeBERTa

Explore ensemble techniques

Use attention visualization for interpretability

