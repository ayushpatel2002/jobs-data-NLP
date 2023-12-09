# Overview: Natural Language Processing for Job Advertisement Classification

## Introduction
This project focuses on developing an automated system for classifying job advertisements into relevant categories. With the growth of online job hunting platforms, accurate categorization of job ads is crucial for better exposure to appropriate candidates and improving user experience. The project comprises three main tasks: Basic Text Pre-Processing, Generating Feature Representations, and Job Advertisement Classification.

## Task 1: Basic Text Pre-processing
### Environment
**Python 3 and Jupyter Notebook**

### Introduction
This task involves essential text pre-processing techniques, focusing on processing job advertisement descriptions. It covers steps like information extraction, tokenization, text normalization, and vocabulary building.

### Objectives
1. **Extract Information**: Identify and extract key details from job advertisements.
2. **Tokenization**: Utilize regex patterns for tokenizing job descriptions.
3. **Lowercase Conversion**: Transform all text to lowercase.
4. **Word Length Filtering**: Exclude words with less than 2 characters.
5. **Stopword Removal**: Eliminate common stopwords using `stopwords_en.txt`.
6. **Single Occurrence Removal**: Remove words appearing only once across all descriptions.
7. **Top Frequency Words Removal**: Omit the top 50 most frequent words.
8. **Save Processed Text**: Store cleaned job descriptions in text files.
9. **Vocabulary Building**: Create a unique word vocabulary.

### Implementation Details
- Libraries used include NLTK, Pandas, NumPy, Matplotlib, etc.
- Detailed code and explanations for each pre-processing step.
- Output: Processed job descriptions and a vocabulary list.

## Task 2: Generating Feature Representations for Job Advertisement Descriptions
### Environment
**Python 3 and Jupyter Notebook**

### Introduction
Focusing on creating various feature representations for job advertisements using the job description text.

### Objectives
- Produce diverse feature representations for job advertisements.
- Implement Bag-of-words model and word embeddings (e.g., FastText, GoogleNews300, Word2Vec, or Glove).

### Implementation Details
- Libraries used include pandas, numpy, sklearn, matplotlib, gensim, etc.
- Output: `count_vectors.txt` with sparse count vector representation of job advertisement descriptions.

### Usage
Demonstrates generating feature representations from textual data for NLP tasks like classification and clustering.

## Task 3: Job Advertisement Classification
### Environment
**Python 3 and Jupyter Notebook**

### Introduction
Dedicated to constructing machine learning models for categorizing job advertisements, with experiments on different feature sets and language models.

### Objectives
- Develop models for classifying job advertisement categories.
- Evaluate different language models and feature sets' impact on model accuracy.

### Implementation Details
- Libraries used include pandas, numpy, sklearn, matplotlib, etc.
- Evaluation Metrics: Confusion matrix, cross-validation, and statistical methods.

### Usage
Provides insights into machine learning application in text classification, evaluating the impact of different features and language models on classification accuracy.

## Libraries and Tools
- Python with libraries such as Pandas, NumPy, Sklearn, NLTK, Gensim.
- Jupyter Notebook for documentation and code implementation.

## Usage
This project offers insights into applying NLP techniques for real-world problems like job advertisement classification. It demonstrates text processing, feature extraction, and machine learning model application in a structured way.

## Data Analysis and Observations

### Model Accuracy Analysis
- **Count Vector Representation** achieves the highest average cross-validation accuracy at `85.15%` for job advertisement descriptions. This suggests that frequency and specificity of keywords in job descriptions are critical factors.
- **Unweighted Vector Representation** shows a `78.21%` accuracy for descriptions and `78.99%` for titles alone, indicating a slightly better performance with titles.
- **TF-IDF Weighted Vector Representation** records `76.67%` accuracy for descriptions and `76.29%` for titles, with a marginal edge for descriptions.
- Combining titles and descriptions does not significantly improve accuracy; in fact, it slightly reduces it to `77.83%` for unweighted and `77.06%` for weighted vectors.

### Key Insights
1. **Frequency Matters**: Count Vector Representation's effectiveness is likely due to the repetitive nature of important terms in job descriptions.
2. **Keyword Specificity**: Field-specific jargon or keywords in job descriptions are captured effectively by Count vectors.
3. **Dimensionality**: The high-dimensional nature of Count vectors aids in capturing a wide array of terms from the corpus.

### Analysis of Job Advertisement Title and Description on Model Accuracy
- Titles alone demonstrate a good predictive capability, slightly outperforming descriptions in unweighted vector scenarios.
- Combining title and description does not enhance, but rather slightly decreases, the model's accuracy.

### Probable Reasons for Observed Results
- **Redundancy and Dilution**: Combining titles with descriptions might introduce redundancy and dilute important features.
- **Different Linguistic Patterns**: Titles and descriptions have distinct structures, potentially introducing noise when combined.
- **Representation Limitations**: Current vector representation methods might not effectively capture the combined semantics.
- **Overfitting Risk**: The risk of overfitting may increase when combining titles and descriptions, especially in smaller datasets.
- **Merging Strategy**: The simple concatenation approach might not be optimal for merging title and description information.

Note: The data for this project consists of a collection of job advertisements categorized into different folders based on job types.
