# Task 1: Basic Text Pre-processing in NLP

## Environment
**Python 3 and Jupyter Notebook**

## Introduction
This notebook, titled "Task 1: Basic Text Pre-processing," is a part of "Assignment 2: Milestone I" in Natural Language Processing. It provides a comprehensive guide to essential text pre-processing techniques, focusing on processing job advertisement descriptions. The steps include information extraction, tokenization, text normalization, and vocabulary building.

## Objectives
1. **Extract Information**: Identify and extract key details from job advertisements.
2. **Tokenization**: Utilize regex patterns for tokenizing the job descriptions.
3. **Lowercase Conversion**: Transform all text to lowercase for uniformity.
4. **Word Length Filtering**: Exclude words with less than 2 characters.
5. **Stopword Removal**: Eliminate common stopwords using `stopwords_en.txt`.
6. **Single Occurrence Removal**: Remove words appearing only once across all descriptions.
7. **Top Frequency Words Removal**: Omit the top 50 most frequent words.
8. **Save Processed Text**: Store the cleaned job descriptions in text files.
9. **Vocabulary Building**: Create a unique word vocabulary from the processed texts.

## Implementation Details
- **Libraries Used**: The notebook includes Python libraries such as NLTK, Pandas, NumPy, Matplotlib, and others for data processing and visualization.
- **Data Inspection and Loading**: Examination of data folders, categories, and text documents.
- **Pre-Processing Steps**: Detailed code and explanations for each pre-processing step.
- **Output**: The notebook outputs processed job descriptions and a vocabulary list.
