# Duplicate Product Detection in E-Commerce

This project focuses on detecting duplicate products in e-commerce datasets, utilizing various data cleaning, feature extraction, and similarity comparison techniques to identify duplicates among a large collection of products.

## Project Structure

- **Data Loading**: Loads a JSON dataset containing product information from multiple e-commerce websites.
- **Data Preprocessing Functions**: Functions to normalize units, clean and standardize product titles, and clean feature values for consistency.
- **Similarity Calculation**: Custom functions for calculating set similarity and weighted similarity between product features.
- **Feature Processing**: Processes product features to group similar features and standardize their representation.
- **Model ID Extraction**: Function to identify potential model IDs in product titles using regular expressions.
- **Binary Vector Creation**: Transforms cleaned data into binary vectors for similarity comparison.
- **Locality-Sensitive Hashing (LSH) and Clustering**: Applies LSH to binary vector matrix and uses Agglomerative clustering for identifying duplicates.
- **Evaluation**: Bootstrapping method for evaluating pair quality, completeness, and F1* measure, visualized using matplotlib.

## Usage

Ensure you have Python with pandas, numpy, matplotlib, sklearn, and re libraries installed. Update the `file_path` variable with your dataset path. To Run the script to execute preprocessing, feature extraction, similarity calculations, LSH, clustering, and evaluation. The script is currently set to do five bootstraps iterations, and in addition also loops over different amounts of rows per band used for LSH. These iterations will eventually be shown in a plot. However, because both the bootstrap loops are active and the loops over r, the total running time is really long.

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- re (regular expression library)
