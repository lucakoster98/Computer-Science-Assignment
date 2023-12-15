#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:13:23 2023

@author: lucakoster
"""

import json
import re
import random
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from collections import Counter
from collections import defaultdict
from itertools import combinations
from sklearn.cluster import AgglomerativeClustering
import operator
import multiprocessing
import matplotlib.pyplot as plt

# Define the file path
file_path = '/Users/lucakoster/Desktop/MSc BAQM/BLOK 2/Computer Science for Business Analytics/Paper/TVs-all-merged.json'

# Open the file and load its content
with open(file_path, 'r') as file:
    data = json.load(file)


def normalize_unit(unit):
    # Variations of 'inch','hertz','cdmu','pounds' and 'watt'
    inch_variations = [' Inch', ' inches',
                       ' ”', ' -inch', ' inch', 'inch', '\"']
    for i in inch_variations:
        unit = unit.replace(i, 'inch')
    hertz_variations = [' Hertz', ' hertz', 'Hz', ' HZ', ' hz', ' -hz', 'hz']
    for i in hertz_variations:
        unit = unit.replace(i, 'hertz')
    cdmu_variations = [' cd/m\u00c2\u00b2', ' cd/m\u00c2\u00b2',
                       ' cd/m2', ' cd/m\u00b2', ' cdm2', ' Nit']
    for i in cdmu_variations:
        unit = unit.replace(i, 'cdmu')
    pound_variations = [' lbs', ' pounds', ' lbs.', ' lb.', 'lbs.', 'lb.']
    for i in pound_variations:
        unit = unit.replace(i, 'lbs')
    watt_variations = [' watt', ' watts',  'W ', 'w ']
    for i in watt_variations:
        unit = unit.replace(i, 'watt')
    return unit


def normalize_title(title):
    title.lower()
    normalize_unit(title)
    title = title.replace('-', '')
    title = title.replace('\"', '')
    title = title.replace(' /', '')
    title = title.replace('"', '')
    title = title.replace('(', '')
    title = title.replace(')', '')
    title = title.replace(',', '')
    title = ' '.join(w for w in title.split() if any(x.isdigit() for x in w))
    return title

def clean_value(value):
    value = normalize_unit(value)
    value = value.lower()
    value = value.replace('without', '-')
    value = value.replace('with', '+')
    value = value.replace('+', '')
    value = value.replace('-', '')
    value = value.replace('and', ' ')
    value = value.replace('|', ' ')
    value = value.replace('&#', '')
    value = value.replace(' x ', 'x')
    value = value.replace('yes', '1')
    value = value.replace('no', '0')
    value = value.replace('true', '1')
    value = value.replace('false', '0')
    value = value.replace('/', '')
    value = value.replace(',', '')
    value = value.replace('.', '')
    value = value.replace(')', '')
    value = value.replace('(', '')
    return value


def normalize_brand(brandname):
    brandname = brandname.lower
    return brandname

#Calculate similarity between two lists based on the size of their intersection compared to the smaller set
def calculate_set_similarity(a, b):
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / min(len(set_a), len(set_b))

#Calculate weighted similarity between two strings using SequenceMatcher and set similarity
def calculate_weighted_similarity(a, b, weight_seq, weight_set):
    sequence_similarity = SequenceMatcher(None, a, b).ratio()
    set_similarity = calculate_set_similarity(a.split(), b.split())
    return weight_seq * sequence_similarity + weight_set * set_similarity

#Process feature similarity and group similar features
def process_feature_similarity(sorted_features, threshold, weight_seq, weight_set):
    similar_features = {sorted_features[0][0]: []}
    for i, (feature, _) in enumerate(sorted_features[1:], start=1):
        j = 0
        while calculate_weighted_similarity(feature, sorted_features[j][0], weight_seq, weight_set) < threshold:
            j += 1
            if j == i:
                break
        if j < i:
            similar_key = sorted_features[j][0]
            similar_features.setdefault(similar_key, []).append(feature)
    return similar_features

#Replace occepurrences of similar features in text with a common representative feature
def replace_similar_features(text, feature_map):
    for key, values in feature_map.items():
        for value in values:
            text = text.replace(value, key)
    return text

#Standardize feature names in text
def standardize_feature_name(text, feature_map):
    text = replace_similar_features(text, feature_map)
    return text.lower().replace(' ', '')

# Process and count features in the dataset
feature_frequency = Counter(
    k for key in data for item in data[key] for k in item['featuresMap'])
feature_frequency = {feature: freq for feature,
                     freq in feature_frequency.items() if freq > 1}

# Sort features by frequency
sorted_features = sorted(feature_frequency.items(),
                         key=operator.itemgetter(1), reverse=True)

# Threshold and weights for similarity calculation
threshold_similarity = 0.8  
weight_seqmatch = 0.6
weight_jacsim = 0.4

# Grouping similar features
grouped_features = process_feature_similarity(
    sorted_features, threshold_similarity, weight_seqmatch, weight_jacsim)

# Filter out feature groups with only one representation
grouped_features = {k: v for k, v in grouped_features.items() if len(v) > 0}


def clean_brand(brandname):
    return brandname.lower()

def clean_brand_name(brand_name):
    # Use regex to keep only letters (removes numbers, symbols, etc.)
    return re.sub(r'[^a-zA-Z]', '', brand_name)


# Create a set for known brands
known_brands = set()

# First pass to collect known brands from featuresMap
for key, products in data.items():
    for product in products:
        for variant in ['Brand', 'Brand Name', 'Brand Name:']:
            if variant in product.get('featuresMap', {}):
                brand_name = clean_brand(product['featuresMap'][variant])
                known_brands.add(brand_name)


def extract_brand_from_title(title, known_brands):
    title_words = title.lower().split()
    for brand in known_brands:
        if brand in title_words:
            return brand

    # Fallback: use the first word of the title as the brand, cleaned
    return clean_brand_name(title_words[0])

def sanitize_string(s):
    # Replace or remove specific characters
    return s.replace('/', '').replace('-', '')


def find_model_id_in_title(title):
    # Extended regular expression pattern for model IDs
    pattern = r'\b[A-Z0-9-]{4,20}\b'
    
    # Finding all potential model IDs
    potential_ids = re.findall(pattern, title)

    # Exclude common non-ID patterns and prioritize IDs with letters and numbers
    potential_ids = [id for id in potential_ids if not re.fullmatch(r'\d{4,5}', id)]  # Exclude 4-5 digit numbers
    potential_ids = [id for id in potential_ids if re.search(r'[A-Z]+.*\d+|\d+.*[A-Z]+', id)]

    # If multiple IDs are found, prioritize the longer one, as it's more likely to be the model ID
    if len(potential_ids) > 1:
        potential_ids.sort(key=len, reverse=True)

    return potential_ids[0] if potential_ids else None


# Initialize DataFrame
CleanData = pd.DataFrame(columns=['modelID', 'title', 'titleMID', 'shop', 'brand', 'features'])

# Temporary lists to store data
modelIDs = []
titles = []
titleMIDs = []
shops = []
brands = []
features = []


# Iterate over each product in the dataset
for key in data.keys():
    for item in data[key]:
        # Append the model ID and shop name
        modelIDs.append(key)
        shops.append(item['shop'])

        # Normalize and append the title
        normalized_title = normalize_title(item['title'])
        titles.append(normalized_title)
        
        # Extract and append model ID from title
        extracted_model_id = find_model_id_in_title(normalized_title)
        titleMIDs.append(extracted_model_id)
        
        # Try to find brand in featuresMap
        brand_name = ''
        for variant in ['Brand', 'Brand Name', 'Brand Name:']:
            if variant in item.get('featuresMap', {}):
                brand_name = clean_brand(item['featuresMap'][variant])
                break

        # If brand not found in featuresMap, try to extract from title
        if not brand_name:
            brand_name = extract_brand_from_title(item['title'], known_brands)

        brands.append(brand_name)

        # Process key-value pairs
        kvpi = []
        for kvp, value in item.get('featuresMap', {}).items():
            if replace_similar_features(kvp, grouped_features) not in grouped_features.keys():
                continue
            else:
                k = standardize_feature_name(kvp, grouped_features)
                value = clean_value(value)
                kvpi.append(k + ': ' + value)
        #Process and append the features
        features.append(kvpi)

     
# Assign data to the DataFrame
CleanData['modelID'] = modelIDs
CleanData['title'] = titles
CleanData['titleMID'] = titleMIDs
CleanData['shop'] = shops
CleanData['brand'] = brands
CleanData['features'] = features

#%%
#Bootstrapping 
pair_quality_values = []
pair_completeness_values = []
f1_star_values = []
fractions_of_comparisons = []
f1_measure_values = []

bootstrap_iterations = 5
print('Starting with ' + str(bootstrap_iterations) + ' bootstrap iterations')
for i in range(bootstrap_iterations):
    print("--------------------------------- Bootstrap " + str(
        i + 1) + '---------------------------------')
    kept_indices = []
    for j in range(len(CleanData)):
        rand = random.randint(1, len(CleanData))
        if rand not in kept_indices:
            kept_indices.append(rand)
    deleted_indices = []
    for v in range(len(CleanData)):
        if v not in kept_indices:
            deleted_indices.append(v)
    CleanData_bootstrap = CleanData.drop(deleted_indices)
    CleanData_bootstrap = CleanData_bootstrap.reset_index(drop=True)
    print("Amount of randomly selected products: " + str(len(CleanData_bootstrap)))
    print('Percent of original dataset: ' + str(round((len(CleanData_bootstrap) / len(CleanData)) * 100)) + '%')

#CleanData_bootstrap = CleanData
# %%


    # Count the occurrences of each word
    word_count = defaultdict(int)
    for title in CleanData_bootstrap['title']:
        for word in title.split():
            word_count[word] += 1
    
    # Assuming 'features' is a column in CleanData containing lists of feature strings
    for feature_list in CleanData_bootstrap['features']:
        for feature in feature_list:
            for word in feature.split():
                word_count[word] += 1
    
    # Filter out words that occur only once
    model_words = [word for word, count in word_count.items() if count > 1]
    
    
    
    # Combine relevant columns for each product
    combined_text = CleanData_bootstrap.apply(lambda row: ' '.join(
        [str(row['title']), ' '.join(row['features'])]), axis=1)
    
    
    # Step 3: Create binary vector for each product
    binary_vectors = []
    for i in range(len(CleanData_bootstrap)):
        product_words = set(CleanData_bootstrap['title'][i].split())
        for feature in CleanData_bootstrap['features'][i]:
            product_words.update(feature.split())
    
        binary_vector = [1 if word in product_words else 0 for word in model_words]
        binary_vectors.append(binary_vector)
    
    # Step 4: Assemble the binary vector matrix
    binary_vector_matrix = np.transpose(np.array(binary_vectors))
    
    
    # %%
    
    # This following line has to be uncommented if the one desires to make the plots
    #different_values_r = [1,2,3,4,5,6,8,10,14,18,24,28,36]
    different_values_r = [3,4,5,6,8,10,14,18,24,28,36]
    # Define the number of hash functions (n) and similarity threshold (t)
    n = len(binary_vectors)
    for i in different_values_r:
        print('r = ' + str(i) + ' ------------------------------------------------------------------------')
        r = i  # Number of rows
        b = max(1, int(n * 0.5 // r))  # Number of bands, ensuring b*r >= n*0.5
        t = pow((1/b), (1/r))  # The similarity threshold
        
        def isPrime(x):
            return all(x % i != 0 for i in range(2, int(x ** 0.5) + 1))
        
        
        def findPrimeNum(num):
            while True:
                if isPrime(num):
                    return num
                num += 1
        
        
        max_val = findPrimeNum(n)  # A large prime number
        
        
        def create_hash_functions(num_hash_functions, max_val):
            hash_funcs = []
            for i in range(num_hash_functions):
                a = random.randint(1, max_val - 1)
                b = random.randint(0, max_val - 1)
                hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % max_val)
            return hash_funcs
        
        
        hash_funcs = create_hash_functions(n, max_val)
        
        # Create Signature Matrix
        signature_matrix = np.full((n, len(binary_vectors[0])), np.inf)
        
        # Transpose to iterate columns
        for i, vector in enumerate(np.array(binary_vectors).T):
            for j, val in enumerate(vector):
                if val:
                    for k, hash_func in enumerate(hash_funcs):
                        signature_matrix[k, i] = min(
                            signature_matrix[k, i], hash_func(j))
        
        # LSH and Bucketing
        lsh_buckets = {}
        for i, signature in enumerate(signature_matrix.T):
            for band in range(b):
                band_signature = tuple(signature[band * r:(band + 1) * r])
                bucket_key = (band, hash(band_signature))
                lsh_buckets.setdefault(bucket_key, []).append(i)
        
        
        def valid_pair(index1, index2, dataframe):
            # Exclude pairs from the same shop
            if dataframe.at[index1, 'shop'] == dataframe.at[index2, 'shop']:
                return False
        
            # Exclude pairs with different brands if both brands are known
            brand1 = dataframe.at[index1, 'brand']
            brand2 = dataframe.at[index2, 'brand']
            if brand1 and brand2 and brand1 != brand2:
                return False
            # the next line has to be uncommented one if you does no want to filter on titleMIDs
            #return True 
        
            # Exclude pairs with different titleMIDs, these last 7 lines of code can be commented out if one desires not to filter on titleMIDs
            model_id1 = dataframe.at[index1, 'titleMID']
            model_id2 = dataframe.at[index2, 'titleMID']
            # If both model IDs are None, or they match, return True
            if model_id1 is None or model_id2 is None or model_id1 == model_id2:
                return True
            # If model IDs are different and not None, return False
            return False
        
        
        # Generate Candidate Pairs with Shop and Brand Check
        candidate_pairs = set()
        for bucket in lsh_buckets.values():
            if len(bucket) > 1:
                for i, j in combinations(bucket, 2):
                    if valid_pair(i, j, CleanData_bootstrap):
                        candidate_pairs.add(tuple(sorted((i, j))))
        
        print("Number of candidate pairs:", len(candidate_pairs))
        
        
        # Group by 'ModelID' and count the number of items in each group
        grouped = CleanData_bootstrap.groupby('modelID').size()
        
        # Calculate the number of pairs for each group and sum them
        true_duplicate_count = sum(n * (n - 1) // 2 for n in grouped if n > 1)
        
        print("Number of true duplicate pairs based on identical ModelIDs:",
              true_duplicate_count)
        
        
        # Initialize the set for true duplicates
        true_duplicates = set()
        # Iterate over all possible pairs of products
        for i in range(len(CleanData_bootstrap)):
            for j in range(i + 1, len(CleanData_bootstrap)):
                # Check if the pair has the same modelID
                if CleanData_bootstrap.at[i, 'modelID'] == CleanData_bootstrap.at[j, 'modelID']:
                    # Add the pair as a tuple to the set of true duplicates
                    tupl = (i, j)
                    true_duplicates.add(tupl)
        
        
        # Calculate the number of true duplicates found within candidate_pairs
        true_duplicates_found = sum(
            1 for i, j in candidate_pairs if CleanData_bootstrap.at[i, 'modelID'] == CleanData_bootstrap.at[j, 'modelID'])
        
        # Pair Quality (PQ)
        pair_quality = true_duplicates_found / \
            len(candidate_pairs) if len(candidate_pairs) > 0 else 0
        
        # Pair Completeness (PC)
        pair_completeness = true_duplicates_found / \
            true_duplicate_count if true_duplicate_count > 0 else 0
        
        
        # F1*-measure
        if pair_quality + pair_completeness > 0:
            f1_star_measure = (2 * pair_quality * pair_completeness) / \
                (pair_quality + pair_completeness)
        else:
            f1_star_measure = 0
        
        total_number_possible_comparisons = sum(
            np.arange(0, len(CleanData) + 1, 1).tolist())
        
        print("Pair Quality:", pair_quality)
        print("Pair Completeness:", pair_completeness)
        print("F1*-measure:", f1_star_measure)
            
        
        #%%
        #Clustering
        def calculate_jaccard_similarity(index1, index2, sig_matrix):
            vector1 = sig_matrix[:, index1]
            vector2 = sig_matrix[:, index2]
            diff_vector = vector1 - vector2
            diff_vector[diff_vector != 0] = 1
            jaccard_distance = sum(diff_vector) / len(diff_vector)
            return 1 - jaccard_distance
        
        # Constructing the Similarity Matrix
        SimilarityMatrix = np.ones((len(CleanData.index), len(CleanData.index))) * 10000000
        for row in range(SimilarityMatrix.shape[0]):
            for col in range(SimilarityMatrix.shape[1]):
                if (row, col) in candidate_pairs or (col, row) in candidate_pairs:
                    SimilarityMatrix[row][col] = 1 - calculate_jaccard_similarity(row, col, signature_matrix)
                    
        dis_threshold = 0.72
            
        
        # Applying Hierarchical Clustering
        hierarchical_clustering = AgglomerativeClustering(metric='precomputed', linkage='single', distance_threshold=dis_threshold,
                                                          n_clusters=None).fit_predict(SimilarityMatrix)
        cluster_buckets = {}
        for index, cluster in enumerate(hierarchical_clustering):
            cluster_buckets.setdefault(cluster, []).append(index)
        cluster_buckets = {k: v for k, v in cluster_buckets.items() if len(v) > 1}
        
        # Identifying Pairs
        identified_pairs = []
        for cluster in cluster_buckets:
            for combination in combinations(cluster_buckets[cluster], 2):
                identified_pairs.append(tuple(sorted(combination)))
        
        # Calculating Metrics
        true_positives = set(identified_pairs).intersection(true_duplicates)
        false_positives = set(identified_pairs) - set(true_duplicates)
        false_negatives = set(true_duplicates) - set(identified_pairs)
        
        precision_score = len(true_positives) / (len(true_positives) + len(false_positives))
        recall_score = len(true_positives) / (len(true_positives) + len(false_negatives))
        
        f1_measure = 2 * (precision_score * recall_score) / (precision_score + recall_score)
        
        # Append results to lists
        pair_quality_values.append(pair_quality)
        pair_completeness_values.append(pair_completeness)
        f1_star_values.append(f1_star_measure)
        number_of_products = len(CleanData_bootstrap)
        total_number_possible_comparisons = (number_of_products * (number_of_products - 1)) / 2
        fraction_of_comparisons = len(candidate_pairs) / total_number_possible_comparisons
        fractions_of_comparisons.append(fraction_of_comparisons)
        f1_measure_values.append(f1_measure)
        
        print('Pair Quality after clustering = ' + str(precision_score))
        print('Pair Completeness after clustering    = ' + str(recall_score))
        print('F1-measure  = ' + str(f1_measure))

# Plotting the results
plt.figure(figsize=(12, 8))

# Plotting Pair Quality
plt.subplot(2, 2, 1)
plt.plot(fractions_of_comparisons, pair_quality_values, marker='o')
plt.title('Pair Quality')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Quality')

# Plotting Pair Completeness
plt.subplot(2, 2, 2)
plt.plot(fractions_of_comparisons, pair_completeness_values, marker='o')
plt.title('Pair Completeness')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('Pair Completeness')

# Plotting F1*-measure
plt.subplot(2, 2, 3)
plt.plot(fractions_of_comparisons, f1_star_values, marker='o')
plt.title('F1*-measure')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1*-measure')

# Plotting F1 measure after clustering
plt.subplot(2, 2, 4)
plt.plot(fractions_of_comparisons, f1_measure_values, marker='o')
plt.title('F1-Measure After Clustering')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1-Measure')

plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
