# Information Retrieval Final Code

import os
import shutil
import ijson
import os
import json
import string
import math
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix  # For sparse vector representation

# Initialize NLTK tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
punctuation = set(string.punctuation)
punctuation.update(["``", "''"])


# This is to split the dataset
def split_dataset(folder_path, num_parts=2):
    # List all text files in the folder
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    total_files = len(all_files)

    # Shuffle files to ensure even distribution, if needed
    all_files.sort()  # Sorting can help in keeping track or can be randomized for a more balanced split

    # Calculate the number of files per part
    files_per_part = total_files // num_parts
    remainder = total_files % num_parts

    # Create individual folders and distribute files across them
    for part in range(num_parts):
        part_folder = os.path.join(folder_path, f"split_part_{part + 1}")
        os.makedirs(part_folder, exist_ok=True)

        # Calculate start and end indices for this part
        start_idx = part * files_per_part + min(part, remainder)
        end_idx = start_idx + files_per_part + (1 if part < remainder else 0)

        # Move files to the current part folder
        for i in range(start_idx, end_idx):
            src_path = os.path.join(folder_path, all_files[i])
            dst_path = os.path.join(part_folder, all_files[i])
            shutil.move(src_path, dst_path)

    print(
        f"Dataset split into {num_parts} parts, with files distributed across 'split_part_' folders."
    )


# This is to generate tf idf and the inverted index
# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [
        word.lower()
        for word in tokens
        if word.lower() not in stop_words and word not in punctuation
    ]
    return [stemmer.stem(word) for word in tokens]


# Distributed document processing to calculate TF and partial IDF
def process_documents_distributed(folder_path, output_prefix, print_interval=200):
    tf = defaultdict(dict)
    partial_idf = defaultdict(int)
    inverted_index = defaultdict(list)
    document_count = 0

    for idx, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            unique_id = filename.split("_")[1].replace(".txt", "")

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            processed_text = preprocess_text(text)
            if not processed_text:
                continue

            document_count += 1
            term_counts = defaultdict(int)
            for token in processed_text:
                term_counts[token] += 1
                if unique_id not in inverted_index[token]:
                    inverted_index[token].append(unique_id)

            total_terms = len(processed_text)
            for term, count in term_counts.items():
                tf[unique_id][term] = (
                    1 + math.log10(count / total_terms + 0.1) if count > 0 else 0
                )
                partial_idf[term] += 1

            # Print progress every `print_interval` documents
            if (idx + 1) % print_interval == 0:
                print(f"Processed {idx + 1} documents on {output_prefix}...")

    # Save partial results for later merging
    with open(f"{output_prefix}_tf.json", "w") as tf_file:
        json.dump(tf, tf_file)
    with open(f"{output_prefix}_partial_idf.json", "w") as idf_file:
        json.dump(partial_idf, idf_file)
    with open(f"{output_prefix}_inverted_index.json", "w") as index_file:
        json.dump(inverted_index, index_file)

    print(f"Completed processing {document_count} documents on {output_prefix}.")
    return document_count


# Main function to iterate over split folders
def process_all_splits(main_folder_path):
    split_folders = [
        f for f in os.listdir(main_folder_path) if f.startswith("split_part_")
    ]

    for split_folder in split_folders:
        folder_path = os.path.join(main_folder_path, split_folder)
        output_prefix = split_folder  # Use split folder name as prefix

        # Check if output files already exist
        tf_file_path = f"{output_prefix}_tf.json"
        idf_file_path = f"{output_prefix}_partial_idf.json"
        index_file_path = f"{output_prefix}_inverted_index.json"

        if (
            os.path.exists(tf_file_path)
            and os.path.exists(idf_file_path)
            and os.path.exists(index_file_path)
        ):
            print(f"Skipping {split_folder}, output files already exist.")
            continue  # Skip this split if outputs already exist

        print(f"Starting processing for folder: {split_folder}")
        process_documents_distributed(folder_path, output_prefix)


# This is to merge the generated tf idf
# Function to save merged data periodically
def save_intermediate(merged_dict, filename):
    with open(filename, "w") as f:
        json.dump(merged_dict, f)


# Load and merge JSON files incrementally
def load_and_merge_incrementally(file_list, key_type, save_every=1000):
    merged_dict = defaultdict(lambda: None)
    batch_counter = 0

    for i, file_name in enumerate(file_list):
        with open(file_name, "r") as f:
            partial_dict = json.load(f)
            for key, value in partial_dict.items():
                if isinstance(value, list):  # For lists, like in inverted_index
                    if merged_dict[key] is None:
                        merged_dict[key] = []
                    merged_dict[key].extend(value)
                elif isinstance(value, dict):  # For dictionaries, like in tf
                    if merged_dict[key] is None:
                        merged_dict[key] = {}
                    for sub_key, sub_value in value.items():
                        merged_dict[key][sub_key] = (
                            merged_dict[key].get(sub_key, 0) + sub_value
                        )
                elif isinstance(value, int):  # For integers, like in partial_idf
                    if merged_dict[key] is None:
                        merged_dict[key] = 0
                    merged_dict[key] += value
                else:
                    raise TypeError(
                        f"Unsupported data type for key {key}: {type(value)}."
                    )

        # Save intermediate results periodically
        if (i + 1) % save_every == 0:
            batch_counter += 1
            save_intermediate(
                merged_dict, f"intermediate_{key_type}_{batch_counter}.json"
            )
            merged_dict.clear()  # Clear memory after saving to disk

    # Save final batch if any remaining data
    if merged_dict:
        batch_counter += 1
        save_intermediate(merged_dict, f"intermediate_{key_type}_{batch_counter}.json")


# Calculate final IDF values based on merged partial IDF counts
def calculate_final_idf(merged_partial_idf, total_document_count):
    final_idf = {}
    for term, doc_freq in merged_partial_idf.items():
        final_idf[term] = math.log10(total_document_count / doc_freq)
    return final_idf


# Final merge of intermediate files
def final_merge_intermediate_files(intermediate_files):
    final_merged_dict = defaultdict(lambda: None)
    for filename in intermediate_files:
        with open(filename, "r") as f:
            partial_dict = json.load(f)
            for key, value in partial_dict.items():
                if isinstance(value, list):
                    if final_merged_dict[key] is None:
                        final_merged_dict[key] = []
                    final_merged_dict[key].extend(value)
                elif isinstance(value, dict):
                    if final_merged_dict[key] is None:
                        final_merged_dict[key] = {}
                    for sub_key, sub_value in value.items():
                        final_merged_dict[key][sub_key] = (
                            final_merged_dict[key].get(sub_key, 0) + sub_value
                        )
                elif isinstance(value, int):
                    if final_merged_dict[key] is None:
                        final_merged_dict[key] = 0
                    final_merged_dict[key] += value
                else:
                    raise TypeError(
                        f"Unsupported data type for key {key}: {type(value)}."
                    )

    return final_merged_dict


def final_merge_intermediate_files_chunk(intermediate_files, chunk_size=50000000):
    final_merged_dict = defaultdict(lambda: None)

    for filename in intermediate_files:
        print(f"Processing file: {filename}")
        with open(filename, "r") as f:
            buffer = ""
            char_count = 0

            while True:
                chunk = f.read(chunk_size)  # Read a chunk of characters
                if not chunk:
                    break  # End of file reached

                buffer += chunk
                char_count += len(chunk)

                try:
                    partial_dict = json.loads(buffer)  # Attempt to parse JSON
                    buffer = ""  # Clear buffer after successful parsing

                    for key, value in partial_dict.items():
                        if isinstance(value, list):
                            if final_merged_dict[key] is None:
                                final_merged_dict[key] = []
                            final_merged_dict[key].extend(value)
                        elif isinstance(value, dict):
                            if final_merged_dict[key] is None:
                                final_merged_dict[key] = {}
                            for sub_key, sub_value in value.items():
                                final_merged_dict[key][sub_key] = (
                                    final_merged_dict[key].get(sub_key, 0) + sub_value
                                )
                        elif isinstance(value, int):
                            if final_merged_dict[key] is None:
                                final_merged_dict[key] = 0
                            final_merged_dict[key] += value
                        else:
                            raise TypeError(
                                f"Unsupported data type for key {key}: {type(value)}."
                            )

                    if char_count % (chunk_size * 10) == 0:
                        print(f"Processed {char_count} characters from {filename}")

                except json.JSONDecodeError:
                    continue

        print(f"Finished processing file: {filename}")
    return final_merged_dict


# Main merge function
def merge_multiple_files(num_splits):
    # tf_files = [f"split_part_{i + 1}_tf.json" for i in range(num_splits)]
    # partial_idf_files = [f"split_part_{i + 1}_partial_idf.json" for i in range(num_splits)]
    # inverted_index_files = [f"split_part_{i + 1}_inverted_index.json" for i in range(num_splits)]

    # Merge each type of file incrementally
    # load_and_merge_incrementally(tf_files, "tf")
    # load_and_merge_incrementally(partial_idf_files, "partial_idf")
    # load_and_merge_incrementally(inverted_index_files, "inverted_index")

    # Gather intermediate files and perform final merge
    final_tf = final_merge_intermediate_files_chunk(
        [f for f in os.listdir() if f.startswith("intermediate_tf")]
    )
    with open("final_tf.json", "w") as tf_file:
        json.dump(final_tf, tf_file)
    print("Done merging TF files.")
    total_document_count = len(final_tf)
    del final_tf

    final_partial_idf = final_merge_intermediate_files(
        [f for f in os.listdir() if f.startswith("intermediate_partial_idf")]
    )
    final_idf = calculate_final_idf(final_partial_idf, total_document_count)
    with open("final_idf.json", "w") as idf_file:
        json.dump(final_idf, idf_file)
    del final_partial_idf
    del final_idf

    final_inverted_index = final_merge_intermediate_files_chunk(
        [f for f in os.listdir() if f.startswith("intermediate_inverted_index")]
    )
    with open("final_inverted_index.json", "w") as index_file:
        json.dump(final_inverted_index, index_file)
    print("Done merging Inverted Index files.")
    del final_inverted_index  # Clear memory

    print("Merging completed and final files saved.")


# This is to process queries


# Preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [
        word.lower()
        for word in tokens
        if word.lower() not in stop_words and word not in punctuation
    ]
    return [stemmer.stem(word) for word in tokens]


# Build sparse TF-IDF vectors for VSM
def vsm_collection(tf_dict, idf_dict, inverted_index):
    vocabulary = sorted(inverted_index.keys())
    term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    tf_idf_vectors = {}

    for doc_id, term_freqs in tf_dict.items():
        indices = []
        values = []
        for term, tf_value in term_freqs.items():
            if term in term_to_index:
                idx = term_to_index[term]
                values.append(tf_value * idf_dict.get(term, 0))
                indices.append(idx)

        tf_idf_vectors[doc_id] = csr_matrix(
            (values, (np.zeros(len(indices)), indices)), shape=(1, len(term_to_index))
        )

    return tf_idf_vectors, term_to_index

def vsm_collection_updated(tf_dict, idf_dict, inverted_index): #mikaels
    vocabulary = sorted(inverted_index.keys())
    term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    tf_idf_vectors = {}

    for doc_id, term_freqs in tf_dict.items():
        indices = []
        values = []
        for term, tf_value in term_freqs.items():
            if term in term_to_index:
                idx = term_to_index[term]
                # Convert tf_value and idf to float
                tf_value_float = float(tf_value)  # Convert tf_value to float
                idf_value_float = float(idf_dict.get(term, 0))  # Convert idf to float
                
                tf_idf_value = tf_value_float * idf_value_float  # Multiply as floats
                
                values.append(tf_idf_value)  # Append the float value
                indices.append(idx)

        if indices:  # Check if there are any valid indices
            tf_idf_vectors[doc_id] = csr_matrix(
                (values, (np.zeros(len(indices)), indices)), shape=(1, len(term_to_index))
            )
        else:
            tf_idf_vectors[doc_id] = csr_matrix((1, len(term_to_index)))  # Empty vector for this doc

    return tf_idf_vectors, term_to_index

def vsm_collection_updated_again(tf_dict, idf_dict, inverted_index):
    vocabulary = sorted(inverted_index.keys())
    term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
    tf_idf_vectors = {}
    total_docs = len(tf_dict)
    print_interval = 0.1  # Print progress every 10%
    next_print = print_interval * total_docs

    for i, (doc_id, term_freqs) in enumerate(tf_dict.items()):
        indices = []
        values = []
        for term, tf_value in term_freqs.items():
            if term in term_to_index:
                idx = term_to_index[term]
                tf_value_float = float(tf_value)
                idf_value_float = float(idf_dict.get(term, 0))
                
                tf_idf_value = tf_value_float * idf_value_float
                values.append(tf_idf_value)
                indices.append(idx)

        if indices:
            tf_idf_vectors[doc_id] = csr_matrix(
                (values, (np.zeros(len(indices)), indices)), shape=(1, len(term_to_index))
            )
        else:
            tf_idf_vectors[doc_id] = csr_matrix((1, len(term_to_index)))  # Empty vector for this doc

        # Print progress every 10%
        if i + 1 >= next_print:
            print(f"Progress: {int((i + 1) / total_docs * 100)}% done with VSM collection.")
            next_print += print_interval * total_docs

    print("Completed VSM collection.")
    return tf_idf_vectors, term_to_index



def vsm_query_results(
    query_file_path, tf_idf_vectors, term_to_index, idf_dict, output_csv_path
):
    df = pd.read_csv(query_file_path)
    results = []

    for _, row in df.iterrows():
        query_id = row["Query number"]
        processed_query = preprocess_text(row["Query"])
        if not processed_query:
            continue

        tf_query = defaultdict(int)
        for term in processed_query:
            tf_query[term] += 1

        query_vector = np.zeros(len(term_to_index))
        for term, tf in tf_query.items():
            if term in term_to_index:
                index = term_to_index[term]
                query_vector[index] = (1 + math.log10(tf + 0.1)) * idf_dict.get(term, 0)

        cosine_sims = {}
        query_vector = csr_matrix(query_vector)

        for doc_id, doc_vector in tf_idf_vectors.items():
            dot_product = query_vector.dot(doc_vector.T).toarray()[0, 0]
            if dot_product > 0:
                cosine_sims[doc_id] = dot_product / (
                    np.linalg.norm(query_vector.toarray())
                    * np.linalg.norm(doc_vector.toarray())
                )

        top_10_docs = sorted(cosine_sims.items(), key=lambda x: x[1], reverse=True)[:10]
        for doc_id, _ in top_10_docs:
            results.append({"Query number": query_id, "doc_number": doc_id})

    pd.DataFrame(results).to_csv(output_csv_path, index=False)


# This is to evaluate our results
def load_results(output_file):
    """Load the output results from a CSV file."""
    return pd.read_csv(output_file)


def load_ground_truth(ground_truth_file):
    """Load the ground truth data from a CSV file."""
    return pd.read_csv(ground_truth_file)


def compute_average_precision(retrieved, relevant):
    """Compute the average precision for a single query."""
    if len(relevant) == 0:
        return 0.0

    ap = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_count += 1
            ap += relevant_count / (i + 1)  # Precision at this rank

    return ap / len(relevant)  # Normalize by the number of relevant documents


def compute_mean_average_precision(results, ground_truth, k):
    """Compute MAP@K."""
    average_precisions = []

    for query_id in results["Query number"].unique():
        # Get relevant documents for this query
        relevant = set(
            ground_truth[ground_truth["Query_number"] == query_id]["doc_number"]
        )

        # Get top K retrieved documents for this query
        retrieved = (
            results[results["Query number"] == query_id]["doc_number"].head(k).tolist()
        )

        ap = compute_average_precision(retrieved, relevant)
        average_precisions.append(ap)

    return (
        sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    )


def compute_average_recall(retrieved, relevant):
    """Compute the average recall for a single query."""
    if len(relevant) == 0:
        return 0.0

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    # Recall = TP / (TP + FN)
    tp = len(retrieved_set.intersection(relevant_set))  # True positives
    recall = tp / len(relevant_set)

    return recall


def compute_mean_average_recall(results, ground_truth, k):
    """Compute MAR@K."""
    average_recalls = []

    for query_id in results["Query number"].unique():
        # Get relevant documents for this query
        relevant = set(
            ground_truth[ground_truth["Query_number"] == query_id]["doc_number"]
        )

        # Get top K retrieved documents for this query
        retrieved = (
            results[results["Query number"] == query_id]["doc_number"].head(k).tolist()
        )

        recall = compute_average_recall(retrieved, relevant)
        average_recalls.append(recall)

    return sum(average_recalls) / len(average_recalls) if average_recalls else 0.0


def evaluate(output_file, ground_truth_file, k_values, result_file):
    """Evaluate the results and print MAP@K and MAR@K, saving results to a file."""
    results = load_results(output_file)
    ground_truth = load_ground_truth(ground_truth_file)

    with open(result_file, "w", encoding="utf-8") as f:
        for k in k_values:
            map_k = compute_mean_average_precision(results, ground_truth, k)
            mar_k = compute_mean_average_recall(results, ground_truth, k)

            output_str = f"MAP@{k}: {map_k:.4f}\nMAR@{k}: {mar_k:.4f}\n"
            print(output_str)
            f.write(output_str)

def vsm_query_results_topidf(
    query_file_path, tf_idf_vectors, term_to_index, idf_dict, inverted_index, output_csv_path
):
    df = pd.read_csv(query_file_path)
    results = []

    for _, row in df.iterrows():
        query_id = row["Query number"]
        processed_query = preprocess_text(row["Query"])
        if not processed_query:
            continue

        # Step 1: Find the term with the highest IDF in the query
        tf_query = defaultdict(int)
        max_idf_term = None
        max_idf_value = -1
        for term in processed_query:
            tf_query[term] += 1
            if term in idf_dict and idf_dict[term] > max_idf_value:
                max_idf_term = term
                max_idf_value = idf_dict[term]

        # Skip if no term has an IDF value (unlikely)
        if not max_idf_term or max_idf_term not in inverted_index:
            continue

        # Step 2: Build the query vector using the standard TF-IDF formula
        query_vector = np.zeros(len(term_to_index))
        for term, tf in tf_query.items():
            if term in term_to_index:
                index = term_to_index[term]
                query_vector[index] = (1 + math.log10(tf + 0.1)) * idf_dict.get(term, 0)

        # Step 3: Only use documents from the postings list of the highest IDF term
        relevant_docs = inverted_index[max_idf_term]
        cosine_sims = {}
        query_vector = csr_matrix(query_vector)

        for doc_id in relevant_docs:
            if doc_id in tf_idf_vectors:
                doc_vector = tf_idf_vectors[doc_id]
                dot_product = query_vector.dot(doc_vector.T).toarray()[0, 0]
                if dot_product > 0:
                    cosine_sims[doc_id] = dot_product / (
                        np.linalg.norm(query_vector.toarray())
                        * np.linalg.norm(doc_vector.toarray())
                    )

        # Step 4: Sort and get the top 10 documents based on similarity scores
        top_10_docs = sorted(cosine_sims.items(), key=lambda x: x[1], reverse=True)[:10]
        for doc_id, _ in top_10_docs:
            results.append({"Query number": query_id, "doc_number": doc_id})

    pd.DataFrame(results).to_csv(output_csv_path, index=False)



number_of_splits = 41
# split_dataset("full_docs", number_of_splits)
# process_all_splits("full_docs")
# merge_multiple_files(number_of_splits)


def load_large_json(file_path, print_interval=0.1):
    """
    Load a large single JSON object file with progress updates.
    
    Parameters:
    - file_path: str, the path to the JSON file.
    - print_interval: float, percentage interval to print progress (0.1 = 10%).
    
    Returns:
    - data: dict, the loaded JSON data.
    """
    # Get the total size of the file in bytes
    total_size = os.path.getsize(file_path)
    loaded_size = 0
    progress = 0

    data = {}
    
    # Open file and use ijson to parse it incrementally
    with open(file_path, 'r', encoding='utf-8') as f:
        parser = ijson.kvitems(f, '')  # Parse top-level key-value pairs in JSON

        for key, value in parser:
            data[key] = value  # Store the term and its tf value

            # Calculate loaded size based on approximate byte length of each pair
            loaded_size += len(key.encode('utf-8')) + len(str(value).encode('utf-8'))
            new_progress = loaded_size / total_size

            # Print progress at specified intervals
            if new_progress - progress >= print_interval:
                progress = new_progress
                print(f"Loading {file_path}: {int(progress * 100)}% done")

    print(f"Completed loading {file_path}.")
    return data




with open("final_idf.json", "r", encoding="utf-8") as idf_file:
    idf = {item: value for item, value in ijson.kvitems(idf_file, "")}
print("Loaded IDF data...")

with open("final_inverted_index.json", "r", encoding="utf-8") as index_file:
    inverted_index = {item: value for item, value in ijson.kvitems(index_file, "")}
print("Loaded inverted index data...")

with open("final_tf.json", "r", encoding="utf-8") as tf_file:
    tf = {item: value for item, value in ijson.kvitems(tf_file, "")}
print("Loaded TF data...")



def vsm_query_results_with_topidf_updated(
    query_file_path, tf_idf_vectors, term_to_index, idf_dict, inverted_index, output_csv_path
):
    df = pd.read_csv(query_file_path, sep='\t')
    results = []
    total_queries = len(df)
    print_step = max(1, total_queries // 10)  # Print every 10% of queries

    for idx, row in enumerate(df.iterrows()):
        query_id = row[1]["Query number"]
        processed_query = preprocess_text(row[1]["Query"])
        if not processed_query:
            continue

        # Step 1: Find the term with the highest IDF in the query
        tf_query = defaultdict(int)
        max_idf_term = None
        max_idf_value = -1
        for term in processed_query:
            tf_query[term] += 1
            if term in idf_dict and idf_dict[term] > max_idf_value:
                max_idf_term = term
                max_idf_value = idf_dict[term]

        # Skip if no term has an IDF value (unlikely)
        if not max_idf_term or max_idf_term not in inverted_index:
            continue

        # Step 2: Build the query vector using the standard TF-IDF formula
        query_vector = np.zeros(len(term_to_index))
        for term, tf in tf_query.items():
            if term in term_to_index:
                index = term_to_index[term]
                query_vector[index] = (1 + math.log10(tf + 0.1)) * float(idf_dict.get(term, 0))

        # Step 3: Only use documents from the postings list of the highest IDF term
        relevant_docs = inverted_index[max_idf_term]
        cosine_sims = {}
        query_vector = csr_matrix(query_vector)

        for doc_id in relevant_docs:
            if doc_id in tf_idf_vectors:
                doc_vector = tf_idf_vectors[doc_id]
                dot_product = query_vector.dot(doc_vector.T).toarray()[0, 0]
                if dot_product > 0:
                    cosine_sims[doc_id] = dot_product / (
                        np.linalg.norm(query_vector.toarray()) * np.linalg.norm(doc_vector.toarray())
                    )

        # Step 4: Sort and get the top 10 documents based on similarity scores
        top_10_docs = sorted(cosine_sims.items(), key=lambda x: x[1], reverse=True)[:10]
        for doc_id, _ in top_10_docs:
            results.append({"Query number": query_id, "doc_number": doc_id})

        # Print progress every 10% of total queries
        if (idx + 1) % print_step == 0:
            print(f"Processing queries: {int((idx + 1) / total_queries * 100)}% done")

    # Save results to CSV
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print("Completed processing all queries.")




tf_idf_vectors,term_to_index = vsm_collection_updated_again(tf,idf,inverted_index)
vsm_query_results_with_topidf_updated("dev_queries.tsv",tf_idf_vectors,term_to_index,idf,inverted_index,"output_test.csv")


# Finally evaluate
output_file = "output_test.csv"  # Your output results file
ground_truth_file = "dev_query_results.csv"  # Your ground truth file
k_values = [3, 10]
result_file = "evaluation_results.txt"  # File to save the evaluation results
evaluate(output_file, ground_truth_file, k_values, result_file)
