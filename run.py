import os
import json
import torch
import torch.nn.functional as F
import joblib
import pandas as pd
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= CONFIGURATION =================

# --- MODEL PATHS (MUST MATCH YOUR SAVED LOCATIONS) ---
OLD_MODEL_DIR = r"\\BULGARIA\Users\Intern\Downloads\job_domain_roberta_model-20260113T104916Z-3-001\job_domain_roberta_model"
NEW_MODEL_DIR = r"\\BULGARIA\Users\Intern\Downloads\jobbert_v2_finetuned"

# --- FILE PATHS ---
INPUT_CSV_PATH = "searchString.csv"  # <--- NAME OF YOUR INPUT FILE
OUTPUT_CSV_PATH = "output2.csv"

# --- BATCHING PARAMETER ---
BATCH_SIZE = 50  # Process 50 rows at a time

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GLOBALS FOR LOADED ASSETS ---
old_tokenizer, old_model, old_labels = None, None, None
new_tokenizer, new_model, new_label_encoder = None, None, None


# ================= MODEL LOADING FUNCTIONS (Copied from Flask App) =================

def load_old_model():
    global old_tokenizer, old_model, old_labels
    print("Loading OLD RoBERTa model...")

    OLD_LABELS_PATH = os.path.join(OLD_MODEL_DIR, "labels.json")

    old_tokenizer = AutoTokenizer.from_pretrained(OLD_MODEL_DIR, use_fast=False)
    old_model = AutoModelForSequenceClassification.from_pretrained(OLD_MODEL_DIR)
    old_model.to(device)
    old_model.eval()

    try:
        with open(OLD_LABELS_PATH, "r", encoding="utf-8") as f:
            category_to_id_str = json.load(f)
            old_labels = {int(v): k for k, v in category_to_id_str.items()}
    except Exception as e:
        print(f"Could not load labels.json for old model: {e}. Falling back to model config.")
        old_labels = old_model.config.id2label

    print("OLD Model loaded.")


def load_new_model():
    global new_tokenizer, new_model, new_label_encoder
    print("Loading NEW JobBERT-v2 model...")

    NEW_ENCODER_PATH_FULL = os.path.join(NEW_MODEL_DIR, "label_encoder.pkl")

    new_tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL_DIR)
    new_model = AutoModelForSequenceClassification.from_pretrained(NEW_MODEL_DIR)
    new_model.to(device)
    new_model.eval()

    new_label_encoder = joblib.load(NEW_ENCODER_PATH_FULL)
    print("NEW Model loaded.")

# ================= BATCH PREDICTION FUNCTIONS (Modified to return time and results) =================

def predict_batch_old(tokenizer, model, labels_map, texts):
    start_time = time.time()
    model.to(device)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=-1)

    batch_results = []
    for i in range(len(texts)):
        # Collect top 3 results for one text
        top_3_preds = []
        for j in range(3):
            idx = top_indices[i][j].item()
            prob = top_probs[i][j].item()
            top_3_preds.append({
                "category": labels_map[idx],
                "confidence": round(prob * 100, 2)
            })

        batch_results.append(top_3_preds)

    end_time = time.time()
    response_time = end_time - start_time

    return batch_results, response_time


def predict_batch_new(tokenizer, model, label_encoder, texts):
    start_time = time.time()
    model.to(device)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=-1)

    batch_results = []
    for i in range(len(texts)):
        top_3_preds = []
        for j in range(3):
            idx = top_indices[i][j].item()
            prob = top_probs[i][j].item()
            category_name = label_encoder.inverse_transform([idx])[0]

            top_3_preds.append({
                "category": category_name,
                "confidence": round(prob * 100, 2)
            })
        batch_results.append(top_3_preds)

    end_time = time.time()
    response_time = end_time - start_time

    return batch_results, response_time


# ================= MAIN PROCESSING SCRIPT =================

def process_csv_in_batches():
    # 1. Load Models ONCE
    load_old_model()
    load_new_model()

    print(f"\nStarting batch processing of {INPUT_CSV_PATH} with batch size {BATCH_SIZE}...")

    # 2. Read Input CSV in Chunks (MODIFIED FOR NO HEADER)
    try:
        # Read CSV with NO HEADER (header=None)
        # Since there's no header, pandas assigns column names as 0, 1, 2...
        # We assume the job title is in the FIRST column, which will be named '0'.
        input_df_iter = pd.read_csv(
            INPUT_CSV_PATH,
            header=None,  # <--- KEY CHANGE: Tells pandas there is no header row
            usecols=[0],  # <--- KEY CHANGE: Explicitly load only the first column (index 0)
            chunksize=BATCH_SIZE
        )
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_CSV_PATH}'. Please check the path.")
        return
    except Exception as e:
        print(f"ERROR reading CSV: {e}. Ensure the input file exists and is readable.")
        return

    # 3. Initialize Output DataFrame Structure
    all_results = []
    row_counter = 0  # To generate the required ID column

    # 4. Process Batches
    for i, chunk_df in enumerate(input_df_iter):
        print(f"--- Processing Batch {i + 1} (Rows {row_counter + 1} to {row_counter + len(chunk_df)}) ---")

        # Extract the search terms from the first column (which pandas named '0')
        current_texts = chunk_df[0].tolist()  # <--- KEY CHANGE: Reference column by its integer index '0'

        # --- Process OLD Model ---
        old_preds, time_old = predict_batch_old(
            old_tokenizer, old_model, old_labels, current_texts
        )

        # --- Process NEW Model ---
        new_preds, time_new = predict_batch_new(
            new_tokenizer, new_model, new_label_encoder, current_texts
        )

        # --- Collect and Structure Results for the current batch ---
        for j in range(len(current_texts)):
            row_counter += 1

            result_row = {
                'id': row_counter,
                'search_term': current_texts[j]
            }

            n_required = 3
            for n in range(n_required):
                # Category Rank
                cat_rank = n + 1

                # Old Model Data
                result_row[f'cat_old_{cat_rank}'] = old_preds[j][n]['category']
                result_row[f'conf_old_{cat_rank}'] = old_preds[j][n]['confidence']

                # New Model Data
                result_row[f'cat_new_{cat_rank}'] = new_preds[j][n]['category']
                result_row[f'conf_new_{cat_rank}'] = new_preds[j][n]['confidence']


            all_results.append(result_row)

        print(f"Batch {i + 1} completed. Time: Old={time_old:.4f}s, New={time_new:.4f}s")

    # 5. Finalize and Save Output CSV
    if all_results:
        output_df = pd.DataFrame(all_results)

        output_df = output_df[[
            'id',
            'search_term',

            'conf_old_1', 'cat_old_1',
            'conf_old_2', 'cat_old_2',
            'conf_old_3', 'cat_old_3',

            'conf_new_1', 'cat_new_1',
            'conf_new_2', 'cat_new_2',
            'conf_new_3', 'cat_new_3'
        ]]

        output_df.to_csv(OUTPUT_CSV_PATH, index=False)

        print(f"\n--- Processing Complete ---")
        print(f"Total {len(output_df)} records processed.")
        print(f"Output saved to: {OUTPUT_CSV_PATH}")
    else:
        print("No data was processed.")


if __name__ == "__main__":
    process_csv_in_batches()