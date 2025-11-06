# Incremental Training Project

## InfoCL

Code for Findings of EMNLP 2023: *[InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective](https://arxiv.org/abs/2310.06362)*
### Setup

```bash
pip install -r requirements.txt
```


## Overview
This project implements incremental training for sentence classification tasks using BERT-based models. It supports continual learning across multiple tasks with configurable datasets.

## Quick Start Guide

Choose your training approach based on your situation:

| Your Situation | Script to Use | Section to Read |
|----------------|---------------|-----------------|
| 🆕 First time training the model | `main.py` | [Use Case 1: Initial Training](#usage) |
| 🔄 Adding new documents/labels to existing model | `run_incremental_train.py` | [Use Case 2: Continuing Training](#use-case-2-continuing-training-with-new-documents) |

## Prerequisites
- Python 3.x
- PyTorch
- Transformers
- pandas
- PyYAML

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BallaRakesh/info_cl.git
cd info_cl
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

**IMPORTANT:** Before running the project, you must configure the following files:

### Step 1: Update `configs/default.yaml`

Edit `configs/default.yaml` and update these parameters according to your task:

```yaml
# DataArguments
task_name: "SentenceClassification"  # or "RelationExtraction"
data_path: "datasets"
dataset_name: "HWU64"  # or "FewRel"
max_seq_length: 256
overwrite_cache: False
pad_to_max_length: False
num_tasks: 7
class_per_task: 4
model_arch: "BertForSentenceClassification"  # or "BertForRelationExtraction"

# Model Configuration
model_name_or_path: "distilbert/distilbert-base-uncased"  # or "bert-base-uncased"
config_name: "distilbert/distilbert-base-uncased"
tokenizer_name: "distilbert/distilbert-base-uncased"
train_batch_size: 4
eval_batch_size: 4
stage1_epochs: 12
stage2_epochs: 8
device: "cpu"  # or "cuda" if GPU available
```

### Step 2: Update `configs/constants.py`

Set the data loading method flag:

```python
use_custom_train_test_split = True
```

### Step 3: Prepare Label Mapping

For sequence classification tasks, ensure your label mapping is defined in:
```
datasets/HWU64/id2label.json
```

This file should contain the mapping between label IDs and label names.

## Data Format

This project supports two data loading methods:

### Method 1: JSON Data Loading (All Labels Data)

When you have all label data in JSON format, the data should be structured as:

```json
[
  {
    "Query": "Your input text here",
    "Reasoning": "Explanation or reasoning",
    "Response": {
      "Complaint Category": "Main category",
      "Complaint Sub Category": "Sub category"
    }
  }
]
```

The code will load from three files:
- `6500_train_data_2.json`
- `6500_test_data_2.json`
- `6500_validate_data_2.json`

And create a DataFrame with columns:
- `Query`: Input text
- `Reasoning`: Reasoning/explanation
- `Category`: Complaint Category
- `Sub Category`: Complaint Sub Category

**Example JSON Entry:**
```json
{
  "Query": "I want to cancel my flight booking",
  "Reasoning": "User wants to cancel their reservation",
  "Response": {
    "Complaint Category": "Booking",
    "Complaint Sub Category": "Cancellation"
  }
}
```

**Resulting DataFrame Row:**
| Query | Reasoning | Category | Sub Category |
|-------|-----------|----------|--------------|
| I want to cancel my flight booking | User wants to cancel their reservation | Booking | Cancellation |

### Method 2: CSV Data Loading

When using CSV files with the flag `use_custom_train_test_split = True`, place your data in:
- `round_4_train_test/train.csv`
- `round_4_train_test/test.csv`

CSV format should include:
```csv
input_ids,attention_mask,labels
"[101, 2023, 2003, ...]","[1, 1, 1, ...]",0
```

**Example CSV Row:**
```csv
input_ids,attention_mask,labels
"[101, 1045, 2215, 2000, 6542, 2026, 3462, 8945, 102]","[1, 1, 1, 1, 1, 1, 1, 1, 1]",3
```

Each row will be parsed into:
```python
{
    'input_ids': [101, 1045, 2215, 2000, 6542, 2026, 3462, 8945, 102],  # tokenized input
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1],                      # attention mask
    'labels': 3                                                          # label ID
}
```

**Note:** 
- `input_ids` are tokenized representations of your text (e.g., using BERT tokenizer)
- `attention_mask` indicates which tokens to attend to (1) or ignore (0)
- `labels` should correspond to the IDs in `datasets/HWU64/id2label.json`

## Usage

### Running Incremental Training

After completing the configuration steps above, run:

```bash
python main.py
```

This will:
- Load the configuration from `configs/default.yaml`
- Apply the data loading method based on `use_custom_train_test_split` flag
- Load training and test data
- Perform incremental training across multiple tasks
- Save model checkpoints after each stage

### Experiment Workflow

To run an experiment and check how the incremental training works:

1. ✅ Update `configs/default.yaml` with your task configuration
2. ✅ Set `use_custom_train_test_split = True` in `configs/constants.py`
3. ✅ Prepare your data in the appropriate format (Method 1 or Method 2)
4. ✅ Ensure label mapping exists in `datasets/HWU64/id2label.json`
5. ✅ Run `python main.py`

The system will automatically:
- Load data based on your configuration
- Split tasks according to `num_tasks` and `class_per_task`
- Train incrementally across all tasks
- Save checkpoints for each stage

## Project Structure

```
.
├── main.py                          # Main training script
├── configs/
│   ├── default.yaml                 # Main configuration file
│   └── constants.py                 # Constants and flags
├── datasets/
│   └── HWU64/
│       └── id2label.json           # Label mapping for classification
├── round_4_train_test/              # CSV data files (Method 2)
│   ├── train.csv
│   └── test.csv
├── requirements.txt                 # Python dependencies
├── models/                          # Saved model checkpoints
└── README.md                        # This file
```

## Incremental Training

The incremental training approach allows you to:
- **Continual Learning**: Train on multiple tasks sequentially
- **Task Segmentation**: Automatically split classes across tasks
- **Checkpoint Saving**: Save model state after each training stage
- **Flexible Data Loading**: Support multiple data formats (JSON/CSV)
- **Catastrophic Forgetting Prevention**: Maintain performance on previous tasks

### Training Stages

- **Stage 1**: Initial training on first task (epochs: `stage1_epochs`)
- **Stage 2+**: Incremental training on subsequent tasks (epochs: `stage2_epochs`)

## Notes

- Make sure to have sufficient disk space for model checkpoints
- Training progress is automatically saved after each epoch
- Use `Ctrl+C` to safely stop training (checkpoint will be saved)
- For GPU training, change `device: "cpu"` to `device: "cuda"` in `default.yaml`
- Adjust batch sizes based on your available memory

## Troubleshooting

If you encounter issues:

1. **Configuration errors**: 
   - Verify all paths in `configs/default.yaml` are correct
   - Ensure `use_custom_train_test_split` flag matches your data format

2. **Data loading errors**:
   - Check that JSON/CSV files exist at specified paths
   - Verify data format matches the expected structure
   - Ensure `id2label.json` contains all necessary labels

3. **Memory errors**:
   - Reduce `train_batch_size` and `eval_batch_size`
   - Decrease `max_seq_length`
   - Use a smaller model (e.g., distilbert instead of bert-base)

4. **Model loading errors**:
   - Ensure previous checkpoints are compatible with current code version
   - Check that model architecture matches the configuration

## Quick Reference

### Key Configuration Parameters

| Parameter | Location | Purpose | Example Value |
|-----------|----------|---------|---------------|
| `task_name` | `configs/default.yaml` | Type of task | `"SentenceClassification"` |
| `dataset_name` | `configs/default.yaml` | Dataset to use | `"HWU64"` |
| `num_tasks` | `configs/default.yaml` | Number of incremental tasks | `7` |
| `class_per_task` | `configs/default.yaml` | Classes per task | `4` |
| `stage1_epochs` | `configs/default.yaml` | Epochs for first task | `12` |
| `stage2_epochs` | `configs/default.yaml` | Epochs for subsequent tasks | `8` |
| `use_custom_train_test_split` | `configs/constants.py` | Data loading method | `True` |

### Supported Models

- `bert-base-uncased`
- `distilbert/distilbert-base-uncased` (recommended for faster training)
- Any HuggingFace BERT-compatible model

### Supported Tasks

- **SentenceClassification**: Text classification tasks
- **RelationExtraction**: Entity relation extraction

## Tips for Best Results

1. **Start Small**: Test with fewer epochs and smaller batch sizes first
2. **Monitor Performance**: Check validation metrics after each task
3. **Adjust Task Split**: Modify `num_tasks` and `class_per_task` based on your dataset
4. **Use GPU**: Training on GPU (`device: "cuda"`) is significantly faster
5. **Data Quality**: Ensure your training data is clean and properly labeled

## Example Workflow - Initial Training

```bash
# 1. Clone the repository
git clone [your-repo-url]
cd [repo-name]

# 2. Install dependencies
pip install -r requirements.txt

# 3. Update configuration
nano configs/default.yaml  # Edit your settings

# 4. Set data loading method
nano configs/constants.py  # Set use_custom_train_test_split = True

# 5. Verify data files exist
ls round_4_train_test/train.csv
ls round_4_train_test/test.csv

# 6. Run training
python main.py
```

---

## Use Case 2: Continuing Training with New Documents

After you've trained your initial model and receive **new documents for classification**, you can continue incremental training without retraining from scratch.

### Scenario

You've already trained a model on several tasks (e.g., iterations 1-6), and now you have:
- A new set of documents to classify
- New labels/categories that weren't in previous training
- A saved model and memory from previous training

### Configuration Steps

#### Step 1: Update `configs/constants.py`

Add the following parameters to specify your previous model and training history:

```python
# Path to your previously trained model
my_model = "/home/ntlpt19/TF_testing_EXT/dummy_data/lic_data/model_save/slow_model"

# Path to memory file from last training task (e.g., memory_data_5.pkl from iteration 5)
memory_path = '/home/ntlpt19/TF_testing_EXT/dummy_data/lic_data/model_save/memory_data_5.pkl'

# New task key for current iteration (e.g., "itr7" for 7th iteration)
new_task_key = "itr7"

# Enable custom train/test split
use_custom_train_test_split = True

# Define your training history - which labels were trained in which iterations
manual_task_plan = {
    "itr1": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected"
    ],
    "itr2": [
        "Policy is not issued even after premium is paid",
        "Delay in refund of excess premium",
        "Premium receipt not issued",
        "Grievance regarding maturity claim not settled"
    ],
    # ... add all previous iterations ...
    "itr6": [
        "Previous labels from iteration 6"
    ],
    "itr7": [
        "New label 1 for current training",
        "New label 2 for current training",
        "New label 3 for current training",
        "New label 4 for current training"
    ]
}
```

#### Step 2: Update Version Information

Update both `configs/constants.py` and `configs/default.yaml` with:
- Current version number
- Labels for the current iteration (itr7)

#### Step 3: Prepare Your New Training Data

Place your new data in CSV format at:
- `round_4_train_test/train.csv`
- `round_4_train_test/test.csv`

### CSV Data Format for Incremental Training

Your CSV files should include both **old labels** (from previous training) and **new labels** (for current iteration):

```csv
input_ids,attention_mask,labels,org_query,org_label
"[101, 1045, 2215, 2000, 6542, ...]","[1, 1, 1, 1, 1, ...]",0,"I want to cancel my policy","Disputes concerning eligibility of surrender value"
"[101, 2026, 3952, 2003, 2025, ...]","[1, 1, 1, 1, 1, ...]",1,"My claim is not settled","New label 1 for current training"
```

**Column Description:**
- `input_ids`: Tokenized input (list of integers)
- `attention_mask`: Attention mask (list of 1s and 0s)
- `labels`: Original label ID (can be any number, will be remapped)
- `org_query`: Original text query (optional, for reference)
- `org_label`: Original label name (text)

**Example CSV Entry:**
```csv
input_ids,attention_mask,labels,org_query,org_label
"[101, 1045, 2215, 2000, 6542, 2026, 3952, 102]","[1, 1, 1, 1, 1, 1, 1, 1]",5,"I want to cancel my claim","New label 1 for current training"
```

### How Label Filtering Works

The system uses `incremental_label2id` to intelligently map your labels:

1. **Label Matching**: The code reads the `org_label` column from your CSV
2. **Case-Insensitive Search**: It searches for matching labels in `incremental_label2id`
3. **ID Assignment**: When a match is found, it assigns the correct label ID
4. **Data Filtering**: Only data matching labels in `incremental_label2id` is kept

**Simple Example:**

```python
# Your incremental_label2id mapping
incremental_label2id = {
    "New label 1 for current training": 24,  # Continues from previous max ID (23)
    "New label 2 for current training": 25,
    "New label 3 for current training": 26,
    "New label 4 for current training": 27
}

# CSV row with org_label = "New label 1 for current training"
# Original labels = 5 (from CSV)
# After filtering: labels = 24 (remapped using incremental_label2id)
```

**Filtering Process:**
1. Read CSV row: `org_label = "New label 1 for current training"`
2. Search in `incremental_label2id`: Found! → ID = 24
3. Replace label: `labels = 24` (instead of original 5)
4. Create clean data dictionary:
   ```python
   {
       'input_ids': [101, 1045, 2215, 2000, 6542, 2026, 3952, 102],
       'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1],
       'labels': 24  # Remapped ID
   }
   ```

**Why This Works:**
- ✅ Maintains consistency with previous training (labels 0-23 from itr1-6)
- ✅ Adds new labels starting from 24 (for itr7)
- ✅ Filters out any data not relevant to current training
- ✅ Ensures label IDs don't conflict with previous iterations

### Running Incremental Training

Once configuration is complete, run:

```bash
python run_incremental_train.py
```

This script will:
1. Load your previously trained model from `my_model` path
2. Load memory from previous training (`memory_path`)
3. Read the new task labels from `manual_task_plan[new_task_key]`
4. Load and filter training data using `incremental_label2id`
5. Continue training on new labels while preserving old knowledge
6. Save updated model and memory for future iterations

### Complete Incremental Training Workflow

```bash
# 1. Update constants.py with model path, memory path, and task history
nano configs/constants.py

# 2. Update version number in config files
nano configs/default.yaml

# 3. Prepare your new training data (train.csv and test.csv)
# Include both old and new labels in the CSV

# 4. Verify data files exist
ls round_4_train_test/train.csv
ls round_4_train_test/test.csv

# 5. Run incremental training
python run_incremental_train.py

# 6. New model and memory will be saved for next iteration
```

### Key Differences: main.py vs run_incremental_train.py

| Aspect | `main.py` | `run_incremental_train.py` |
|--------|-----------|----------------------------|
| **Use Case** | Initial training from scratch | Continue training with new data |
| **Model Loading** | Initializes new model | Loads existing trained model |
| **Memory** | Creates new memory | Loads existing memory |
| **Task Plan** | Uses automatic task splitting | Uses `manual_task_plan` |
| **Label IDs** | Starts from 0 | Continues from previous max ID |
| **When to Use** | First time training | Adding new categories/documents |

### Visual Summary: Incremental Training Flow

```
Previous Training (itr1-6)
    ├── Trained Model → saved to my_model path
    ├── Memory → saved as memory_data_5.pkl
    └── Labels 0-23 → defined in manual_task_plan

                    ↓

New Documents Arrive (itr7)
    ├── New labels 24-27 → defined in manual_task_plan["itr7"]
    ├── CSV data → includes org_label column
    └── incremental_label2id → maps labels to IDs

                    ↓

run_incremental_train.py
    ├── Loads previous model and memory
    ├── Filters CSV data using incremental_label2id
    ├── Remaps label IDs (e.g., 5 → 24)
    └── Trains only on new labels while preserving old knowledge

                    ↓

Updated Model
    ├── Can classify labels 0-27
    ├── New memory saved for itr8
    └── Ready for next iteration
```

### Important Notes for Incremental Training

⚠️ **Critical Requirements:**

1. **Label Continuity**: Label IDs must be continuous across iterations
   - itr1-6: Labels 0-23
   - itr7: Labels 24-27 (starts from 24, not 0)

2. **Memory Compatibility**: Always use memory from the immediately previous iteration
   - For itr7, use memory_data_6.pkl (not memory_data_5.pkl)

3. **Task History**: `manual_task_plan` must include ALL previous iterations
   - Missing iterations will cause label mapping errors

4. **CSV Requirements**: 
   - Must include `org_label` column for label matching
   - Can include data from both old and new labels
   - Only data matching `incremental_label2id` will be used

5. **Version Tracking**: Update version numbers in both:
   - `configs/constants.py`
   - `configs/default.yaml`

### Troubleshooting Incremental Training

**Issue: Label Mismatch Error**
- **Cause**: `org_label` in CSV doesn't match any key in `incremental_label2id`
- **Solution**: Check spelling and ensure case-insensitive matching

**Issue: Model Not Loading**
- **Cause**: Incorrect `my_model` path
- **Solution**: Verify the path exists and contains model files

**Issue: Memory Loading Error**
- **Cause**: Wrong memory file version or path
- **Solution**: Ensure memory file matches the previous iteration

**Issue: Duplicate Label IDs**
- **Cause**: Label IDs overlap with previous iterations
- **Solution**: Start new labels from `max(previous_labels) + 1`

## License

[Add your license information here]

## Contact

[Add your contact information or contribution guidelines here]