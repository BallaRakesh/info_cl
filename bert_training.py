import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Load and filter CSV data based on labels
def load_csv_data(train_file_path, filter_labels):
    """
    Load CSV file and filter by specified labels
    """
    # Load training data
    train_df = pd.read_csv(train_file_path)
    
    print(f"Original training samples: {len(train_df)}")
    
    # Filter by labels if provided
    # if filter_labels is not None and len(filter_labels) > 0:
    #     train_df = train_df[train_df['org_label'].isin(filter_labels)]
    #     print(f"Filtered training samples: {len(train_df)}")
        
    if filter_labels is not None and len(filter_labels) > 0:
        train_df = train_df[
            train_df['org_label'].str.strip().str.lower().isin(
                [lbl.strip().lower() for lbl in filter_labels]
            )
        ]
        print(f"Filtered training samples: {len(train_df)}")


    # Extract queries and categories
    train_queries = train_df['org_query'].astype(str).str.strip().tolist()
    train_categories = train_df['org_label'].astype(str).str.strip().tolist()
    
    # Get unique categories
    all_categories = list(set(train_categories))
    print(f"Found {len(all_categories)} unique categories: {all_categories}")
    
    return train_queries, train_categories, all_categories

# Step 2: Load test data for evaluation only
def load_test_data(test_file_path, filter_labels):
    """
    Load test CSV file and filter by specified labels (for evaluation only)
    """
    test_df = pd.read_csv(test_file_path)
    
    print(f"Original test samples: {len(test_df)}")
    
    # Filter by labels if provided
    # if filter_labels is not None and len(filter_labels) > 0:
    #     test_df = test_df[test_df['org_label'].isin(filter_labels)]
    #     print(f"Filtered test samples: {len(test_df)}")
    
    # Filter by labels if provided
    if filter_labels is not None and len(filter_labels) > 0:
        test_df = test_df[
            test_df['org_label'].str.strip().str.lower().isin(
                [lbl.strip().lower() for lbl in filter_labels]
            )
        ]
        print(f"Filtered test samples: {len(test_df)}")

    # Extract queries and categories
    test_queries = test_df['org_query'].astype(str).str.strip().tolist()
    test_categories = test_df['org_label'].astype(str).str.strip().tolist()
    
    return test_queries, test_categories

# Step 3: Prepare datasets with train/val split
def prepare_datasets(train_queries, train_categories, val_split=0.1):
    """
    Prepare train and validation datasets with label encoding
    Split 10% of training data for validation
    """
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_categories)
    
    # Create label mappings
    unique_labels = label_encoder.classes_
    id2label = {i: label for i, label in enumerate(unique_labels)}
    label2id = {label: i for i, label in enumerate(unique_labels)}
    
    # Encode labels
    train_labels = label_encoder.transform(train_categories)
    
    # Split train into train and validation
    train_queries_split, val_queries, train_labels_split, val_labels = train_test_split(
        train_queries, train_labels, test_size=val_split, random_state=42, stratify=train_labels
    )
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'query': train_queries_split,
        'labels': train_labels_split
    })
    
    val_dataset = Dataset.from_dict({
        'query': val_queries,
        'labels': val_labels
    })
    
    return train_dataset, val_dataset, id2label, label2id, len(unique_labels), label_encoder

# Step 4: Prepare test dataset for evaluation
def prepare_test_dataset(test_queries, test_categories, label_encoder):
    """
    Prepare test dataset using the same label encoder from training
    """
    test_labels = label_encoder.transform(test_categories)
    
    test_dataset = Dataset.from_dict({
        'query': test_queries,
        'labels': test_labels
    })
    
    return test_dataset

# Step 5: Preprocessing function
def preprocess_function(examples, tokenizer):
    """
    Tokenize the text data
    """
    return tokenizer(examples["query"], truncation=True, padding=True, max_length=512)

# Step 6: Compute metrics function
def compute_metrics(eval_pred):
    """
    Compute accuracy for evaluation
    """
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Step 7: Generate confusion matrix
def save_confusion_matrix(y_true, y_pred, labels, output_path):
    """
    Generate and save confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(max(10, len(labels)), max(8, len(labels))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()
    
    return cm


from sklearn.metrics import confusion_matrix
from texttable import Texttable

def confusion_matrix_view(true_label, pred_label, labels, logger=None):
    cf_matrix = confusion_matrix(true_label, pred_label)

    table = Texttable()
    table.add_row([" "] + [i for i in labels])  # full labels
    table.set_max_width(5000)

    for idx, r in enumerate(cf_matrix):
        table.add_row([labels[idx]] + [str(i) for i in cf_matrix[idx]])

    table_str = table.draw()

    if logger:
        logger.info("\n" + table_str)

    return table_str


# Step 8: Evaluate model and generate metrics
def evaluate_model(trainer, test_dataset, id2label, iteration_name, output_dir):
    """
    Evaluate model and generate confusion matrix and classification report
    """
    print("\n" + "="*50)
    print(f"Evaluating on test set for {iteration_name}...")
    print("="*50)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids
    
    # Convert numeric labels to string labels
    pred_label_names = [id2label[i] for i in pred_labels]
    true_label_names = [id2label[i] for i in true_labels]
    
    # Get unique labels in sorted order
    unique_labels = sorted(list(set(true_label_names)))
    
    # Generate confusion matrix
    cm_output_path = f"{output_dir}/{iteration_name}_confusion_matrix.png"
    cmatrix = confusion_matrix_view(true_label_names, pred_label_names, 
                               unique_labels, logger=None)
    # cm = save_confusion_matrix(true_label_names, pred_label_names, 
    #                            unique_labels, 
    #                            output_path=cm_output_path)
    
    # Save confusion matrix as CSV
    # cm_df = pd.DataFrame(cm, index=unique_labels, columns=unique_labels)
    # cm_csv_path = f"{output_dir}/{iteration_name}_confusion_matrix.csv"
    # cm_df.to_csv(cm_csv_path)
    # print(f"Confusion matrix CSV saved to: {cm_csv_path}")
    
    # Generate classification report
    report = classification_report(true_label_names, pred_label_names, 
                                   target_names=unique_labels, 
                                   digits=4)
    print("\n" + "="*50)
    print(f"Classification Report for {iteration_name}:")
    print("="*50)
    print(report)
    
    cm_txt_path = f"{output_dir}/{iteration_name}_metrics.txt"

    with open(cm_txt_path, "w", encoding="utf-8") as f:
        f.write("===== CONFUSION MATRIX =====\n")
        f.write(cmatrix)
        f.write("\n\n===== CLASSIFICATION REPORT =====\n")
        f.write(report)
    
    # Save classification report
    # report_dict = classification_report(true_label_names, pred_label_names, 
    #                                    target_names=unique_labels, 
    #                                    digits=4, 
    #                                    output_dict=True)
    # report_df = pd.DataFrame(report_dict).transpose()
    # report_csv_path = f"{output_dir}/{iteration_name}_classification_report.csv"
    # report_df.to_csv(report_csv_path)
    # print(f"Classification report saved to: {report_csv_path}")
    
    # Get test accuracy
    test_results = trainer.evaluate(test_dataset)
    print(f"\nTest Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Test Loss: {test_results['eval_loss']:.4f}")
    
    return report, cmatrix

# Main fine-tuning function for single iteration
def finetune_iteration(train_csv_path, test_csv_path, filter_labels, iteration_name,
                       model_name="bert-base-uncased", 
                       base_output_dir="./fine_tuned_models",
                       num_epochs=30):
    """
    Fine-tune the model for a single iteration
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        filter_labels: List of labels to filter
        iteration_name: Name of the iteration (e.g., 'itr1')
        model_name: Pretrained model name
        base_output_dir: Base directory to save models
        num_epochs: Number of training epochs
    """
    # Create iteration-specific output directory
    output_dir = os.path.join(base_output_dir, iteration_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(f"Starting Iteration: {iteration_name}")
    print(f"Filter Labels: {filter_labels}")
    print("="*70)
    
    print("\n" + "="*50)
    print("Loading training data...")
    print("="*50)
    
    train_queries, train_categories, all_categories = load_csv_data(
        train_csv_path, filter_labels
    )
    
    if len(train_queries) == 0:
        raise ValueError(f"No training data found for iteration {iteration_name}. Please check your filter labels.")
    
    print("\n" + "="*50)
    print("Preparing datasets (90% train, 10% validation)...")
    print("="*50)
    
    train_dataset, val_dataset, id2label, label2id, num_labels, label_encoder = prepare_datasets(
        train_queries, train_categories, val_split=0.1
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Number of classes: {num_labels}")
    
    print("\n" + "="*50)
    print("Loading tokenizer and model...")
    print("="*50)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("Preprocessing datasets...")
    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), 
        batched=True
    )
    
    val_dataset = val_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), 
        batched=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    from transformers import TrainerCallback
    
    class SaveEveryNEpochsCallback(TrainerCallback):
        def __init__(self, n=10):
            self.n = n
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if state.epoch and int(state.epoch) % self.n == 0:
                print(f"Saving checkpoint at epoch {int(state.epoch)}")
                control.should_save = True
            else:
                control.should_save = False
            return control
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Using validation split from training data
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[SaveEveryNEpochsCallback(n=10)]
    )
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train()
    
    print("\n" + "="*50)
    print("Saving best model...")
    print("="*50)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mappings
    label_mappings = {
        'id2label': id2label,
        'label2id': label2id
    }
    with open(f"{output_dir}/label_mappings.json", 'w') as f:
        json.dump(label_mappings, f, indent=2)
    
    print(f"Model saved to: {output_dir}")
    print(f"Label mappings saved to: {output_dir}/label_mappings.json")
    
    # Now load and evaluate on test set
    print("\n" + "="*50)
    print("Loading test data for evaluation...")
    print("="*50)
    
    test_queries, test_categories = load_test_data(test_csv_path, filter_labels)
    
    if len(test_queries) == 0:
        print(f"Warning: No test data found for iteration {iteration_name}. Skipping test evaluation.")
        return trainer, id2label, label2id, None, None
    
    test_dataset = prepare_test_dataset(test_queries, test_categories, label_encoder)
    
    test_dataset = test_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), 
        batched=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate and generate metrics
    report_dict, cm = evaluate_model(trainer, test_dataset, id2label, iteration_name, output_dir)
    
    return trainer, id2label, label2id, report_dict, cm

# Main function to handle multiple iterations
def finetune_multiple_iterations(train_csv_path, test_csv_path, iterations_config,
                                 model_name="bert-base-uncased",
                                 base_output_dir="./fine_tuned_models",
                                 num_epochs=30):
    """
    Fine-tune the model for multiple iterations
    
    Args:
        train_csv_path: Path to training CSV file
        test_csv_path: Path to test CSV file
        iterations_config: Dictionary with iteration names as keys and label lists as values
                          Example: {'itr1': ['label1', 'label2'], 'itr2': ['label3']}
        model_name: Pretrained model name
        base_output_dir: Base directory to save models
        num_epochs: Number of training epochs
    """
    results = {}
    
    for iteration_name, filter_labels in iterations_config.items():
        try:
            trainer, id2label, label2id, report_dict, cm = finetune_iteration(
                train_csv_path=train_csv_path,
                test_csv_path=test_csv_path,
                filter_labels=filter_labels,
                iteration_name=iteration_name,
                model_name=model_name,
                base_output_dir=base_output_dir,
                num_epochs=num_epochs
            )
            
            results[iteration_name] = {
                'id2label': id2label,
                'label2id': label2id,
                'classification_report': report_dict,
                'confusion_matrix': cm
            }
            
            print(f"\n✓ Iteration {iteration_name} completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error in iteration {iteration_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

# Example usage
if __name__ == "__main__":
    # File paths for your CSV files
    train_csv_path = "/home/ng6281/Rupankar_Dev/gopal/rakesh/code/Continual_learning_poc/InfoCL/round_4_train_test/train.csv"
    test_csv_path = "/home/ng6281/Rupankar_Dev/gopal/rakesh/code/Continual_learning_poc/InfoCL/round_4_train_test/test.csv"
    
    # Define iterations with labels to filter
    iterations_config_old = {
    "itr1": [
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon"
    ],
    "itr2": [
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon",
        "Alteration in policy not effected.",
        "No Response for recording Change of address",
        "Statement of account not received",
        "Non-payment of penal interest"
    ],
    "itr3": [
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon",
        "Alteration in policy not effected.",
        "No Response for recording Change of address",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Non-receipt of Premium receipt",
        "Dispute concerning statement of account or premium position statement"
    ],
    "itr4": [
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon",
        "Alteration in policy not effected.",
        "No Response for recording Change of address",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Non-receipt of Premium receipt",
        "Dispute concerning statement of account or premium position statement",
        "Dispute concerning claim value",
        "Commutation value/cash option not paid",
        "Request for Servicing Branch transfer is not effected",
        "Maturity claim is not paid"
    ],
    'itr5':[
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon",
        "Alteration in policy not effected.",
        "No Response for recording Change of address",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Non-receipt of Premium receipt",
        "Dispute concerning statement of account or premium position statement",
        "Dispute concerning claim value",
        "Commutation value/cash option not paid",
        "Request for Servicing Branch transfer is not effected",
        "Maturity claim is not paid",
        "Policy Benefit option not effected",
        "Response for processing or payment of Policy Loan is not sent",
        "Annuity/pension instalments not paid",
        "Non-receipt of Duplicate policy"
    ],
    "itr6": [
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon",
        "Alteration in policy not effected.",
        "No Response for recording Change of address",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Non-receipt of Premium receipt",
        "Dispute concerning statement of account or premium position statement",
        "Dispute concerning claim value",
        "Commutation value/cash option not paid",
        "Request for Servicing Branch transfer is not effected",
        "Maturity claim is not paid",
        "Policy Benefit option not effected",
        "Response for processing or payment of Policy Loan is not sent",
        "Annuity/pension instalments not paid",
        "Non-receipt of Duplicate policy",
        "Requirements for revival not communicated or raised",
        "Response for issuance of duplicate policy is not sent",
        "After submission of all reinstatement (revival) requirements"
    ],
    "itr7": [
        "Survival Benefit is not paid",
        "Surrender Value not paid",
        "No Response for noting a new nomination",
        "Payment of premium not acted upon or wrongly acted upon",
        "Alteration in policy not effected.",
        "No Response for recording Change of address",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Non-receipt of Premium receipt",
        "Dispute concerning statement of account or premium position statement",
        "Dispute concerning claim value",
        "Commutation value/cash option not paid",
        "Request for Servicing Branch transfer is not effected",
        "Maturity claim is not paid",
        "Policy Benefit option not effected",
        "Response for processing or payment of Policy Loan is not sent",
        "Annuity/pension instalments not paid",
        "Non-receipt of Duplicate policy",
        "Requirements for revival not communicated or raised",
        "Response for issuance of duplicate policy is not sent",
        "After submission of all reinstatement (revival) requirements",
        "Disputes concerning eligibility of surrender value",
        "Disputes concerning correctness of surrender value",
        "Reinstatement requirements raised by Insurer not acceptable"
    ]
}

    iterations_config = {
    "itr1": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected"
    ],
    "itr2": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected",
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest"
    ],
    "itr3": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected",
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Survival Benefit is not paid",
        "Non-receipt of Duplicate policy",
        "No Response for recording Change of address"
    ],
    "itr4": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected",
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Survival Benefit is not paid",
        "Non-receipt of Duplicate policy",
        "No Response for recording Change of address",
        "Disputes concerning correctness of surrender value",
        "Payment of premium not acted upon or wrongly acted upon",
        "No Response for noting a new nomination"
    ],
    "itr5": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected",
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Survival Benefit is not paid",
        "Non-receipt of Duplicate policy",
        "No Response for recording Change of address",
        "Disputes concerning correctness of surrender value",
        "Payment of premium not acted upon or wrongly acted upon",
        "No Response for noting a new nomination",
        "Non-receipt of Premium receipt",
        "Maturity claim is not paid",
        "Commutation value/cash option not paid"
    ],
    "itr6": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected",
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Survival Benefit is not paid",
        "Non-receipt of Duplicate policy",
        "No Response for recording Change of address",
        "Disputes concerning correctness of surrender value",
        "Payment of premium not acted upon or wrongly acted upon",
        "No Response for noting a new nomination",
        "Non-receipt of Premium receipt",
        "Maturity claim is not paid",
        "Commutation value/cash option not paid",
        "Dispute concerning claim value",
        "Annuity/pension instalments not paid",
        "Reinstatement requirements raised by Insurer not acceptable",
        "Surrender Value not paid"
    ],
    "itr7": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected",
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest",
        "Survival Benefit is not paid",
        "Non-receipt of Duplicate policy",
        "No Response for recording Change of address",
        "Disputes concerning correctness of surrender value",
        "Payment of premium not acted upon or wrongly acted upon",
        "No Response for noting a new nomination",
        "Non-receipt of Premium receipt",
        "Maturity claim is not paid",
        "Commutation value/cash option not paid",
        "Dispute concerning claim value",
        "Annuity/pension instalments not paid",
        "Reinstatement requirements raised by Insurer not acceptable",
        "Surrender Value not paid",
        "Alteration in policy not effected.",
        "After submission of all reinstatement (revival) requirements",
        "Dispute concerning statement of account or premium position statement"
    ]
    }

    
    try:
        # Fine-tune the model for all iterations
        results = finetune_multiple_iterations(
            train_csv_path=train_csv_path,
            test_csv_path=test_csv_path,
            iterations_config=iterations_config,
            model_name="distilbert/distilbert-base-uncased",#"bert-base-uncased",
            base_output_dir="./fine_tuned_models",
            num_epochs=15
        )
        
        print("\n" + "="*70)
        print("All iterations completed!")
        print("="*70)
        
        # Print summary
        for iteration_name, result in results.items():
            print(f"\n{iteration_name}:")
            print(f"  - Model saved in: ./fine_tuned_models/{iteration_name}/")
            print(f"  - Number of classes: {len(result['id2label'])}")
            
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure your CSV files exist and update the file paths.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()