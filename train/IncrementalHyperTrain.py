import logging
import os
import json
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, Callable, Tuple, Dict, List
from types import SimpleNamespace
from pathlib import Path
from functools import partial
from copy import deepcopy
import pickle
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    DataCollator,
    TrainerCallback,
    EvalPrediction,
    set_seed,
    DataCollatorWithPadding,
)

from data import BaseData, BaseDataset
from .DefaultTrain import default_train
from .AdversarialTrain import adversarial_train
from .MoCoTrain import moco_train, get_moco_hidden_states
from .DefaultEvaluate import default_evaluate
from .NCMEvaluate import ncm_evaluate
from .NewOldTrain import new_old_train
from utils import select_exemplars, compute_forgetting_rate, get_hidden_states
from configs.constants import use_custom_train_test_split, manual_task_plan, my_model, memory_path,save_dir
from data import (
    FewRelData,
    TACREDData,
    MAVENData,
    HWU64Data,
)
from model import (
    BertForRelationExtraction,
    BertForSentenceClassification,
    BertForEventDetection,
    BertMoCoForRelationExtraction,
    BertMoCoForSentenceClassificationDeberta, BertMoCoForSentenceClassificationDistbert, BertMoCoForSentenceClassificationBert,
    BertMoCoForEventDetection,
)


task_to_data_reader = {
    "FewRel": FewRelData,
    "TACRED": TACREDData,
    "MAVEN": MAVENData,
    "HWU64": HWU64Data,
}

task_to_additional_special_tokens = {
    "RelationExtraction": ["[E11]", "[E12]", "[E21]", "[E22]"]
}

task_to_model = {
    "RelationExtraction": BertForRelationExtraction,
    "SentenceClassification": BertForSentenceClassification,
    "EventDetection": BertForEventDetection,
}


logger = logging.getLogger(__name__)


def incremental_hyper_train(
    args: SimpleNamespace = None,
    data_collator: Optional[DataCollator] = None,
    model_checkpoint_path: Optional[str] = None,
    previous_memory_path: Optional[str] = None,
    new_task_key: str = None,  # e.g., "itr8" for new samples
):
    """
    Incremental training function that:
    1. Loads a pre-trained model from checkpoint
    2. Loads previous memory data (exemplars from old tasks)
    3. Trains on new samples as a single new task
    4. Continues the two-stage training approach

    Args:
        args: Training arguments
        data_collator: Optional custom data collator
        model_checkpoint_path: Path to pre-trained model checkpoint (overrides my_model from constants)
        previous_memory_path: Path to previous memory data directory (default: ../model_save)
        new_task_key: Key in manual_task_plan for new samples (e.g., "itr8")
    """

    # Use provided checkpoint path or fall back to constants
    model_checkpoint_path = my_model
    previous_memory_path = memory_path


    logger.info(f"***** Incremental Training *****")
    logger.info(f"Loading pre-trained model from: {model_checkpoint_path}")
    logger.info(f"Loading previous memory from: {previous_memory_path}")

    # Setup tokenizer
    additional_special_tokens = task_to_additional_special_tokens[args.task_name] \
        if args.task_name in task_to_additional_special_tokens else []

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
        additional_special_tokens=additional_special_tokens,
    )

    if data_collator is None:
        default_data_collator = DataCollatorWithPadding(tokenizer)

    stage1_data_collator = default_data_collator
    stage2_data_collator = default_data_collator
    evaluate_data_collator = default_data_collator

    # Setup training functions based on stage types
    if args.stage1_type == 'default':
        stage1_train = default_train
    elif args.stage1_type == 'moco':
        stage1_train = moco_train
        task_to_model["RelationExtraction"] = BertMoCoForRelationExtraction
        if args.model_name_or_path == "bert-base-uncased":
            task_to_model["SentenceClassification"] = BertMoCoForSentenceClassificationBert
        elif args.model_name_or_path == "microsoft/deberta-v3-base":
            task_to_model["SentenceClassification"] = BertMoCoForSentenceClassificationDeberta
        elif args.model_name_or_path == "distilbert/distilbert-base-uncased":
            task_to_model["SentenceClassification"] = BertMoCoForSentenceClassificationDistbert
        task_to_model["EventDetection"] = BertMoCoForEventDetection
    else:
        raise NotImplementedError

    if args.stage2_type == 'default':
        stage2_train = default_train
    elif args.stage2_type == 'adversarial':
        stage2_train = adversarial_train
    elif args.stage2_type == 'moco':
        stage2_train = moco_train
    elif args.stage2_type == 'new_old':
        stage2_train = new_old_train
    else:
        raise NotImplementedError

    if args.ncm_evaluate:
        evaluate = ncm_evaluate
    else:
        evaluate = default_evaluate

    # Load pre-trained model first to get old label mappings
    ModelForContinualLearning = task_to_model[args.task_name]

    logger.info("=" * 80)
    logger.info("Step 1: Loading pre-trained model to extract old label mappings...")
    logger.info("=" * 80)

    if not os.path.isdir(model_checkpoint_path):
        raise ValueError(f"Model checkpoint path '{model_checkpoint_path}' is not a directory. "
                        f"Expected a directory with config.json and model.safetensors")

    # Load old model config to get old labels
    # First load the config file manually to handle list format
    import json

    config_path = os.path.join(model_checkpoint_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # ---- Fix id2label if it's a list ----
    if isinstance(config_dict.get("id2label"), list):
        config_dict["id2label"] = {i: label for i, label in enumerate(config_dict["id2label"])}

    # ---- Fix label2id if necessary ----
    if isinstance(config_dict.get("label2id"), list):
        config_dict["label2id"] = {label: i for i, label in enumerate(config_dict["label2id"])}

    # ---- Save fixed config to a temp file ----
    fixed_config_path = os.path.join(model_checkpoint_path, "fixed_config.json")
    with open(fixed_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
        
    old_config = AutoConfig.from_pretrained(fixed_config_path)

    old_id2label = old_config.id2label
    old_label2id = old_config.label2id

    num_old_labels = len(old_label2id)
    logger.info(f"Loaded old model with {num_old_labels} labels")
    logger.info(f"Old labels: {list(old_label2id.keys())[:10]}...")  # Show first 10

    # Load data (contains only new labels)
    logger.info("=" * 80)
    logger.info("Step 2: Loading data with new labels...")
    logger.info("=" * 80)

    set_seed(args.seed)
    data = task_to_data_reader[args.dataset_name](args)

    new_data_label2id = data.label2id
    new_data_id2label = data.id2label
    logger.info(f"Data contains {len(new_data_label2id)} new labels")
    logger.info(f"New labels from data: {list(new_data_label2id.keys())}")

    # Determine new task labels from manual_task_plan
    if new_task_key is None:
        # Find the next task key (e.g., if we have itr1-itr7, find itr8)
        existing_keys = sorted([k for k in manual_task_plan.keys()])
        logger.info(f"Existing task keys in manual_task_plan: {existing_keys}")
        if len(existing_keys) == 0:
            raise ValueError("No tasks found in manual_task_plan. Please add your new task data.")
        # Use the last key as the new task
        new_task_key = existing_keys[-1]
        logger.warning(f"No new_task_key provided. Using last key in manual_task_plan: {new_task_key}")

    if new_task_key not in manual_task_plan:
        raise ValueError(f"Task key '{new_task_key}' not found in manual_task_plan. "
                        f"Please add it to configs/constants.py")

    new_labels_names = manual_task_plan[new_task_key]
    logger.info(f"New task key: {new_task_key}")
    logger.info(f"New labels from manual_task_plan: {new_labels_names}")

    # Create extended label mappings (old + new)
    logger.info("=" * 80)
    logger.info("Step 3: Extending label mappings (old + new)...")
    logger.info("=" * 80)

    extended_label2id = old_label2id.copy()
    extended_id2label = old_id2label.copy()

    # Add new labels with sequential IDs starting from num_old_labels
    new_label_ids = []
    next_id = num_old_labels

    for new_label_name in new_labels_names:
        normalized_label = new_label_name.strip().lower()

        # Check if this label already exists in old labels (case-insensitive)
        label_exists = False
        for old_label in old_label2id.keys():
            if old_label.strip().lower() == normalized_label:
                logger.warning(f"Label '{new_label_name}' already exists in old labels as '{old_label}'. Using existing ID.")
                new_label_ids.append(old_label2id[old_label])
                label_exists = True
                break

        # If label doesn't exist, add it with new ID
        if not label_exists:
            # Find the exact match in new_data_label2id to preserve original casing
            matched_label = None
            for data_label in new_data_label2id.keys():
                if data_label.strip().lower() == normalized_label:
                    matched_label = data_label
                    break

            if matched_label is None:
                logger.warning(f"Label '{new_label_name}' not found in data labels. Skipping.")
                continue

            extended_label2id[matched_label] = next_id
            extended_id2label[next_id] = matched_label
            new_label_ids.append(next_id)
            logger.info(f"Added new label: '{matched_label}' -> ID {next_id}")
            next_id += 1

    if len(new_label_ids) == 0:
        raise ValueError("No valid new labels found. Check your manual_task_plan configuration.")

    num_total_labels = len(extended_label2id)
    logger.info(f"Extended label mappings created:")
    logger.info(f"  - Old labels: {num_old_labels}")
    logger.info(f"  - New labels: {len(new_label_ids)}")
    logger.info(f"  - Total labels: {num_total_labels}")
    logger.info(f"New label IDs: {new_label_ids}")
    # Update args with extended label mappings
    args.id2label = extended_id2label
    args.label2id = extended_label2id
    label_list = extended_id2label

    # Load pre-trained model with extended label config
    logger.info("=" * 80)
    logger.info("Step 4: Loading model and extending classifier head...")
    logger.info("=" * 80)

    # Create config with extended labels
    config = AutoConfig.from_pretrained(
        fixed_config_path,
        num_labels=num_total_labels,
        classifier_dropout=args.classifier_dropout,
        label2id=extended_label2id,
        id2label=extended_id2label,
    )
    config.global_args = args
    # Load model with old weights (will have mismatch in classifier head)
    model = ModelForContinualLearning.from_pretrained(
        model_checkpoint_path,
        config=config,
        ignore_mismatched_sizes=True,  # Important: allows loading with different classifier size
    )

    logger.info(f"Model loaded successfully with extended classifier head!")
    logger.info(f"Classifier head extended from {num_old_labels} to {num_total_labels} classes")

    # Load previous memory data
    logger.info("=" * 80)
    logger.info("Step 5: Loading previous memory data...")
    logger.info("=" * 80)
    memory_data = {}
    memory_file_path = None
    new_task_id = 0

    # Check if previous_memory_path is a file or directory
    if os.path.isfile(previous_memory_path):
        # Direct file path
        memory_file_path = previous_memory_path
        logger.info(f"Loading memory from file: {memory_file_path}")

        with open(memory_file_path, "rb") as f:
            memory_data = pickle.load(f)

        logger.info(f"Loaded memory data for {len(memory_data)} previous classes")

        # Extract task_id from filename (e.g., memory_data_5.pkl -> task_id = 6 for next task)
        filename = os.path.basename(memory_file_path)
        if "memory_data_" in filename:
            try:
                last_task_id = int(filename.split("_")[-1].replace(".pkl", ""))
                new_task_id = last_task_id + 1
            except ValueError:
                logger.warning(f"Could not extract task_id from filename '{filename}'. Using default task_id=0")
                new_task_id = 0

    elif os.path.isdir(previous_memory_path):
        # Directory path - find the latest memory file
        memory_files = sorted([f for f in os.listdir(previous_memory_path)
                              if f.startswith("memory_data_") and f.endswith(".pkl")])

        if len(memory_files) > 0:
            # Load the latest memory file
            latest_memory_file = memory_files[-1]
            memory_file_path = os.path.join(previous_memory_path, latest_memory_file)
            logger.info(f"Loading memory from: {memory_file_path}")

            with open(memory_file_path, "rb") as f:
                memory_data = pickle.load(f)

            logger.info(f"Loaded memory data for {len(memory_data)} previous classes")

            # Extract task_id from filename (e.g., memory_data_6.pkl -> task_id = 7 for next task)
            last_task_id = int(memory_files[-1].split("_")[-1].replace(".pkl", ""))
            new_task_id = last_task_id + 1
        else:
            logger.warning("No previous memory data found in directory. Starting with empty memory.")
    else:
        logger.warning(f"Memory path '{previous_memory_path}' not found. Starting with empty memory.")

    logger.info(f"Starting incremental training as Task {new_task_id + 1}")

    # Load data
    if use_custom_train_test_split:
        data.load_train_test_data(incremental_label2id=extended_label2id)
    else:
        data.read_and_preprocess(tokenizer, seed=args.seed)
    model.to(args.device)

    # Get seen labels (all previous + new)
    seen_label_ids = list(memory_data.keys()) + new_label_ids
    seen_label_ids = [lid for lid in seen_label_ids if lid != -1]
    seen_target_names = [label_list[tmp_label] for tmp_label in seen_label_ids]

    cur_target_names = [label_list[tmp_label] for tmp_label in new_label_ids]

    # Prepare memory representations
    cur_memory_repr, cur_memory_labels = [], []
    old_memory_repr, old_memory_labels = None, None
    '''
    update the below logic to handle the memory repr loading properly
    1) if memory was loaded from a file, look for the repr file in the same directory
    2) or use the original script for making the memory repr
    '''
    if len(memory_data) > 0:
        # Load previous memory representations
        repr_file_path = None

        # Determine where to look for memory repr file
        if memory_file_path and os.path.isfile(memory_file_path):
            # Memory was loaded from a file, look for repr file in same directory
            memory_dir = os.path.dirname(memory_file_path)
            filename = os.path.basename(memory_file_path)

            # Convert memory_data_N.pkl -> memory_repr_labels_N.pt
            if "memory_data_" in filename:
                task_id_str = filename.split("_")[-1].replace(".pkl", "")
                repr_filename = f"memory_repr_labels_{task_id_str}.pt"
                repr_file_path = os.path.join(memory_dir, repr_filename)

        elif os.path.isdir(previous_memory_path):
            # Memory path is a directory, find the latest repr file
            memory_repr_files = sorted([f for f in os.listdir(previous_memory_path)
                                       if f.startswith("memory_repr_labels_") and f.endswith(".pt")])
            if len(memory_repr_files) > 0:
                latest_repr_file = memory_repr_files[-1]
                repr_file_path = os.path.join(previous_memory_path, latest_repr_file)

        # Load memory representations if file exists
        if repr_file_path and os.path.exists(repr_file_path):
            logger.info(f"Loading memory representations from: {repr_file_path}")

            memory_repr_data = torch.load(repr_file_path, map_location=args.device)
            old_memory_repr = memory_repr_data['memory_repr']
            old_memory_labels = memory_repr_data['memory_labels']

            # Flatten if needed
            if isinstance(old_memory_repr, list):
                old_memory_repr = torch.cat(old_memory_repr, dim=0)
            if isinstance(old_memory_labels, list):
                old_memory_labels = torch.cat(old_memory_labels, dim=0)

            cur_memory_repr.append(old_memory_repr)
            cur_memory_labels.append(old_memory_labels)
        else:
            logger.warning(f"Memory representation file not found. Will compute from scratch if needed.")

    start_time = time.time()

    logger.info(f"***** Incremental Task-{new_task_id + 1} ({new_task_key}) *****")
    logger.info(f"New classes: {' '.join(cur_target_names)}")
    logger.info(f"All seen classes: {' '.join(seen_target_names)}")

    # Stage 1: Fast training on new data
    logger.info("Starting Stage 1: Fast training on new samples...")
    stage1_train_data = data.filter(new_label_ids, 'train')
    stage1_train_dataset = BaseDataset(stage1_train_data)

    stage1_train(
        model,
        args,
        num_train_epochs=args.stage1_epochs,
        data_collator=stage1_data_collator,
        train_dataset=stage1_train_dataset,
    )
    logger.info("Completed Stage 1: Fast training")

    # Select exemplars for new labels
    logger.info("Selecting exemplars from new samples...")
    for label in new_label_ids:
        tmp_train_data = data.filter(label, 'train')
        memory_data[label] = select_exemplars(model, args, evaluate_data_collator, tmp_train_data)
        tmp_memory_repr, tmp_memory_labels = get_moco_hidden_states(
            model, args, evaluate_data_collator,
            BaseDataset(memory_data[label]),
            return_type='torch',
            return_labels=True
        )
        cur_memory_repr.append(tmp_memory_repr)
        cur_memory_labels.append(tmp_memory_labels)

    logger.info("<<<<<<<<<<<<<<<<< Updated Memory >>>>>>>>>>>>>>>>>")

    # Save updated memory data

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"memory_data_{new_task_id}.pkl")
    logger.info(f"Saving updated memory to: {save_path}")
    with open(save_path, "wb") as f:
        pickle.dump(memory_data, f)

    repr_save_path = os.path.join(save_dir, f"memory_repr_labels_{new_task_id}.pt")
    logger.info(f"Saving memory representations to: {repr_save_path}")
    torch.save({
        "memory_repr": cur_memory_repr,
        "memory_labels": cur_memory_labels
    }, repr_save_path)

    # Stage 2: Slow training on memory (old + new exemplars)
    logger.info("Starting Stage 2: Slow training on memory data...")
    stage2_train_dataset = BaseDataset(memory_data)

    if args.stage2_type == 'new_old':
        memory_repr = torch.cat(cur_memory_repr, dim=0)
        memory_labels = torch.cat(cur_memory_labels, dim=0)
        stage2_train(
            model,
            args,
            num_train_epochs=args.stage2_epochs,
            data_collator=stage2_data_collator,
            train_dataset=stage2_train_dataset,
            memory_repr=memory_repr,
            memory_labels=memory_labels,
        )
    else:
        stage2_train(
            model,
            args,
            num_train_epochs=args.stage2_epochs,
            data_collator=stage2_data_collator,
            train_dataset=stage2_train_dataset,
        )
    logger.info("Completed Stage 2: Slow training")

    # Evaluation
    logger.info("Starting evaluation...")
    cur_test_data = data.filter(new_label_ids, 'test')
    cur_test_dataset = BaseDataset(cur_test_data)
    logger.info("<<<<<<<<<<<<<<<<< Current Task Performance >>>>>>>>>>>>>>>>>")
    cur_metric, _ = evaluate(
        model,
        args,
        data_collator=evaluate_data_collator,
        eval_dataset=cur_test_dataset,
        cur_train_data=data,
        memory_data=memory_data,
        cur_labels=new_label_ids,
        seen_labels=new_label_ids,
        verbose=True,
    )
    # Save final model
    final_model_path = os.path.join(save_dir, f"model_after_task_{new_task_id}")
    logger.info(f"Saving updated model to: {final_model_path}")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    '''
    1) update the saving model logic 
    2) add one more evaluation after training,if required , when we have completed test data, old labels test data + new labels test data
       need to add the flag to run the below code
    '''
    
    '''history_test_data = data.filter(seen_label_ids, 'test')
    history_test_dataset = BaseDataset(history_test_data)

    logger.info("<<<<<<<<<<<<<<<<< Overall Performance on All Tasks >>>>>>>>>>>>>>>>>")
    total_metric, detail_metric = evaluate(
        model,
        args,
        data_collator=evaluate_data_collator,
        eval_dataset=history_test_dataset,
        cur_train_data=data,
        memory_data=memory_data,
        cur_labels=new_label_ids,
        seen_labels=seen_label_ids,
        verbose=True,
    )
    logger.info(f"Overall performance: {total_metric * 100:.3f}")
    
    '''

    end_time = time.time()
    elapsed = end_time - start_time

    logger.info(f"***** Incremental Training Completed *****")
    logger.info(f"New task performance: {cur_metric * 100:.3f}")
    logger.info(f"Time elapsed: {elapsed:.2f} seconds")


    logger.info("=" * 80)
    logger.info("***** All Done! *****")
    logger.info("=" * 80)
    logger.info(f"For next incremental training, update configs/constants.py:")
    logger.info(f"  my_model = '{final_model_path}'")
    logger.info(f"  memory_path = '{save_path}'")
    logger.info("=" * 80)

    return {
        "new_task_performance": cur_metric * 100,
        # "overall_performance": total_metric * 100,
        "model_path": final_model_path,
        "memory_path": save_path,
        "new_task_id": new_task_id,
    }
