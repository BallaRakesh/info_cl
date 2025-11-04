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
from configs.constants import use_custom_train_test_split, manual_task_plan
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


def default_hyper_train(
    args: SimpleNamespace = None,
    data_collator: Optional[DataCollator] = None,
):
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

    all_cur_metric_rec = []
    all_total_metric_rec = []
    forgetting_rate_rec = []


    for cur_round in range(args.num_exp_rounds):
        cur_seed = args.seed + cur_round * 100
        set_seed(cur_seed)

        data = task_to_data_reader[args.dataset_name](args)

        args.id2label = data.id2label
        args.label2id = data.label2id
        num_labels = len(args.id2label)
        label_list = args.id2label
        if use_custom_train_test_split:
            task_seq = []
            label2id_norm = {k.strip().lower(): v for k, v in args.label2id.items()}
            for itr_name, labels in manual_task_plan.items():
                ids = []
                for label in labels:
                    nor_label = label.strip().lower()
                    if nor_label in label2id_norm:
                        ids.append(label2id_norm[nor_label])
                    else:
                        ids.append(-1)
                        print(f"[Warning] Label not found in mapping: {label}")
                task_seq.append(ids)
            max_len = max(len(seq) for seq in task_seq)
            task_seq = [seq + [-1] * (max_len - len(seq)) for seq in task_seq]
            task_seq = np.array(task_seq, dtype=int).tolist()
        else:
            task_seq = list(range(len(label_list)))
            if len(task_seq) != args.num_tasks * args.class_per_task:
                task_seq.extend([-1] * (args.num_tasks * args.class_per_task - len(task_seq)))
                random.shuffle(task_seq)
                task_seq = np.array(task_seq)
            else:
                random.shuffle(task_seq)
                task_seq = np.argsort(task_seq)

            if isinstance(args.class_per_task, int):
                task_seq = task_seq.reshape((args.num_tasks, args.class_per_task)).tolist()
            elif isinstance(args.class_per_task, list):
                tmp_seq = []
                cur = 0
                for n in args.class_per_task:
                    tmp_seq.append(task_seq[cur:cur + n].tolist())
                    cur += n
                task_seq = tmp_seq

        ModelForContinualLearning = task_to_model[args.task_name]

        config = AutoConfig.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            classifier_dropout=args.classifier_dropout,
            label2id=args.label2id,
            id2label=args.id2label,
        )
        config.global_args = args
        model = ModelForContinualLearning.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        # model.resize_token_embeddings(config.vocab_size + len(additional_special_tokens)) #????????
        if use_custom_train_test_split:
            data.load_train_test_data()
        else:
            data.read_and_preprocess(tokenizer, seed=cur_seed)

        model.to(args.device)

        cur_metric_rec = []
        total_metric_rec = []
        detail_metric_rec = []

        memory_data = {}

        old_memory_repr, old_memory_labels = None, None

        for task_id in range(args.num_tasks):
            start_time = time.time()  # ⏱️ start timer
            cur_labels = [label_id for label_id in task_seq[task_id] if label_id != -1]
            cur_target_names = [label_list[tmp_label] for tmp_label in cur_labels]
            seen_labels = [label_id for label_id in np.array(task_seq[:task_id + 1]).flatten().tolist() if label_id != -1]
            seen_target_names = [label_list[tmp_label] for tmp_label in seen_labels]
            
            cur_memory_repr, cur_memory_labels = [], []
            if old_memory_repr is not None:
                cur_memory_repr.append(old_memory_repr)
                cur_memory_labels.append(old_memory_labels)

            logger.info(f"***** Round-{cur_round + 1} Task-{task_id + 1} *****")
            
            logger.info(f"Current classes: {' '.join(cur_target_names)}")

            stage1_train_data = data.filter(cur_labels, 'train')

            stage1_train_dataset = BaseDataset(stage1_train_data)

            stage1_train(
                model,
                args,
                num_train_epochs=args.stage1_epochs,
                data_collator=stage1_data_collator,
                train_dataset=stage1_train_dataset,
            )
            logger.info("completed Fast training")

            for label in cur_labels:
                tmp_train_data = data.filter(label, 'train')
                memory_data[label] = select_exemplars(model, args, evaluate_data_collator, tmp_train_data)
                tmp_memory_repr, tmp_memory_labels = get_moco_hidden_states(model, args, evaluate_data_collator, BaseDataset(memory_data[label]), return_type='torch', return_labels=True)
                cur_memory_repr.append(tmp_memory_repr)
                cur_memory_labels.append(tmp_memory_labels)
            logger.info("<<<<<<<<<<<<<<<<< Updated Memory >>>>>>>>>>>>>>>>>")

            memory_save_dir = "/home/ng6281/Rupankar_Dev/gopal/rakesh/code/Continual_learning_poc/InfoCL/model_save"
            save_path = f"/home/ng6281/Rupankar_Dev/gopal/rakesh/code/Continual_learning_poc/InfoCL/model_save/memory_data_{task_id}.pkl"
            
            os.makedirs(memory_save_dir, exist_ok=True)
            
            with open(save_path, "wb") as f:
                pickle.dump(memory_data, f)
                
            torch.save({
                "memory_repr": cur_memory_repr,
                "memory_labels": cur_memory_labels
            }, os.path.join(memory_save_dir, f"memory_repr_labels_{task_id}.pt"))

            if task_id != 0:
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
            logger.info("Completed Slow Training")
            cur_test_data = data.filter(cur_labels, 'test')
            history_test_data = data.filter(seen_labels, 'test')

            cur_test_dataset = BaseDataset(cur_test_data)
            history_test_dataset = BaseDataset(history_test_data)
            
            logger.info("<<<<<<<<<<<<<<<<< cur_metric ON  cur_test_data >>>>>>>>>>>>>>>>>")
            cur_metric, _ = evaluate(
                model,
                args,
                data_collator=evaluate_data_collator,
                eval_dataset=cur_test_dataset,
                cur_train_data=data,
                memory_data=memory_data,
                cur_labels=cur_labels,
                seen_labels=cur_labels,
                verbose=True,
            )
            logger.info("<<<<<<<<<<<<<<<<< history_metric ON  history_test_dataset >>>>>>>>>>>>>>>>>")

            total_metric, detail_metric = evaluate(
                model,
                args,
                data_collator=evaluate_data_collator,
                eval_dataset=history_test_dataset,
                cur_train_data=data,
                memory_data=memory_data,
                cur_labels=cur_labels,
                seen_labels=seen_labels,
                verbose=True,
            )

            if args.stage2_type == 'new_old':
                old_memory_repr, old_memory_labels = get_moco_hidden_states(model, args, evaluate_data_collator, BaseDataset(memory_data), return_type='torch', return_labels=True)

            cur_metric_rec.append(cur_metric * 100)
            total_metric_rec.append(total_metric * 100)
            detail_metric_rec.append(detail_metric)

            logger.info(f"***** Round-{cur_round + 1} Task-{task_id + 1} *****")
            logger.info(f"History test metrics: {' '.join([str(round(metric, 3)) for metric in total_metric_rec])}")
            logger.info(f"Current test metrics: {' '.join([str(round(metric, 3)) for metric in cur_metric_rec])}")
            logger.info(f"*****$$$$$$$$$$$$$$$$$$$ Competed  Task-{task_id + 1} $$$$$$$$$$$$$$$$$$$*****")
            logger.info("#############################################################################")
            logger.info("#############################################################################")

            # --- end timing ---
            end_time = time.time()
            elapsed = end_time - start_time
            logger.info(f"⏳ Task-{task_id + 1} completed in {elapsed:.2f} seconds")
        all_cur_metric_rec.append(cur_metric_rec)
        all_total_metric_rec.append(total_metric_rec)
        forgetting_rate_rec.append(compute_forgetting_rate(detail_metric_rec, task_seq, args.id2label, mode='task'))

    all_cur_metric_rec = np.array(all_cur_metric_rec).mean(axis=0).tolist()
    all_total_metric_rec = np.array(all_total_metric_rec).mean(axis=0).tolist()
    forgetting_rate_rec = np.mean(forgetting_rate_rec)

    logger.info(f"***** Experiment over *****")
    logger.info(f"Average history test metrics: {' '.join([str(round(metric, 3)) for metric in all_total_metric_rec])}")
    logger.info(f"Average current test metrics: {' '.join([str(round(metric, 3)) for metric in all_cur_metric_rec])}")
    logger.info(f"Average forgetting rate: {forgetting_rate_rec}")
