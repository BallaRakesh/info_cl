import os
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import ast
from .BaseData import BaseData
import torch
import re
from collections import defaultdict
class HWU64Data(BaseData):
    def __init__(self, args):
        super().__init__(args)

    def preprocess(self, raw_data, label, tokenizer, org_label):
        print(label,'>>>>>>>>>>>>>>>>>>>>>>' ,len(raw_data))
        res = []
        # result = tokenizer(raw_data)
        result = tokenizer(
            raw_data,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"   # optional, if you want torch tensors
        )

        for idx in range(len(raw_data)):
            res.append({
                'idx': str(idx)+'_'+str(org_label),
                'org_query':raw_data[idx],
                'org_label': org_label,
                'input_ids': result['input_ids'][idx],
                'attention_mask': result['attention_mask'][idx],
                'labels': label,
            })
        return res

    def read_and_preprocess(self, tokenizer, seed=None):
        json_file1 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_train_data_2.json'
        json_file2 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_test_data_2.json'
        json_file3 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_validate_data_2.json'

        final_data = []
        # --- Step 1: Load the JSON file ---
        for json_file in [json_file1, json_file2, json_file3]:
            with open(json_file, "r") as f:
                data = json.load(f)
            final_data.extend(data)
        raw_data = pd.DataFrame([
            {
                "Query": item["Query"],
                "Reasoning": item["Reasoning"],
                "Category": item["Response"]["Complaint Category"],
                "Sub Category": item["Response"]["Complaint Sub Category"]
            }
            for item in final_data
        ])
        # raw_data = pd.read_csv(open(os.path.join(self.args.data_path, self.args.dataset_name, 'NLU-Data-Home-Domain-Annotated-All.csv')), sep=";")
        # raw_data = pd.read_csv('/home/ng6281/Rupankar_Dev/gopal/rakesh/code/Continual_learning_poc/InfoCL/9_sept_no_reasoning_res.csv')
        # if seed is not None:
        #     random.seed(seed)

        # raw_data['label'] = raw_data['scenario'] + '_' + raw_data['intent']
        raw_data['label'] = raw_data["Sub Category"]
        # delete_columns = ['answer_annotation', 'scenario', 'intent', 'userid', 'answerid', 'suggested_entities', 'answer', 'question', 'status', 'notes']
        delete_columns = [
                        # "Sub Category",
                        "Category",
                        "Reasoning"
                        "result",
                        "Predicted_Category",
                        "Predicted_Sub_Category",
                        "Predicted_Sentiment",
                        "Predicted_Frustration",
                        "Predicted_Disappointment",
                        "Predicted_Confusion",
                        "Predicted_Excitement",
                        "Predicted_Sadness",
                        "Predicted_Gratitude"
                    ]

        for column in delete_columns:
            if column in raw_data:
                del raw_data[column]

        raw_data = raw_data.dropna(axis=0, how='any')

        self.train_data = {label: [] for label in range(len(self.id2label))}
        self.val_data = {label: [] for label in range(len(self.id2label))}
        self.test_data = {label: [] for label in range(len(self.id2label))}
        all_train_rows = []
        all_test_rows = []


        for label in self.id2label:
            print('Processing label:', label)
            # cur_data = raw_data[raw_data['label'] == label]['answer_normalised'].tolist()
            # cur_data = raw_data[raw_data['label'] == label]['Query'].tolist()
            cur_data = raw_data[raw_data['label'].str.strip().str.lower() == label.strip().lower()]['Query'].tolist()
            
            # print(cur_data)
            print(len(cur_data))
            cur_data = self.preprocess(cur_data, self.label2id[label], tokenizer, label)
            random.shuffle(cur_data)
            total_cnt = len(cur_data)# if len(cur_data) < 195 else 194
            test_cnt = total_cnt // 5
            
            train_part = cur_data[test_cnt:total_cnt]
            test_part = cur_data[:test_cnt]
            
            all_train_rows.extend(train_part)
            all_test_rows.extend(test_part)
            
            clean_train_part = [
                {k: v for k, v in item.items() if k not in ['idx', 'org_query', 'org_label']}
                for item in train_part
            ]
            clean_test_part = [
                {k: v for k, v in item.items() if k not in ['idx', 'org_query','org_label']}
                for item in test_part
            ]
            
            self.train_data[self.label2id[label]] = clean_train_part
            self.test_data[self.label2id[label]] = clean_test_part
            


        # ✅ Convert all label data to DataFrames
        train_df = pd.DataFrame(all_train_rows)
        test_df = pd.DataFrame(all_test_rows)
        save_dir = "../model_save"
        os.makedirs(save_dir, exist_ok=True)
        # ✅ Save as single CSV files
        train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
        test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)



    def load_train_test_data(self, incremental_label2id=None):
        """

        Load train.csv and test.csv and reconstruct self.train_data and self.test_data
        
        Args:
            train_csv_path: Path to train.csv file
            test_csv_path: Path to test.csv file
        
        Returns:
            tuple: (self.train_data, self.test_data)
        """
        train_csv_path = 'round_4_train_test/train.csv'
        test_csv_path = 'round_4_train_test/test.csv'
        # Load the CSV files
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        
        # Initialize the dictionaries
        self.train_data = {}
        self.test_data = {}
        
        # Helper function to convert string representations back to lists
        def parse_field(field):
            if isinstance(field, str):
                # Check if it's a tensor string representation
                if field.strip().startswith('tensor'):
                    # Extract numbers from tensor string using regex
                    numbers = re.findall(r'-?\d+', field)
                    tensor_list = [int(n) for n in numbers]
                    # Convert back to tensor
                    return torch.tensor(tensor_list)
                else:
                    # Regular list string - convert to tensor
                    try:
                        import ast
                        tensor_list = ast.literal_eval(field)
                        return torch.tensor(tensor_list)
                    except:
                        return field
            return field
        
                # Iterate over id2label (same as original logic)
        for label in self.id2label:
            print('Loading label:', label)
            
            # Filter train data using case-insensitive matching (same as original)
            train_label_rows = train_df[
                train_df['org_label'].str.strip().str.lower() == label.strip().lower()
            ]
            
            # Filter test data using case-insensitive matching
            test_label_rows = test_df[
                test_df['org_label'].str.strip().str.lower() == label.strip().lower()
            ]
            
            print(f"Train count: {len(train_label_rows)}, Test count: {len(test_label_rows)}")
            
            # Determine the label ID to use
            label_id = None
            if incremental_label2id is not None:
                # Search for matching label using case-insensitive comparison
                for inc_label, inc_id in incremental_label2id.items():
                    if inc_label.strip().lower() == label.strip().lower():
                        label_id = inc_id
                        break
            
            # Clean and prepare train data (exclude idx, org_query, org_label)
            clean_train_part = []
            for _, row in train_label_rows.iterrows():
                clean_train_part.append({
                    'input_ids': parse_field(row['input_ids']),
                    'attention_mask': parse_field(row['attention_mask']),
                    'labels': label_id if label_id is not None else int(row['labels'])
                })
            
            # Clean and prepare test data (exclude idx, org_query, org_label)
            clean_test_part = []
            for _, row in test_label_rows.iterrows():
                clean_test_part.append({
                    'input_ids': parse_field(row['input_ids']),
                    'attention_mask': parse_field(row['attention_mask']),
                    'labels': label_id if label_id is not None else int(row['labels'])
                })
            
            # Store in dictionaries with label2id mapping (use label_id if found, otherwise self.label2id[label])
            final_label_id = label_id if label_id is not None else self.label2id[label]
            self.train_data[final_label_id] = clean_train_part
            self.test_data[final_label_id] = clean_test_part

        print(f"\n✅ Loaded train data with {len(self.train_data)} labels")
        print(f"✅ Loaded test data with {len(self.test_data)} labels")
        print(self.train_data.keys())
        if incremental_label2id is not None:
            incremental_list = defaultdict(list)
            for k, v in incremental_label2id.items():
                incremental_list[v].append(k)
            for label_id in sorted(self.train_data.keys()):
                label_name = incremental_list[label_id]
                print(f"Label {label_id} ({label_name}): Train={len(self.train_data[label_id])}, Test={len(self.test_data[label_id])}")
        else:
            # Print data counts per label
            for label_id in sorted(self.train_data.keys()):
                label_name = self.id2label[label_id]
                print(f"Label {label_id} ({label_name}): Train={len(self.train_data[label_id])}, Test={len(self.test_data[label_id])}")
                
                
    def load_train_test_data_testing(self, incremental_label2id=None):
        """

        Load train.csv and test.csv and reconstruct self.train_data and self.test_data
        
        Args:
            train_csv_path: Path to train.csv file
            test_csv_path: Path to test.csv file
        
        Returns:
            tuple: (self.train_data, self.test_data)
        """
        train_csv_path = 'round_4_train_test/train.csv'
        test_csv_path = 'round_4_train_test/test.csv'
        # Load the CSV files
        train_df = pd.read_csv(train_csv_path)
        test_df = pd.read_csv(test_csv_path)
        
        # Initialize the dictionaries
        self.train_data = {}
        self.test_data = {}
        
        # Helper function to convert string representations back to lists
        def parse_field(field):
            if isinstance(field, str):
                # Check if it's a tensor string representation
                if field.strip().startswith('tensor'):
                    # Extract numbers from tensor string using regex
                    numbers = re.findall(r'-?\d+', field)
                    tensor_list = [int(n) for n in numbers]
                    # Convert back to tensor
                    return torch.tensor(tensor_list)
                else:
                    # Regular list string - convert to tensor
                    try:
                        import ast
                        tensor_list = ast.literal_eval(field)
                        return torch.tensor(tensor_list)
                    except:
                        return field
            return field
        
                # Iterate over id2label (same as original logic)
        for label, label_id in incremental_label2id.items():#self.id2label:
            print('Loading label:', label)
            
            # Filter train data using case-insensitive matching (same as original)
            train_label_rows = train_df[
                train_df['org_label'].str.strip().str.lower() == label.strip().lower()
            ]
            
            # Filter test data using case-insensitive matching
            test_label_rows = test_df[
                test_df['org_label'].str.strip().str.lower() == label.strip().lower()
            ]
            
            print(f"Train count: {len(train_label_rows)}, Test count: {len(test_label_rows)}")
            
            # Determine the label ID to use
            # label_id = None
            # if incremental_label2id is not None:
            #     # Search for matching label using case-insensitive comparison
            #     for inc_label, inc_id in incremental_label2id.items():
            #         if inc_label.strip().lower() == label.strip().lower():
            #             label_id = inc_id
            #             break
            
            # Clean and prepare train data (exclude idx, org_query, org_label)
            clean_train_part = []
            for _, row in train_label_rows.iterrows():
                clean_train_part.append({
                    'input_ids': parse_field(row['input_ids']),
                    'attention_mask': parse_field(row['attention_mask']),
                    'labels': label_id if label_id is not None else int(row['labels'])
                })
            
            # Clean and prepare test data (exclude idx, org_query, org_label)
            clean_test_part = []
            for _, row in test_label_rows.iterrows():
                clean_test_part.append({
                    'input_ids': parse_field(row['input_ids']),
                    'attention_mask': parse_field(row['attention_mask']),
                    'labels': label_id if label_id is not None else int(row['labels'])
                })
            
            # Store in dictionaries with label2id mapping (use label_id if found, otherwise self.label2id[label])
            final_label_id = label_id if label_id is not None else self.label2id[label]
            self.train_data[final_label_id] = clean_train_part
            self.test_data[final_label_id] = clean_test_part

        print(f"\n✅ Loaded train data with {len(self.train_data)} labels")
        print(f"✅ Loaded test data with {len(self.test_data)} labels")
        print(self.train_data.keys())
        if incremental_label2id is not None:
            incremental_list = defaultdict(list)
            for k, v in incremental_label2id.items():
                incremental_list[v].append(k)
            for label_id in sorted(self.train_data.keys()):
                label_name = incremental_list[label_id]
                print(f"Label {label_id} ({label_name}): Train={len(self.train_data[label_id])}, Test={len(self.test_data[label_id])}")
        else:
            # Print data counts per label
            for label_id in sorted(self.train_data.keys()):
                label_name = self.id2label[label_id]
                print(f"Label {label_id} ({label_name}): Train={len(self.train_data[label_id])}, Test={len(self.test_data[label_id])}")
                
