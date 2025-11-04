import numpy as np

# manual plan as dict of label names
manual_task_plan = { 
    "itr1": [
        "Disputes concerning eligibility of surrender value",
        "Response for processing or payment of Policy Loan is not sent",
        "Response for issuance of duplicate policy is not sent",
        "Request for Servicing Branch transfer is not effected"
    ],
    "itr2": [
        "Policy Benefit option not effected",
        "Requirements for revival not communicated or raised",
        "Statement of account not received",
        "Non-payment of penal interest"
    ],
    "itr3": [
        "Survival Benefit is not paid",
        "Non-receipt of Duplicate policy",
        "No Response for recording Change of address"
    ],
    "itr4": [
        "Disputes concerning correctness of surrender value",
        "Payment of premium not acted upon or wrongly acted upon",
        "No Response for noting a new nomination"
    ],
    "itr5": [
        "Non-receipt of Premium receipt",
        "Maturity claim is not paid",
        "Commutation value/cash option not paid"
    ],
    "itr6": [
        "Dispute concerning claim value",
        "Annuity/pension instalments not paid",
        "Reinstatement requirements raised by Insurer not acceptable",
        "Surrender Value not paid"
    ],
    "itr7": [
        "Alteration in policy not effected.",
        "After submission of all reinstatement (revival) requirements",
        "Dispute concerning statement of account or premium position statement"
    ]
    }  # paste your dictionary here



# maps from label name to integer ID
label2id = {'Non-receipt of Premium receipt': 0, 'Requirements for revival not communicated or raised': 1, 'Response for processing or payment of Policy Loan is not sent ': 2, 'Alteration in policy not effected.': 3, 'Request for Servicing Branch transfer is not effected': 4, 'No Response for noting a new nomination ': 5, 'Payment of premium not acted upon or wrongly acted upon': 6, 'Response for issuance of duplicate policy is not sent ': 7, 'After submission of all reinstatement (revival) requirements': 8, 'Statement of account not received': 9, 'Policy Benefit option not effected': 10, 'Dispute concerning statement of account or premium position statement': 11, 'Reinstatement requirements raised by Insurer not acceptable': 12, 'No Response for recording Change of address ': 13, 'Non-receipt of Duplicate policy': 14, 'Annuity/pension instalments not paid': 15, 'Survival Benefit is not paid ': 16, 'Disputes concerning eligibility of surrender value': 17, 'Surrender Value not paid': 18, 'Non-payment of penal interest ': 19, 'Disputes concerning correctness of surrender value': 20, 'Dispute concerning claim value': 21, 'Commutation value/cash option not paid': 22, 'Maturity claim is not paid ': 23}


task_seq = []
label2id_norm = {k.strip().lower(): v for k, v in label2id.items()}
    
for itr_name, labels in manual_task_plan.items():
    ids = []
    for label in labels:
        nor_label = label.strip().lower()
        if nor_label in label2id_norm:
            ids.append(label2id_norm[nor_label])
        else:
            # unknown label → use -1 placeholder
            ids.append(-1)
            print(f"[Warning] Label not found in mapping: {label}")
    task_seq.append(ids)
print(task_seq)
# Find the max length among all sublists
max_len = max(len(seq) for seq in task_seq)
# Pad each list with -1 up to max_len
task_seq_padded = [seq + [-1] * (max_len - len(seq)) for seq in task_seq]
# Convert to numpy array (optional, for consistency)
task_seq_padded = np.array(task_seq_padded, dtype=int).tolist()
print(task_seq_padded)



exit('????')



import json
import pandas as pd
json_file1 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_train_data_2.json'
# json_file1 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_train_data_added_syn_org_sep22.json'
json_file2 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_test_data_2.json'
json_file3 = '/home/ng6281/Rupankar_Dev/gopal/final_split_data/fine_tuning_split/6500_validate_data_2.json'

final_data = []
# --- Step 1: Load the JSON file ---
for json_file in [json_file1, json_file2, json_file3]:
    with open(json_file, "r") as f:
        data = json.load(f)
    final_data.extend(data)

# --- Step 2: Create a DataFrame ---
# if it's a single object, wrap it in a list
# if isinstance(data, dict):
#     data = [data]

df = pd.DataFrame([
    {
        "Query": item["Query"],
        "Reasoning": item["Reasoning"],
        "Complaint Category": item["Response"]["Complaint Category"],
        "Complaint Sub Category": item["Response"]["Complaint Sub Category"]
    }
    for item in final_data
])

# --- Step 3: Display the DataFrame ---
print(df.head())

# --- Step 4: Get unique Complaint Sub Category counts ---
subcat_counts = df["Complaint Sub Category"].value_counts()

print("\nUnique Complaint Sub Category counts:\n")
for label, count in subcat_counts.items():
    print(f"{label}: {count}")

exit('////////////////////')












import pickle

# Path to your pickle file
file_path = "/home/ng6281/Rupankar_Dev/gopal/rakesh/code/Continual_learning_poc/model_save_itr1(train_test_overridded)/memory_data_1.pkl"

# Open the pickle file in binary read mode
with open(file_path, "rb") as f:
    data = pickle.load(f)

# Inspect the loaded data

print(type(data))
print(len(data))
for i,j in data.items():
    print(type(i))
    print(type(j))
    print(i, '>>>>>>>>>>>>', len(j))
# print(data)
