use_custom_train_test_split = True

# ===== Incremental Training Configuration =====
# These paths are used when continuing training with new samples

# Path to pre-trained model directory (contains config.json and model.safetensors)
my_model = "/home/ntlpt19/TF_testing_EXT/dummy_data/lic_data/model_save/slow_model"

# Path to previous memory data file (e.g., memory_data_5.pkl from last task)
memory_path = '/home/ntlpt19/TF_testing_EXT/dummy_data/lic_data/model_save/memory_data_5.pkl'

save_dir = "../model_save"
# Key for new task in manual_task_plan (e.g., "itr7" for the 7th iteration)
# This will be loaded automatically, or can be overridden from command line
new_task_key = "itr7"

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
    }