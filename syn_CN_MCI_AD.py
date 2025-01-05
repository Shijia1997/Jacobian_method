from data import ADNIMRIDataset
from utils import cross_validate
from literature_models import J_CNN3DModel, JAL_model
import pandas as pd
import ants

# Read your dataset CSV
dataset = pd.read_csv("jacobian_first_folds_dis.csv")

# Define classification scenarios we want to run
classification_tasks = ["cn_vs_mci_vs_ad", "cn_vs_ad", "cn_vs_mci"]
# For each, we specify how many output classes are needed
def get_num_classes(cls_type):
    if cls_type == "cn_vs_mci_vs_ad":
        return 3
    else:
        return 2  # cn_vs_ad or cn_vs_mci

# We only do 'syn_registered' here, but you could add more if desired
for data_type in ["syn_registered"]:
    for cls_type in classification_tasks:
        num_cls = get_num_classes(cls_type)

        # We'll run cross_validate for each input_type in a loop
        for input_type in ["attention","cross_attention","concat", "single"]:
            cross_validate(
                model_class=J_CNN3DModel,
                dataset=dataset,
                num_classes=num_cls,  # set based on classification task
                device="cuda",
                num_epochs=20,
                learning_rate=0.00015,
                dtype=data_type,       # e.g., "syn_registered"
                model_name="J_CNN3D",
                cls_type=cls_type,     # e.g., "cn_vs_mci_vs_ad", "cn_vs_ad", "cn_vs_mci"
                batch_size=16,
                output_folder=(
                    "/projects/florence_echo/Shijia/ADNI/"
                    "Jacobian_method/Result_test_3_classes"
                ),
                input_type=input_type,  
                thereshold=0.05,     
                apply_rotation=True,
                max_rotation=1,
                fusion_type=None
            )

        # Additionally, run the 'jsm' mode
        cross_validate(
            model_class=J_CNN3DModel,
            dataset=dataset,
            num_classes=num_cls,
            device="cuda",
            num_epochs=20,
            learning_rate=0.00015,
            dtype=data_type,
            model_name="J_CNN3D",
            cls_type=cls_type,
            batch_size=16,
            output_folder=(
                "/projects/florence_echo/Shijia/ADNI/Jacobian_method/Result_test_3_classes"
            ),
            input_type="jsm",
            thereshold=0.05,
            apply_rotation=True,
            max_rotation=1,
            fusion_type=None
        )
