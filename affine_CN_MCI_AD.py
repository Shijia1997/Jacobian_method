from data import ADNIMRIDataset
from utils import cross_validate
from literature_models import J_CNN3DModel, JAL_model
import pandas as pd
import ants

# Read your dataset CSV
dataset = pd.read_csv("jacobian_first_folds_dis.csv")

# We want to handle both 2-class tasks and the 3-class task.
classification_tasks = [
    ("cn_vs_mci_vs_ad", 3),  # 3 classes
    ("cn_vs_ad", 2),         # 2 classes
    ("cn_vs_mci", 2)         # 2 classes
]

for data_type in ["affine_registered"]:
    for cls_type, num_classes in classification_tasks:
        # We'll run cross_validate for each input_type in a loop
        for input_type in ["attention","cross_attention","concat", "single"]:
            cross_validate(
                model_class=J_CNN3DModel,
                dataset=dataset,
                num_classes=num_classes,  # 2 or 3 depending on the task
                device="cuda",
                num_epochs=20,
                learning_rate=0.00015,
                dtype=data_type,       
                model_name="J_CNN3D",
                cls_type=cls_type,     # e.g., "cn_vs_mci_vs_ad", "cn_vs_ad", "cn_vs_mci"
                batch_size=16,
                output_folder="/projects/florence_echo/Shijia/ADNI/Jacobian_method/Result_test_3_classes",
                input_type=input_type,  
                thereshold=0.05,
                apply_rotation=True,
                max_rotation=1,
                fusion_type=None
            )

        # Finally, run the 'jsm' mode
        cross_validate(
            model_class=J_CNN3DModel,
            dataset=dataset,
            num_classes=num_classes,
            device="cuda",
            num_epochs=20,
            learning_rate=0.00015,
            dtype=data_type,
            model_name="J_CNN3D",
            cls_type=cls_type,
            batch_size=16,
            output_folder="/projects/florence_echo/Shijia/ADNI/Jacobian_method/Result_test_3_classes",
            input_type="jsm",
            thereshold=0.05,
            apply_rotation=True,
            max_rotation=1,
            fusion_type=None
        )
