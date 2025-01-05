from data import ADNIMRIDataset
from utils import cross_validate
from literature_models import J_CNN3DModel, JAL_model
import pandas as pd
import ants

dataset = pd.read_csv("jacobian_first_folds_dis.csv")

for data_type in ["syn_registered"]:
    for input_type in ["attention","concat"]:
        cross_validate(
                model_class=J_CNN3DModel,
                dataset=dataset,
                num_classes=2,
                device="cuda",
                num_epochs=20,
                learning_rate=0.00015,
                dtype=data_type,
                model_name="J_CNN3D",
                cls_type="cn_vs_ad",
                batch_size = 16,
                output_folder="/projects/florence_echo/Shijia/ADNI/Jacobian_method/Result_test_new",
                input_type=input_type,
                thereshold=0.2,
                apply_rotation = True,
                max_rotation=1,
                fusion_type = None,
            )



