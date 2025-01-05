from data import ADNIMRIDataset
from utils import cross_validate
from literature_models import J_CNN3DModel, JAL_model
import pandas as pd
import ants

dataset = pd.read_csv("jacobian_first_folds_dis.csv")


for cls_type in ["cn_vs_ad","cn_vs_mci"]:
    cross_validate(
        model_class=None,
        dataset=dataset,
        num_classes=2,
        device="cuda",
        num_epochs=20,
        learning_rate=0.00015,
        dtype="affine_registered",
        model_name="cross_attention",
        cls_type=cls_type,
        alpha_value = 0.97,
        batch_size = 16,
        fusion_type = "attend_only",
        output_folder="/projects/florence_echo/Shijia/ADNI/Jacobian_method/Result_test_cross_attendedonly",
        input_type="cross_attention")
    


