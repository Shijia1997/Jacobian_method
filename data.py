import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import ants
from attention_map import combine_jsm_and_gdp,apply_attention_map,compute_gradient_attention,preprocess_log_jsm,preprocess_log_jsm_negative
import numpy as np

def normalize(image):
    """
    Normalize the image using ANTsPy methods (zero mean, unit variance).
    """
    image = ants.iMath_normalize(image)
    return image


def process_scan(image):
    """
    Process the input image:
    - Normalize intensity
    - Convert to a numpy array
    """
    image = normalize(image)
    image_np = image.numpy()
    return image_np

def random_3D_rotation(image, jsm, max_rotation=10):
    """
    Apply the same random Euler3DTransform rotation to 'image' and 'jsm'.
    If 'jsm' is None or intentionally the same as 'image', we rotate only the image.
    """
    angles_deg = np.random.uniform(-max_rotation, max_rotation, size=3)
    angles_rad = angles_deg * np.pi / 180.0
    
    # Use geometric center instead of center of mass
    center = [dim_size / 2 for dim_size in image.shape]
    
    tx = ants.create_ants_transform(
        transform_type='Euler3DTransform',
        dimension=3,
        center=center
    )
    tx.set_parameters([angles_rad[0], angles_rad[1], angles_rad[2], 0, 0, 0])
    
    # Rotate the MRI
    rotated_image = tx.apply(image, data_type='image', reference=image)
    
    # If jsm is provided, rotate it too; otherwise jsm_out can be None
    if jsm is not None:
        rotated_jsm = tx.apply(jsm, data_type='image', reference=image)
    else:
        rotated_jsm = None

    return rotated_image, rotated_jsm


class ADNIMRIDataset(Dataset):
    def __init__(
        self,
        image_paths,
        image_labels,
        jsm_paths=None,
        input_type="single",
        apply_random_rotation=False,
        max_rotation=10,
        thereshold=0.2
    ):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.jsm_paths = jsm_paths
        self.input_type = input_type
        self.apply_random_rotation = apply_random_rotation
        self.max_rotation = max_rotation
        self.thereshold = thereshold
        
        valid_input_types = ["single", "concat", "attention", "jsm", "cross_attention"]
        if self.input_type not in valid_input_types:
            raise ValueError(f"Invalid input_type '{self.input_type}'. Must be one of {valid_input_types}.")

        if self.input_type == "jsm" and not self.jsm_paths:
            raise ValueError("JSM paths must be provided for 'jsm' input type.")

        if self.input_type in ["concat", "attention", "jsm", "cross_attention"] and self.jsm_paths is None:
            raise ValueError(f"JSM paths must be provided for '{self.input_type}' input type.")

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        label = self.image_labels[idx]
        
        # ---------------------------
        # Case 1: JSM-only scenario
        # ---------------------------
        if self.input_type == "jsm":
            jsm_path = self.jsm_paths[idx]
            jsm = ants.image_read(jsm_path)
            
            if self.apply_random_rotation:
                # Rotate JSM alone
                jsm, _ = random_3D_rotation(jsm, None, self.max_rotation)
            
            jsm_np = jsm.numpy()
            # We preserve sign and threshold
            #jsm_np = preprocess_log_jsm_negative(jsm_np, self.thereshold)
            
            jsm_tensor = torch.from_numpy(jsm_np).float().unsqueeze(0)  
            return jsm_tensor, label

        # ---------------------------
        # Otherwise, load the MRI
        # ---------------------------
        image_path = self.image_paths[idx]
        image = ants.image_read(image_path)
        
        # If we have a jsm_path, read that as well
        if self.jsm_paths:
            jsm_path = self.jsm_paths[idx]
            jsm = ants.image_read(jsm_path)
        else:
            jsm = None

        # ---------------------------
        # Potentially random rotate
        # ---------------------------
        if self.apply_random_rotation:
            # If jsm exists, rotate both; otherwise rotate only the MRI
            image, jsm = random_3D_rotation(image, jsm, max_rotation=self.max_rotation)

        # 1) Convert MRI to numpy
        img_arr = process_scan(image)  # e.g. intensity normalization
        img_tensor = torch.from_numpy(img_arr).float().unsqueeze(0)  # shape [1, D, H, W]

        # 2) Return depends on input_type
        if self.input_type == "single":
            return img_tensor, label

        if self.input_type == "concat":
            jsm_arr = jsm.numpy()
            #jsm_arr = preprocess_log_jsm_negative(jsm_arr, self.thereshold)
            jsm_tensor = torch.from_numpy(jsm_arr).float().unsqueeze(0)
            
            concatenated_img = torch.cat((img_tensor, jsm_tensor), dim=0)
            return concatenated_img, label

        if self.input_type == "attention":
            jsm_arr = jsm.numpy()
            #jsm_arr = preprocess_log_jsm_negative(jsm_arr, self.thereshold)
            jsm_tensor = torch.from_numpy(jsm_arr).float().unsqueeze(0)
            return img_tensor, jsm_tensor, label

        if self.input_type == "cross_attention":
            jsm_arr = jsm.numpy()
            #jsm_arr = preprocess_log_jsm_negative(jsm_arr, self.thereshold)
            jsm_tensor = torch.from_numpy(jsm_arr).float().unsqueeze(0)
            return img_tensor, jsm_tensor, label