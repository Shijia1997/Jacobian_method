import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from literature_models import J_CNN3DModel,JAL_model
from resnet_arc import generate_model
from data import ADNIMRIDataset
from attention_map import compute_gradient_attention, combine_jsm_and_gdp, apply_attention_map
import numpy as np
from cross_attention import CrossAttention3DClassifier





def map_diagnosis_labels(data, classification_type):
    if classification_type == 'cn_vs_mci_vs_ad':
        label_mapping = {1: 0, 2: 1, 3: 2}
    elif classification_type == 'cn_vs_rest':
        label_mapping = {1: 0, 2: 1, 3: 1}
    elif classification_type == 'cn_vs_ad':
        # Filter out cases not in [1, 3]
        data = data[data["DIAGNOSIS"].isin([1, 3])]
        label_mapping = {1: 0, 3: 1}
    elif classification_type == 'cn_vs_mci':
        data = data[data["DIAGNOSIS"].isin([1, 2])]
        label_mapping = {1: 0, 2: 1}
    elif classification_type == 'mci_vs_ad':
        data = data[data["DIAGNOSIS"].isin([2, 3])]
        label_mapping = {2: 0, 3: 1}
    else:
        raise ValueError(f"Unknown classification type: {classification_type}")
    
    data = data.copy()
    data["label"] = data["DIAGNOSIS"].map(label_mapping)
    data = data.reset_index(drop=True)
    return data


def load_data(full_data, data_type, classification_type,input_type):
    """
    Load image paths, JSM paths, and labels based on the provided data type and classification type.

    Parameters:
    - full_data: Pandas DataFrame with at least columns: "DIAGNOSIS", "fold", "linear_registered_path" 
                 and possibly "syn_jacobian_regular", "non_linear_registered_path".
    - data_type: "affine_registered" or "syn_registered"
    - classification_type: E.g., "cn_vs_mci_vs_ad".

    Returns:
    - image_paths: Pandas Series of paths to the MRI images
    - JSM_path: Pandas Series of paths to the Jacobian maps (if present)
    - labels: Pandas Series of labels
    """
    full_data = map_diagnosis_labels(full_data, classification_type)

    if data_type == "affine_registered":
        if "linear_registered_path" not in full_data.columns:
            raise ValueError("Expected 'linear_registered_path' in the dataset for affine_registered data type.")
        image_paths = full_data["linear_registered_path"]
    elif data_type == "syn_registered":
        if "non_linear_registered_path" not in full_data.columns:
            raise ValueError("Expected 'non_linear_registered_path' in the dataset for syn_registered data type.")
        image_paths = full_data["non_linear_registered_path"]
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # JSM path is usually "syn_jacobian_regular" if present
    if input_type == "attention":
        # syn_jacobian is log jacobian change to regular for one kind if possible (now both log so if ifelese)
        JSM_path = full_data["syn_jacobian"] if "syn_jacobian" in full_data else None
    else:
        JSM_path = full_data["syn_jacobian"] if "syn_jacobian" in full_data else None
    labels = full_data["label"]

    return image_paths, JSM_path, labels






def train_model(model,train_dataloader,optimizer,criterion,num_epochs,device,input_type):
    """
    Train the model given different input types: 'attention', 'cross_attention', 'concat', 'single', 'jsm'.
    
    Args:
        model: PyTorch model to be trained.
        train_dataloader: Dataloader that yields training batches.
        optimizer: PyTorch optimizer.
        criterion: Loss function (e.g., CrossEntropyLoss).
        num_epochs: Number of epochs to train.
        device: 'cpu' or 'cuda'.
        input_type: Determines how data is handled in each batch 
                    ('attention', 'cross_attention', 'concat', 'single', or 'jsm').
        alpha_value: Parameter used for attention-based methods.
    """
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in train_dataloader:

            # ---- 1) Attention scenario ----
            if input_type == "attention":
                img_arr, jsm_arr, targets = batch
                img_arr = img_arr.to(device)
                jsm_arr = jsm_arr.to(device)
                targets = targets.to(device)

                # Example: compute gradient-based attention, combine with JSM, etc.
                #gdp = compute_gradient_attention(model, img_arr, device)
                #attention_map = combine_jsm_and_gdp(jsm_arr, gdp, alpha=alpha_value)
                img_arr = apply_attention_map(img_arr, jsm_arr)

                # We'll feed only the updated 'img_arr' to the model here
                outputs = model(img_arr)

            # ---- 2) Cross-Attention scenario ----
            elif input_type == "cross_attention":
                # Typically, the model expects both MRI (img_arr) and JSM (jsm_arr)
                img_arr, jsm_arr, targets = batch
                img_arr = img_arr.to(device)
                jsm_arr = jsm_arr.to(device)
                targets = targets.to(device)

                # Forward pass with cross-attention requires BOTH inputs
                outputs = model(img_arr, jsm_arr)

            # ---- 3) Concatenate, Single, or JSM scenario ----
            elif input_type in ["concat", "single", "jsm"]:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

            else:
                raise ValueError(f"Unknown input type {input_type}")

            # ---- 4) Compute Loss and Optimize ----
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    return model







def evaluate_model(
    model,
    val_dataloader,
    num_classes,
    device,
    input_type
):
    """
    Evaluate the model on a validation set, computing:
      - ROC-AUC
      - PR-AUC
      - Accuracy
      - Macro F1
      - Precision
      - Recall

    Handles multiple 'input_type' scenarios:
      'attention', 'cross_attention', 'concat', 'single', 'jsm'.
    """
    model.eval()

    all_targets = []
    all_probabilities = []
    correct_predictions = 0
    total_predictions = 0

    for batch in val_dataloader:

        # =======================================
        # 1) Attention scenario
        # =======================================
        if input_type == "attention":
            # This scenario requires gradient-based attention (gdp)
            img_arr, jsm_arr, targets = batch
            img_arr = img_arr.to(device)
            jsm_arr = jsm_arr.to(device)
            targets = targets.to(device)

            # Enable gradients for computing attention maps
            
            attention_weighted_img = apply_attention_map(img_arr, jsm_arr)

            # Evaluate without tracking further gradients
            with torch.no_grad():
                outputs = model(attention_weighted_img)
                probabilities = torch.softmax(outputs, dim=1)

        # =======================================
        # 2) Cross-Attention scenario
        # =======================================
        elif input_type == "cross_attention":
            img_arr, jsm_arr, targets = batch
            img_arr = img_arr.to(device)
            jsm_arr = jsm_arr.to(device)
            targets = targets.to(device)

            # No gradient needed for evaluation, so do a no_grad context
            with torch.no_grad():
                outputs = model(img_arr, jsm_arr)
                probabilities = torch.softmax(outputs, dim=1)

        # =======================================
        # 3) Concat, Single, or JSM scenarios
        # =======================================
        elif input_type in ["concat", "single", "jsm"]:
            with torch.no_grad():
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)

        else:
            raise ValueError(f"Unknown input type {input_type}")

        # =======================================
        # 4) Compute predictions and track metrics
        # =======================================
        predicted_classes = probabilities.argmax(dim=1)
        correct_predictions += (predicted_classes == targets).sum().item()
        total_predictions += targets.size(0)

        all_probabilities.extend(probabilities.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    # Convert to numpy
    all_targets_np = np.array(all_targets)
    all_probabilities_np = np.array(all_probabilities)

    # Compute AUC (ROC & PR)
    if num_classes > 2:
        # Multi-class
        roc_auc = roc_auc_score(all_targets_np, all_probabilities_np, multi_class="ovr")
        pr_auc = average_precision_score(all_targets_np, all_probabilities_np, average="macro")
    else:
        # Binary
        roc_auc = roc_auc_score(all_targets_np, all_probabilities_np[:, 1])
        pr_auc = average_precision_score(all_targets_np, all_probabilities_np[:, 1])

    # Other metrics
    predicted_classes_np = all_probabilities_np.argmax(axis=1)
    macro_f1 = f1_score(all_targets_np, predicted_classes_np, average="macro")
    precision = precision_score(all_targets_np, predicted_classes_np, average="macro")
    recall = recall_score(all_targets_np, predicted_classes_np, average="macro")
    accuracy = correct_predictions / total_predictions

    return roc_auc, pr_auc, accuracy, macro_f1, precision, recall


def cross_validate(
    model_class, 
    dataset, 
    num_classes, 
    device, 
    num_epochs, 
    learning_rate, 
    dtype, 
    model_name, 
    cls_type, 
    output_folder, 
    batch_size,
    fusion_type, 
    apply_rotation, 
    max_rotation, 
    thereshold, 
    input_type="single",
    qkv_mode="mri2jsm"   # <--- New argument here!
):
    os.makedirs(output_folder, exist_ok=True)
    results_path = os.path.join(output_folder, f'cross_validation_results_{dtype}_{model_name}_{cls_type}_{input_type}.csv')

    # Determine number of classes
    if cls_type == "cn_vs_mci_vs_ad":
        num_classes = 3
    else:
        num_classes = 2

    # Write CSV header
    with open(results_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Fold", "ROC AUC", "PR AUC", "Accuracy", "Macro F1", "Precision", "Recall"])

    all_metrics = []
    unique_folds = sorted(dataset["fold"].unique())

    for fold_idx in unique_folds:
        print(f"Starting Fold {fold_idx + 1}/{len(unique_folds)}...")

        # Split dataset into train and test
        test_data = dataset[dataset["fold"] == fold_idx]
        train_data = dataset[dataset["fold"] != fold_idx]
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        train_image_paths, train_JSM_paths, train_labels = load_data(train_data, dtype, cls_type, input_type)
        test_image_paths,  test_JSM_paths,  test_labels  = load_data(test_data,  dtype, cls_type, input_type)

        train_image_paths = train_image_paths.tolist()
        test_image_paths  = test_image_paths.tolist()
        train_JSM_paths   = train_JSM_paths.tolist() if train_JSM_paths is not None else None
        test_JSM_paths    = test_JSM_paths.tolist()  if test_JSM_paths is not None else None

        # Determine channels
        num_channels = 1 if input_type in ["single", "attention", "jsm", "cross_attention"] else 2

        # -----------------------------------------------
        # Choose which model to instantiate
        # -----------------------------------------------
        if input_type == "cross_attention":
            # Pass the new qkv_mode here
            model = CrossAttention3DClassifier(
                fusion_type = fusion_type,
                num_heads   = 4,
                num_classes = num_classes,
                qkv_mode    = qkv_mode
            )
        else:
            # Normal model instantiation
            model = model_class(input_channels=num_channels, num_classes=num_classes).to(device)

        # Handle multiple GPUs
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                print(f"Using {num_gpus} GPUs.")
                model = torch.nn.DataParallel(model)
            else:
                print("Using 1 GPU.")
        else:
            print("Using CPU (no GPU found).")

        model = model.to(device)

        train_dataset = ADNIMRIDataset(
            image_paths=train_image_paths,
            image_labels=train_labels,
            jsm_paths=train_JSM_paths if input_type in ["concat", "attention", "jsm", "cross_attention"] else None,
            input_type=input_type,
            apply_random_rotation=apply_rotation,
            max_rotation=max_rotation,
            thereshold=thereshold
        )

        test_dataset = ADNIMRIDataset(
            image_paths=test_image_paths,
            image_labels=test_labels,
            jsm_paths=test_JSM_paths if input_type in ["concat", "attention", "jsm", "cross_attention"] else None,
            input_type=input_type,
            apply_random_rotation=False,
            max_rotation=0,
            thereshold=thereshold
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Train
        model = train_model(model, train_dataloader, optimizer, criterion, num_epochs, device, input_type)

        # Validate
        roc_auc, pr_auc, accuracy, macro_f1, precision, recall = evaluate_model(
            model, test_dataloader, num_classes, device, input_type
        )

        print(
            f"Fold {fold_idx + 1} Metrics: AUC={roc_auc:.4f}, PR AUC={pr_auc:.4f}, "
            f"Accuracy={accuracy:.4f}, F1={macro_f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}"
        )

        fold_metrics = {
            "Fold": fold_idx + 1,
            "ROC AUC": roc_auc,
            "PR AUC": pr_auc,
            "Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Precision": precision,
            "Recall": recall,
        }
        all_metrics.append(fold_metrics)

        # Save results
        with open(results_path, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                fold_metrics["Fold"],
                fold_metrics["ROC AUC"],
                fold_metrics["PR AUC"],
                fold_metrics["Accuracy"],
                fold_metrics["Macro F1"],
                fold_metrics["Precision"],
                fold_metrics["Recall"],
            ])

    print(f"Results saved to {results_path}")
    return results_path




