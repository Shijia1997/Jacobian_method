import argparse
import os
import pandas as pd
import torch
from literature_models import J_CNN3DModel, JAL_model  # Replace with actual module
from data import ADNIMRIDataset  # Replace with actual module
from utils import cross_validate  # Replace with actual module

def main(args):
    # Load the dataset
    dataset = pd.read_csv(args.data_path)

    # Map classification labels based on the task
    if args.cls_type not in ['cn_vs_mci_vs_ad', 'cn_vs_rest', 'cn_vs_ad', 'cn_vs_mci', 'mci_vs_ad']:
        raise ValueError(f"Invalid classification type: {args.cls_type}")

    if args.cls_type == "cn_vs_mci_vs_ad":
        num_classes = 3
    else:
        num_classes = 2

    # Model selection
    if args.model_name == "J_CNN3D":
        model_class = J_CNN3DModel
    elif args.model_name == "JAL":
        model_class = JAL_model
    elif args.model_name == "Res10":
        from resnet_arc import generate_model
        model_class = lambda **kwargs: generate_model(10, n_input_channels=kwargs.get('input_channels', 1), 
                                                      n_classes=kwargs.get('num_classes', 2), 
                                                      shortcut_type="B")
    else:
        raise ValueError(f"Invalid model name: {args.model_name}")

    # Output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Iterate over data types and input types
    for dtype in args.data_types.split(','):
        for input_type in args.input_types.split(','):
            print(f"Running cross-validation for data type: {dtype}, input type: {input_type}")

            # Check if input type is valid
            valid_input_types = ["single", "concat", "attention", "jsm"]
            if input_type not in valid_input_types:
                raise ValueError(f"Invalid input type: {input_type}. Choose from {valid_input_types}.")

            # Handle JSM-only input type
            if input_type == "jsm" and dtype != "syn_registered":
                raise ValueError("JSM input type requires data type to be 'syn_registered'.")

            # Process attention type with alpha_value
            if input_type == "attention":
                cross_validate(
                    model_class=model_class,
                    dataset=dataset,
                    num_classes=num_classes,
                    device=args.device,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    dtype=dtype,
                    model_name=args.model_name,
                    cls_type=args.cls_type,
                    output_folder=args.output_folder,
                    alpha_value=args.alpha_value,
                    batch_size = 16,
                    input_type=input_type
                )
            else:
                # Run cross-validation for other input types
                cross_validate(
                    model_class=model_class,
                    dataset=dataset,
                    num_classes=num_classes,
                    device=args.device,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    dtype=dtype,
                    model_name=args.model_name,
                    cls_type=args.cls_type,
                    output_folder=args.output_folder,
                     alpha_value=args.alpha_value,
                    batch_size = 16,
                    input_type=input_type
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Cross-Validation on ADNI Dataset")

    # Required arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset CSV file")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save cross-validation results")
    parser.add_argument("--model_name", type=str, required=True, choices=["J_CNN3D", "JAL", "Res10"], 
                        help="Model architecture to use")
    parser.add_argument("--cls_type", type=str, required=True, 
                        choices=["cn_vs_mci_vs_ad", "cn_vs_rest", "cn_vs_ad", "cn_vs_mci", "mci_vs_ad"],
                        help="Classification type")

    # Optional arguments
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run the model (e.g., 'cuda', 'cpu')")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--data_types", type=str, required=True, 
                        help="Comma-separated list of data types to use (e.g., 'affine_registered,syn_registered')")
    parser.add_argument("--input_types", type=str, required=True, 
                        help="Comma-separated list of input types for the model (e.g., 'single,concat,attention,jsm')")
    parser.add_argument("--alpha_value", type=float, default=0.95, 
                        help="Alpha value for attention input type (default: 0.95)")

    args = parser.parse_args()
    main(args)
