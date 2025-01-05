import ants
import os
import pandas as pd

raw_image_path = "combined_dataset_corrected.csv"
stripped_output_path = "/dcs07/zwang/data/ADimagestripped/brain"
jacobian_output_path = "/dcs07/zwang/data/syn_jacobian_regular"

# Create a temporary directory for intermediate files
temp_dir = os.path.join(jacobian_output_path, "temp")
os.makedirs(temp_dir, exist_ok=True)

os.makedirs(jacobian_output_path, exist_ok=True)

# Read the fixed template image
fixed_image = ants.image_read(ants.get_ants_data('mni'))
print(f"MNI template resolution: {fixed_image.spacing}")

df = pd.read_csv(raw_image_path)
brain_image_names = df["Image ID"].tolist()

for brain_image_name in brain_image_names:
    print(f"Processing image {brain_image_name}")
    
    # Define a unique prefix for each transformation output
    output_prefix = os.path.join(temp_dir, f"transform_{brain_image_name}_")
    
    moving_image_path = os.path.join(stripped_output_path, f"{brain_image_name}_brain_N4_corrected_.nii")
    if not os.path.exists(moving_image_path):
        print(f"Moving image not found: {moving_image_path}")
        continue
    
    moving_image = ants.image_read(moving_image_path)

    try:
        # Perform SyN nonlinear registration
        nonlinear_result = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform='SyN',
            outprefix=output_prefix
        )
        
        # Check if transform files were generated
        if 'fwdtransforms' in nonlinear_result and len(nonlinear_result['fwdtransforms']) > 0:
            warp_field = nonlinear_result['fwdtransforms'][0]
            
            # Compute regular-Jacobian
            regular_jacobian = ants.create_jacobian_determinant_image(
                domain_image=fixed_image,
                tx=warp_field,
                do_log=False
            )

            regular_jacobian_path = os.path.join(jacobian_output_path, f"syn_regular_jacobian_{brain_image_name}.nii.gz")
            ants.image_write(regular_jacobian, regular_jacobian_path)
            print(f"Successfully processed {brain_image_name}, regular-Jacobian saved to {regular_jacobian_path}")
        else:
            print(f"Registration failed for {brain_image_name}: No transforms generated")
            
    except Exception as e:
        print(f"Error processing {brain_image_name}: {str(e)}")
        continue

print("All processing complete for syn regular jacobian.")
