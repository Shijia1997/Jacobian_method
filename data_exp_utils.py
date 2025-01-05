import ants
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def compute_voxel_stats_log(df, index, column_name):
    """
    Compute various statistics from a log Jacobian ANTs image.

    This function:
    - Separately computes sum, average, and count of voxels showing 
      local expansion (log Jacobian > 0)
    - Separately computes sum, average, and count of voxels showing 
      local contraction (log Jacobian < 0)
    - Computes the average absolute value across all non-zero voxels, 
      regardless of whether they indicate contraction or expansion.

    Parameters:
    - df: Pandas DataFrame containing image paths.
    - index: The row index of the DataFrame.
    - column_name: The column name containing the image path.

    Returns:
    - A dictionary with:
        'expansion_sum', 'expansion_avg', 'expansion_count'
        'contraction_sum', 'contraction_avg', 'contraction_count'
        'abs_non_zero_avg'
    """

    # Read the image using ANTs
    image_path = df.loc[index, column_name]
    image = ants.image_read(image_path)
    
    # Convert the image to a NumPy array
    voxel_values = image.numpy()

    # Consider only non-zero voxels
    non_zero_voxels = voxel_values[voxel_values != 0]

    # Separate into expansion (values > 0) and contraction (values < 0)
    expansion_voxels = non_zero_voxels[non_zero_voxels > 0]
    contraction_voxels = non_zero_voxels[non_zero_voxels < 0]

    # Compute expansion stats
    expansion_count = expansion_voxels.size
    expansion_sum = np.sum(expansion_voxels) if expansion_count > 0 else 0
    expansion_avg = expansion_sum / expansion_count if expansion_count > 0 else 0

    # Compute contraction stats
    contraction_count = contraction_voxels.size
    contraction_sum = np.sum(contraction_voxels) if contraction_count > 0 else 0
    contraction_avg = contraction_sum / contraction_count if contraction_count > 0 else 0

    # Compute absolute average across all non-zero voxels
    abs_non_zero_voxels = np.abs(non_zero_voxels)
    abs_non_zero_count = abs_non_zero_voxels.size
    abs_non_zero_sum = np.sum(abs_non_zero_voxels)
    abs_non_zero_avg = abs_non_zero_sum / abs_non_zero_count if abs_non_zero_count > 0 else 0

    return {
        'expansion_sum': expansion_sum,
        'expansion_avg': expansion_avg,
        'expansion_count': expansion_count,
        'contraction_sum': contraction_sum,
        'contraction_avg': contraction_avg,
        'contraction_count': contraction_count,
        'abs_non_zero_avg': abs_non_zero_avg
    }

def compute_voxel_stats_regular(df, index, column_name):
    """
    Compute various statistics from a regular Jacobian ANTs image.

    This function:
    - Separately computes sum, average, and count of voxels showing 
      local expansion (Jacobian > 1)
    - Separately computes sum, average, and count of voxels showing 
      local contraction (Jacobian < 1)
    - Computes the average absolute deviation from 1 across all non-one voxels.

    Parameters:
    - df: Pandas DataFrame containing image paths.
    - index: The row index of the DataFrame.
    - column_name: The column name containing the image path.

    Returns:
    - A dictionary with:
        'expansion_sum', 'expansion_avg', 'expansion_count'
        'contraction_sum', 'contraction_avg', 'contraction_count'
        'abs_non_one_avg'
    """

    # Read the image using ANTs
    image_path = df.loc[index, column_name]
    image = ants.image_read(image_path)
    
    # Convert the image to a NumPy array
    voxel_values = image.numpy()

    # Consider only non-one voxels
    non_one_voxels = voxel_values[voxel_values != 1]

    # Separate into expansion (values > 1) and contraction (values < 1)
    expansion_voxels = non_one_voxels[non_one_voxels > 1]
    contraction_voxels = non_one_voxels[non_one_voxels < 1]

    # Compute expansion stats
    expansion_count = expansion_voxels.size
    expansion_sum = np.sum(expansion_voxels) if expansion_count > 0 else 0
    expansion_avg = expansion_sum / expansion_count if expansion_count > 0 else 0

    # Compute contraction stats
    contraction_count = contraction_voxels.size
    contraction_sum = np.sum(contraction_voxels) if contraction_count > 0 else 0
    contraction_avg = contraction_sum / contraction_count if contraction_count > 0 else 0

    # Compute average absolute deviation from 1 across all non-one voxels
    abs_non_one_voxels = np.abs(non_one_voxels - 1)
    abs_non_one_count = abs_non_one_voxels.size
    abs_non_one_sum = np.sum(abs_non_one_voxels)
    abs_non_one_avg = abs_non_one_sum / abs_non_one_count if abs_non_one_count > 0 else 0

    return {
        'expansion_sum': expansion_sum,
        'expansion_avg': expansion_avg,
        'expansion_count': expansion_count,
        'contraction_sum': contraction_sum,
        'contraction_avg': contraction_avg,
        'contraction_count': contraction_count,
        'abs_non_one_avg': abs_non_one_avg
    }


def plot_registration_views(df, index):
    """
    Plot orthogonal views (axial, coronal, sagittal) for affine registered, Jacobian, and SyN registered images.

    Parameters:
    - df: Pandas DataFrame containing paths to the images with columns:
          'linear_registered_path', 'non_linear_registered_path', 'syn_jacobian'.
    - index: Index of the row in the dataframe to plot.

    Returns:
    - None. Displays the 3x3 panel plot.
    """
    # Extract paths from the dataframe
    affine_path = df.loc[index, 'linear_registered_path']
    jacobian_path = df.loc[index, 'non_linear_registered_path']
    syn_path = df.loc[index, 'syn_jacobian']

    # Load images using ANTs
    affine_reg = ants.image_read(affine_path)
    jacobian = ants.image_read(jacobian_path)
    syn_reg = ants.image_read(syn_path)

    # Extract slices from the middle along each dimension
    affine_axial = affine_reg.numpy()[:, :, affine_reg.shape[2] // 2]
    affine_coronal = affine_reg.numpy()[:, affine_reg.shape[1] // 2, :]
    affine_sagittal = affine_reg.numpy()[affine_reg.shape[0] // 2, :, :]

    jacobian_axial = jacobian.numpy()[:, :, jacobian.shape[2] // 2]
    jacobian_coronal = jacobian.numpy()[:, jacobian.shape[1] // 2, :]
    jacobian_sagittal = jacobian.numpy()[jacobian.shape[0] // 2, :, :]

    syn_axial = syn_reg.numpy()[:, :, syn_reg.shape[2] // 2]
    syn_coronal = syn_reg.numpy()[:, syn_reg.shape[1] // 2, :]
    syn_sagittal = syn_reg.numpy()[syn_reg.shape[0] // 2, :, :]

    # Create a panel for three orthogonal views
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # Affine Registered
    axs[0, 0].imshow(affine_axial, cmap='gray')
    axs[0, 0].set_title('Affine - Axial')
    axs[0, 0].axis('off')

    axs[1, 0].imshow(affine_coronal, cmap='gray')
    axs[1, 0].set_title('Affine - Coronal')
    axs[1, 0].axis('off')

    axs[2, 0].imshow(affine_sagittal, cmap='gray')
    axs[2, 0].set_title('Affine - Sagittal')
    axs[2, 0].axis('off')

    # Jacobian
    axs[0, 1].imshow(jacobian_axial, cmap='gray')
    axs[0, 1].set_title('Jacobian - Axial')
    axs[0, 1].axis('off')

    axs[1, 1].imshow(jacobian_coronal, cmap='gray')
    axs[1, 1].set_title('Jacobian - Coronal')
    axs[1, 1].axis('off')

    axs[2, 1].imshow(jacobian_sagittal, cmap='gray')
    axs[2, 1].set_title('Jacobian - Sagittal')
    axs[2, 1].axis('off')

    # Syn Registered
    axs[0, 2].imshow(syn_axial, cmap='gray')
    axs[0, 2].set_title('Syn - Axial')
    axs[0, 2].axis('off')

    axs[1, 2].imshow(syn_coronal, cmap='gray')
    axs[1, 2].set_title('Syn - Coronal')
    axs[1, 2].axis('off')

    axs[2, 2].imshow(syn_sagittal, cmap='gray')
    axs[2, 2].set_title('Syn - Sagittal')
    axs[2, 2].axis('off')

    plt.tight_layout()
    plt.show()




def plot_registration_overlay_views(df, index):
    """
    Plot overlays of affine registered image and SyN registered image with Jacobian image.

    Parameters:
    - df: Pandas DataFrame containing paths to the images with columns:
          'linear_registered_path', 'non_linear_registered_path', 'syn_jacobian'.
    - index: Index of the row in the dataframe to plot.

    Returns:
    - None. Displays the 2x3 panel plot for each overlay.
    """
    # Extract paths from the dataframe
    affine_path = df.loc[index, 'linear_registered_path']
    syn_path = df.loc[index, 'non_linear_registered_path']
    jacobian_path = df.loc[index, 'syn_jacobian']

    # Load images using ANTs
    affine_reg = ants.image_read(affine_path)
    syn_reg = ants.image_read(syn_path)
    jacobian = ants.image_read(jacobian_path)

    # Extract slices from the middle along each dimension
    affine_axial = affine_reg.numpy()[:, :, affine_reg.shape[2] // 2]
    affine_coronal = affine_reg.numpy()[:, affine_reg.shape[1] // 2, :]
    affine_sagittal = affine_reg.numpy()[affine_reg.shape[0] // 2, :, :]

    syn_axial = syn_reg.numpy()[:, :, syn_reg.shape[2] // 2]
    syn_coronal = syn_reg.numpy()[:, syn_reg.shape[1] // 2, :]
    syn_sagittal = syn_reg.numpy()[syn_reg.shape[0] // 2, :, :]

    jacobian_data = jacobian.numpy()
    jacobian_axial = jacobian_data[:, :, jacobian.shape[2] // 2]
    jacobian_coronal = jacobian_data[:, jacobian.shape[1] // 2, :]
    jacobian_sagittal = jacobian_data[jacobian.shape[0] // 2, :, :]

    # Determine the range of Jacobian values for accurate color mapping
    jacobian_min = np.min(jacobian_data)
    jacobian_max = np.max(jacobian_data)

    # Create a diverging colormap centered at zero
    cmap = mcolors.TwoSlopeNorm(vmin=jacobian_min, vcenter=0, vmax=jacobian_max)

    # Create overlay plots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # Affine + Jacobian Overlay
    im0 = axs[0, 0].imshow(affine_axial, cmap='gray')
    im1 = axs[0, 0].imshow(jacobian_axial, cmap='coolwarm', alpha=0.5, norm=cmap)
    axs[0, 0].set_title('Affine + Jacobian - Axial')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(affine_coronal, cmap='gray')
    axs[0, 1].imshow(jacobian_coronal, cmap='coolwarm', alpha=0.5, norm=cmap)
    axs[0, 1].set_title('Affine + Jacobian - Coronal')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(affine_sagittal, cmap='gray')
    axs[0, 2].imshow(jacobian_sagittal, cmap='coolwarm', alpha=0.5, norm=cmap)
    axs[0, 2].set_title('Affine + Jacobian - Sagittal')
    axs[0, 2].axis('off')

    # Add colorbar for Jacobian
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.3])
    plt.colorbar(im1, cax=cbar_ax, label='Jacobian Determinant')

    # SyN + Jacobian Overlay
    im2 = axs[1, 0].imshow(syn_axial, cmap='gray')
    axs[1, 0].imshow(jacobian_axial, cmap='coolwarm', alpha=0.5, norm=cmap)
    axs[1, 0].set_title('SyN + Jacobian - Axial')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(syn_coronal, cmap='gray')
    axs[1, 1].imshow(jacobian_coronal, cmap='coolwarm', alpha=0.5, norm=cmap)
    axs[1, 1].set_title('SyN + Jacobian - Coronal')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(syn_sagittal, cmap='gray')
    axs[1, 2].imshow(jacobian_sagittal, cmap='coolwarm', alpha=0.5, norm=cmap)
    axs[1, 2].set_title('SyN + Jacobian - Sagittal')
    axs[1, 2].axis('off')

    # Add colorbar for Jacobian
    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.3])
    plt.colorbar(im1, cax=cbar_ax2, label='Jacobian Determinant')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()



def plot_mri_jsm_overlay_views_loaded(mri_ants, jsm_ants, title_prefix="MRI+JSM"):
    """
    Overlay the JSM (Jacobian) on top of an MRI using in-memory ANTs images.

    Parameters
    ----------
    mri_ants : ants.ANTsImage
        The MRI volume as an ANTs image.

    jsm_ants : ants.ANTsImage
        The JSM volume as an ANTs image.

    title_prefix : str
        A prefix for the subplot titles.

    Returns
    -------
    None. Displays a 1x3 overlay for Axial, Coronal, and Sagittal.
    """
    # Convert them to numpy
    mri_data = mri_ants.numpy()
    jsm_data = jsm_ants.numpy()

    # Extract middle slices
    axial_idx     = mri_data.shape[2] // 2
    coronal_idx   = mri_data.shape[1] // 2
    sagittal_idx  = mri_data.shape[0] // 2

    mri_axial     = mri_data[:, :, axial_idx]
    mri_coronal   = mri_data[:, coronal_idx, :]
    mri_sagittal  = mri_data[sagittal_idx, :, :]

    jsm_axial     = jsm_data[:, :, axial_idx]
    jsm_coronal   = jsm_data[:, coronal_idx, :]
    jsm_sagittal  = jsm_data[sagittal_idx, :, :]

    # Determine a suitable min/max for the JSM for color mapping
    jsm_min = np.min(jsm_data)
    jsm_max = np.max(jsm_data)

    # Create a diverging colormap, centered at 0 (if JSM can be negative/positive)
    cmap = mcolors.TwoSlopeNorm(vmin=jsm_min, vcenter=0.5, vmax=jsm_max)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Axial
    im1 = axs[0].imshow(mri_axial, cmap='gray')
    im1 = axs[0].imshow(jsm_axial, cmap='coolwarm', alpha=0.4, norm=cmap)
    axs[0].set_title(f"{title_prefix} - Axial")
    axs[0].axis('off')

    # Coronal
    im2 = axs[1].imshow(mri_coronal, cmap='gray')
    im2 = axs[1].imshow(jsm_coronal, cmap='coolwarm', alpha=0.4, norm=cmap)
    axs[1].set_title(f"{title_prefix} - Coronal")
    axs[1].axis('off')

    # Sagittal
    im3 = axs[2].imshow(mri_sagittal, cmap='gray')
    im3 = axs[2].imshow(jsm_sagittal, cmap='coolwarm', alpha=0.4, norm=cmap)
    axs[2].set_title(f"{title_prefix} - Sagittal")
    axs[2].axis('off')

    # Colorbar
    cbar = plt.colorbar(im3, ax=axs, fraction=0.025, pad=0.04)
    cbar.set_label("JSM Intensity")

    plt.tight_layout()
    plt.show()
