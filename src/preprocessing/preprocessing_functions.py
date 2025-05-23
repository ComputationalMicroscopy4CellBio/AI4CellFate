import numpy as np
import pandas as pd
import glob
import tifffile as tiff
from scipy.ndimage import label, find_objects


############ TRACKS ONLY ############

class PreProcess:
    
    def __init__(self):
        pass
    
    def preprocess_data(self, data, fixed_length):
        """
        Reshape the input data by padding sequences to a fixed length (1080).
        """
        cell_sequences = {}
        # Iterate over each row in the data
        for row in data:
            cell_id = row[0]  # Extract cell ID from the first column
            # If cell ID not in dictionary, create a new entry
            if cell_id not in cell_sequences:
                cell_sequences[cell_id] = []
            # Append the row (sample) to the corresponding cell's sequence
            cell_sequences[cell_id].append(row[:])  # Exclude cell ID from sequence
        # Pad or truncate sequences for each cell
        processed_data = []
        for cell_id, sequence in cell_sequences.items():
            # Convert list of lists to numpy array
            sequence = np.array(sequence)
            seq_length = len(sequence)
            # Pad or truncate the sequence to the fixed length
            if seq_length < fixed_length:
                # If sequence is shorter, pad with zeros
                padding = np.zeros((fixed_length - seq_length, sequence.shape[1]))
                processed_sequence = np.vstack((sequence, padding))
            elif seq_length > fixed_length:
                # If sequence is longer, truncate to fixed length
                processed_sequence = sequence[:fixed_length, :]
            else:
                # If sequence length matches fixed length, no need for processing
                processed_sequence = sequence
            processed_data.append(processed_sequence)

        processed_data = np.array(processed_data)
        return processed_data
    

    def apply_mean_filter(self, data, window_size):
        # Iterate over each sample (cell)
        filtered_data = []
        for sample in data:
            # Apply mean filtering to each feature over the specified window size
            filtered_sample = np.zeros((sample.shape[0] // window_size, sample.shape[1]))
            for i in range(sample.shape[1]):  # Iterate over features
                feature_values = sample[:, i]  # Extract feature values
                # Reshape feature values into a 2D array with window_size columns
                reshaped_values = feature_values[:len(feature_values) // window_size * window_size].reshape(-1, window_size)
                # Compute the mean of each window and store in the filtered sample
                filtered_feature = np.mean(reshaped_values, axis=1)
                filtered_sample[:, i] = filtered_feature
            filtered_data.append(filtered_sample)
        # Convert list of arrays to numpy array
        filtered_data = np.array(filtered_data)
        return filtered_data
    

def process_track_data(track):
    # This includes all extracted features
    desired_features = [
    "unique_track_id", "xpos", "ypos", "tpos", "cell_av_FRET_norm", "CFP_av", "YFP_av", 
    "CFP_std", "YFP_std", "MajorAxisLength", "MinorAxisLength", "Eccentricity", "EquivDiameter", 
    "Perimeter", "INT_av", "Area", "Aspect_Ratio", "mitosis_time", "fate"]

    track = track[desired_features]
    data_array = track.iloc[:].values # getting data as numpy array

    # Re-structuring data
    preprocess = PreProcess()
    padded_data = preprocess.preprocess_data(data_array, fixed_length=1080)
    padded_data = np.asarray(padded_data, dtype=np.float32)
    filtered_data = np.nan_to_num(padded_data, nan=0.0)

    y_train = np.empty((padded_data.shape[0])) # get fates (last column)
    for cell in range(padded_data.shape[0]):
        y_train[cell] = padded_data[cell][0][-1]

    print("Filtered data shape", filtered_data.shape, y_train.shape)
    return filtered_data, y_train


############ IMAGES ############

def make_subimage(image_stack, sub_size, x_pos, y_pos, t_pos, cell_ids, segmentation):
    unique_cell_ids = np.unique(cell_ids)
    num_cells = len(unique_cell_ids)
    max_time_points = 1080
    channels = image_stack.shape[1]
    pad_x, pad_y = sub_size[0] // 2, sub_size[1] // 2
    
    # Because the segmented images only have 1 channel
    if segmentation == True:
        # Create cropped images and pad images with zeros to avoid border issues
        sub_images = np.zeros((num_cells, max_time_points, sub_size[0], sub_size[1]))
        padded_images = np.pad(image_stack, ((0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode='constant')
    else:
        sub_images = np.zeros((num_cells, max_time_points, channels, sub_size[0], sub_size[1]))
        padded_images = np.pad(image_stack, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)), mode='constant')
    
    for cell_index, cell_id in enumerate(unique_cell_ids):
        cell_mask = cell_ids == cell_id
        cell_x_pos = x_pos[cell_mask]
        cell_y_pos = y_pos[cell_mask]
        cell_t_pos = t_pos[cell_mask]
        
        for t, (x, y, t_frame) in enumerate(zip(cell_x_pos, cell_y_pos, cell_t_pos)):
            x_center = int(x) + pad_x
            y_center = int(y) + pad_y
            t_index = int(t_frame) - 1
            
            if t_index >= padded_images.shape[0]:
                # Skip if the time index exceeds the number of frames in the image stack
                continue
            
            # Define the sub-image centered around the cell

            if segmentation == True:
                cropped_img = padded_images[t_index, y_center-pad_y:y_center+pad_y, x_center-pad_x:x_center+pad_x]
                sub_images[cell_index, t, :, :] = cropped_img
            else:
                cropped_img = padded_images[t_index, :, y_center-pad_y:y_center+pad_y, x_center-pad_x:x_center+pad_x]
                sub_images[cell_index, t, :, :, :] = cropped_img
    
    return sub_images


def load_images_for_fov(fov_index, bioreplicate, segmentation, FRET):

    if segmentation == False: # Acquired images (not segmented or FRET)
        print(f"bioreplicate: {bioreplicate}")
        if bioreplicate == "BR1":
            pattern = f'/Users/inescunha/Documents/GitHub/explanation/{bioreplicate}_Images/Tiff/ALM272_B*_G00{fov_index}_0001.tif'
        else:
            pattern = f'/Users/inescunha/Documents/GitHub/explanation/{bioreplicate}_Images/ALM272_B*_G00{fov_index}_0001*.tif'

    elif FRET == True: # FRET images (in the FRET case: segmentation = True and FRET = True)
        pattern = f'/Users/inescunha/Documents/GitHub/explanation/{bioreplicate}_Images/Segmentations_and_FRET/ALM272_{bioreplicate}_{fov_index}_FRET.tiff'

    else: # Segmented images (segmentation = True and FRET = False)
        pattern = f'/Users/inescunha/Documents/GitHub/explanation/{bioreplicate}_Images/Segmentations_and_FRET/ALM272_{bioreplicate}_{fov_index}_seg.tiff'

    files = glob.glob(pattern)
    print(files)
    if files:
        return tiff.imread(files[0])
    else:
        raise FileNotFoundError(f"No file found for FOV index {fov_index}")
    

def process_all_fovs(track_data, sub_size, bioreplicate = "BR1", segmentation = False, FRET = False):
    # Detect changes in the FOV column

    track_data['fov_change'] = track_data['fov'] != track_data['fov'].shift(1)
    fov_change_indices = track_data.index[track_data['fov_change']].tolist()
    
    all_sub_images = []
    all_cell_ids = []
    all_fates = []
    current_fov_index = 1
    if bioreplicate == "BR2":
        current_fov_index = np.asarray([1, 2, 6, 8, 9]) 
    
    for i in range(len(fov_change_indices)): # len(fov_change_indices)-1
        start_index = fov_change_indices[i]
        end_index = fov_change_indices[i + 1] if i + 1 < len(fov_change_indices) else len(track_data)
        
        fov_mask = (track_data.index >= start_index) & (track_data.index < end_index)
        x_pos = track_data.loc[fov_mask, 'xpos'].values
        y_pos = track_data.loc[fov_mask, 'ypos'].values
        t_pos = track_data.loc[fov_mask, 'tpos'].values
        cell_ids = track_data.loc[fov_mask, 'unique_track_id'].values
        fates = track_data.loc[fov_mask, 'fate'].values
        print(cell_ids)
        if bioreplicate == "BR2":
            fov_images = load_images_for_fov(current_fov_index[i], bioreplicate, segmentation, FRET)
        else:
            fov_images = load_images_for_fov(current_fov_index, bioreplicate, segmentation, FRET)
        sub_images = make_subimage(fov_images, sub_size, x_pos, y_pos, t_pos, cell_ids, segmentation)
        
        all_sub_images.append(sub_images)
        all_cell_ids.extend(np.unique(cell_ids))
        
        # Since the fate is the same for each unique cell, we take the first occurrence
        unique_fates = [fates[cell_ids == cell_id][0] for cell_id in np.unique(cell_ids)]
        all_fates.extend(unique_fates)
        
        if bioreplicate != "BR2":
            current_fov_index += 1  # Move to the next FOV
    
    # Combine all sub-images from different FOVs into a single array
    num_cells = len(all_cell_ids)
    max_time_points = 1080

    if segmentation == True:
        final_sub_images = np.zeros((num_cells, max_time_points, sub_size[0], sub_size[1]))
    
        start_idx = 0
        for sub_images in all_sub_images:
            num_fov_cells = sub_images.shape[0]
            final_sub_images[start_idx:start_idx+num_fov_cells, :, :, :] = sub_images
            start_idx += num_fov_cells
    
    else:
        channels = all_sub_images[0].shape[2]  # Assuming the same number of channels for all FOVs
        
        final_sub_images = np.zeros((num_cells, max_time_points, channels, sub_size[0], sub_size[1]))
        
        start_idx = 0
        for sub_images in all_sub_images:
            num_fov_cells = sub_images.shape[0]
            final_sub_images[start_idx:start_idx+num_fov_cells, :, :, :, :] = sub_images
            start_idx += num_fov_cells
    
    return final_sub_images, np.array(all_fates)


def edge_indexes(image):

    # Check on the first frame and first channel
    first_time_point_first_channel = image[:, 0, 0, :, :]
    indices_to_remove = []

    # Iterate over each cell to check for rows or columns full of zeros at the first time point
    for cell_index in range(first_time_point_first_channel.shape[0]):
        # Extract the cell's data at the first time point
        cell_data = first_time_point_first_channel[cell_index]
        
        # Check if there is any row or column that is entirely zeros
        has_zero_row = np.all(cell_data == 0, axis=1).any()  # Check rows
        has_zero_column = np.all(cell_data == 0, axis=0).any()  # Check columns
        
        if has_zero_row or has_zero_column:
            indices_to_remove.append(cell_index)

    return indices_to_remove

# Because my segmentations weren't binarised
def binarize_segmentation(segmentation):
    return (segmentation > 0).astype(int)

def overimpose(images, segmentation):
    overimposed_images = np.zeros_like(images[:,:,:2,:,:])
    overimposed_images[:,:,0,:,:] = images[:,:,0,:,:] * binarize_segmentation(segmentation)
    overimposed_images[:,:,1,:,:] = images[:,:,1,:,:] * binarize_segmentation(segmentation)
    return overimposed_images


#### Removing daughter cells from tracks and images ####

def daughter_indexes(track_data):
    indices_to_remove = []

    # Iterate over each cell
    for cell_index in range(track_data.shape[0]):
        # Get the `tpos` feature (feature index 3) across all time points for this cell
        tpos_values = track_data[cell_index, :, 3]
        
        # Check if all required values (1, 2, and 3) are present
        if not all(val in tpos_values for val in [1, 2, 3]):
            indices_to_remove.append(cell_index)

    return indices_to_remove


def daughter_trace_removal(tabular_data, image_data):
    """
    Removes information from daughter cells in both tabular and image data.
    
    - For each cell (row) in tabular_data, it finds the first zero in column 4, **ignoring the first time point**.
    - From that time point onward, it sets all values in that row to zero.
    - The same index is used to set corresponding images in image_data to zero.

    Args:
        tabular_data (numpy.ndarray): A 3D matrix of shape (cells, time, features).
        image_data (numpy.ndarray): A 5D matrix of shape (cells, time, channels, height, width).

    Returns:
        tuple: Processed (tabular_data, image_data) with zeroed-out values from the determined index onward.
    """
    processed_tabular = tabular_data.copy()
    processed_images = image_data.copy()
    
    for cell_idx, cell in enumerate(processed_tabular):
        # Ignore the first time point and find the first zero in the 4th column (index 3)
        feature_values = cell[1:, 4]  # Start from the second time point
        first_zero_index = np.argmax(feature_values == 0) + 1 if np.any(feature_values == 0) else None  # Shift by 1
        
        # If a zero is found, zero out both tabular and image data from that time point onward
        if first_zero_index is not None:
            processed_tabular[cell_idx, first_zero_index:, :] = 0
            processed_images[cell_idx, first_zero_index:, :, :, :] = 0  # Zeroing corresponding images
    
    return processed_tabular, processed_images


def remove_debris_from_fov(fov_image):
    """
    Removes debris from a given FOV, keeping only the largest connected component.
    
    Parameters:
    - fov_image (numpy array): 2D array (height, width) representing the FOV.
    
    Returns:
    - cleaned_image (numpy array): 2D array with only the main cell preserved.
    """
    # Identify connected components in the FOV
    labeled_image, num_features = label(fov_image > 0)  # Label all non-zero regions
    
    if num_features == 0:
        # No cells detected, return the input (empty FOV)
        return fov_image

    # Measure the size of each component
    component_sizes = np.array([np.sum(labeled_image == label) for label in range(1, num_features + 1)])

    # Find the largest component (assuming it's the main cell)
    largest_component_label = np.argmax(component_sizes) + 1

    # Create a cleaned image containing only the largest component
    cleaned_image = np.zeros_like(fov_image)
    cleaned_image[labeled_image == largest_component_label] = fov_image[labeled_image == largest_component_label]

    return cleaned_image

# Example usage
# Assuming `cell_images` is the numpy array of shape (cell, channel, height, width)
def clean_all_cells(cell_images):
    """
    Apply debris removal to all cells in a dataset.
    
    Parameters:
    - cell_images (numpy array): Array of shape (cell, channel, height, width).
    
    Returns:
    - cleaned_images (numpy array): Cleaned array of the same shape.
    """
    cleaned_images = np.zeros_like(cell_images)
    for cell_idx in range(cell_images.shape[0]):
        for time_idx in range(cell_images.shape[1]):
            for channel_idx in range(cell_images.shape[2]):
                cleaned_images[cell_idx, time_idx, channel_idx] = remove_debris_from_fov(cell_images[cell_idx, time_idx, channel_idx])
    return cleaned_images


def center_cells(images):
    """
    Centers each cell in the middle of a 20x20 field-of-view.
    
    Args:
        images: NumPy array of shape (num_cells, frame, 20, 20), where each cell is a binary or intensity image.

    Returns:
        Centered images of the same shape (num_cells, frame, 20, 20).
    """
    num_cells, frame, height, width = images.shape
    centered_images = np.zeros_like(images)

    for i in range(num_cells):
        for j in range(frame):
            img = images[i, j]

            # Find nonzero pixel indices
            y_indices, x_indices = np.where(img > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue  # Skip empty images
            
            # Get bounding box
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            # Extract the cell
            cropped_cell = img[y_min:y_max+1, x_min:x_max+1]

            # Compute new position to center the cropped cell
            new_y_start = (height - cropped_cell.shape[0]) // 2
            new_x_start = (width - cropped_cell.shape[1]) // 2
            
            # Place cropped cell into the new centered array
            centered_images[i, j, new_y_start:new_y_start+cropped_cell.shape[0], 
                                new_x_start:new_x_start+cropped_cell.shape[1]] = cropped_cell

    return centered_images


def normalize_channels(cell_images):
    """
    Normalize donor and acceptor channels to preserve their ratio.
    
    Parameters:
    - cell_images (numpy array): Array of shape (cell, channel, height, width).
    
    Returns:
    - normalized_images (numpy array): Normalized array of the same shape.
    """
    # Find the absolute maximum value across all cells and channels
    absolute_max = np.max(cell_images)
    
    # Normalize each channel by the same absolute maximum
    normalized_images = cell_images / absolute_max
    
    return normalized_images

def normalize_images_per_frame(cell_images):
    """
    Normalize donor and acceptor channels to preserve their ratio in each frame.
    
    Parameters:
    - cell_images (numpy array): Array of shape (cell, time, height, width).
    
    Returns:
    - normalized_images (numpy array): Normalized array of the same shape.
    """
    # Find the absolute maximum value across all cells and channels
    cell, frame, height, width = cell_images.shape
    normalized_images = np.zeros_like(cell_images)

    for j in range(frame):
        absolute_max = np.max(cell_images[:,j])
        normalized_images[:,j] = cell_images[:,j] / absolute_max
        
    return normalized_images


def augment_dataset(cell_images, labels, augmentations, augment_times=5, fate_0_label=0, fate_1_label=1, seed=42):
    """
    Augments a dataset of cell movies (cells, time, height, width), balances fate 0 and fate 1,
    and ensures reproducibility by setting a seed.

    Args:
        cell_images (numpy.ndarray): Array of shape (cells, time, height, width).
        labels (numpy.ndarray): 1D array of labels corresponding to cell fates.
        augmentations (function): Function that applies augmentations to a movie sequence.
        augment_times (int): Number of unique augmented copies per movie.
        fate_0_label (int): Label for fate 0 cells.
        fate_1_label (int): Label for fate 1 cells.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Augmented and balanced (final_images, final_labels).
    """
    np.random.seed(seed)  # Ensure reproducibility

    # Separate fate 0 and fate 1 in the dataset
    fate_0_images = cell_images[labels == fate_0_label]
    fate_1_images = cell_images[labels == fate_1_label]

    # Generate multiple unique augmentations per movie
    augmented_fate_0 = np.concatenate([augmentations(movie, augment_times) for movie in fate_0_images], axis=0)
    augmented_fate_1 = np.concatenate([augmentations(movie, augment_times) for movie in fate_1_images], axis=0)

    # Combine original and augmented data
    fate_0_combined = np.concatenate([fate_0_images, augmented_fate_0], axis=0)
    fate_1_combined = np.concatenate([fate_1_images, augmented_fate_1], axis=0)

    # Balance fate 0 to match fate 1 count
    target_count = len(fate_1_combined)
    if len(fate_0_combined) < target_count:
        print("did this happen")
        extra_fate_0 = np.random.choice(len(fate_0_combined), size=target_count - len(fate_0_combined), replace=True)
        balanced_fate_0 = np.concatenate([fate_0_combined, fate_0_combined[extra_fate_0]], axis=0)
    else:
        balanced_fate_0 = fate_0_combined[np.random.choice(len(fate_0_combined), size=target_count, replace=False)]

    # Merge datasets
    final_images = np.concatenate([balanced_fate_0, fate_1_combined], axis=0)
    final_labels = np.array([fate_0_label] * len(balanced_fate_0) + [fate_1_label] * len(fate_1_combined))

    # Shuffle data
    shuffle_indexes = np.random.permutation(len(final_labels))
    final_images, final_labels = final_images[shuffle_indexes], final_labels[shuffle_indexes]

    return final_images, final_labels


def augmentations(movie, augment_times=4, seed=42):
    """
    Applies multiple **unique** augmentations to the whole movie (time-series of images).

    Args:
        movie (numpy.ndarray): Array of shape (time, height, width).
        augment_times (int): Number of unique augmented copies to generate.

    Returns:
        numpy.ndarray: Augmented movies (augment_times, time, height, width).
    """
    all_augmentations = [
        lambda x: np.flip(x, axis=-1),  # Flip horizontally
        lambda x: np.flip(x, axis=-2),  # Flip vertically
        lambda x: np.rot90(x, k=1, axes=(-2, -1)),  # Rotate 90 degrees
        lambda x: np.rot90(x, k=2, axes=(-2, -1)),  # Rotate 180 degrees
        lambda x: np.rot90(x, k=3, axes=(-2, -1))  # Rotate 270 degrees
    ]
    
    np.random.seed(seed)

    # Select random augmentations
    np.random.shuffle(all_augmentations)

    selected_augmentations = all_augmentations[:augment_times]

    # Apply selected augmentations
    augmented_movies = [augment(movie) for augment in selected_augmentations]

    return np.array(augmented_movies)  # Shape: (augment_times, time, height, width)

def stretch_intensities_global(train_images, test_images, epsilon=0.001):
    """
    Stretch the intensities of train_images and test_images globally using the same min and max.
    
    Args:
        train_images (ndarray): Training image data
        test_images (ndarray): Testing image data
        epsilon (float): Value to prevent division by zero

    Returns:
        scaled_train_images (ndarray): Scaled training images
        scaled_test_images (ndarray): Scaled testing images
    """
    # Combine the datasets to compute global min and max
    cells, frames, height, width = train_images.shape
    
    scaled_train_images = np.empty(train_images.shape)
    scaled_test_images = np.empty(test_images.shape)

    for i in range(frames):
        combined_images = np.concatenate([train_images[:, i][train_images[:, i] > 0], test_images[:, i][test_images[:, i] > 0]])
        global_min = np.min(combined_images)
        global_max = np.max(combined_images)

        if global_max > global_min:

            scaled_train_images[:,i] = (train_images[:,i] - global_min) / (global_max - global_min)
            scaled_train_images[:,i] = scaled_train_images[:,i] * (1 - epsilon) + epsilon
            scaled_train_images[:,i][train_images[:,i] == 0] = 0  # Preserve zero values (background)

            scaled_test_images[:,i] = (test_images[:,i] - global_min) / (global_max - global_min)
            scaled_test_images[:,i] = scaled_test_images[:,i] * (1 - epsilon) + epsilon
            scaled_test_images[:,i][test_images[:,i] == 0] = 0  # Preserve zero values (background)

        else:
            # If global_max == global_min, leave unchanged
            scaled_train_images[:,i] = train_images[:,i]
            scaled_test_images[:,i] = test_images[:,i]

    return scaled_train_images, scaled_test_images



#### Filtering "problematic" cells #### 

def find_problematic_cells_and_times(data, threshold=4):
    """
    Efficiently finds indexes of cells and time points with at least one pixel value 
    significantly higher than all other pixels in the same image.
    
    Args:
        data (numpy.ndarray): The input data of shape (cells, time, height, width).
        threshold (int): The minimum difference to consider a pixel problematic.
        
    Returns:
        list: Sorted list of tuples (cell_index, time_index) with problematic pixels.
    """
    problematic_indices = []

    # Iterate over each cell and time point
    for cell_idx in range(data.shape[0]):
        for time_idx in range(data.shape[1]):
            image = data[cell_idx, time_idx]  # Shape: (20, 20)
            
            # Flatten the image to easily exclude each pixel
            flat_image = image.flatten()
            
            # Get max value of the entire image
            max_pixel = np.max(flat_image)
            
            # Get the max value of all other pixels by excluding each pixel one by one
            for pixel_idx, pixel_value in enumerate(flat_image):
                # Mask to exclude the current pixel
                mask = np.ones(flat_image.shape, dtype=bool)
                mask[pixel_idx] = False
                
                # Maximum value among the other pixels
                max_others = np.max(flat_image[mask])
                
                # Check if the current pixel is significantly higher
                if pixel_value > (max_others + threshold):
                    problematic_indices.append((cell_idx, time_idx))
                    break  # No need to check other pixels for this image
    
    return sorted(problematic_indices)


def remove_and_replace(data, cell_index, time_index):
    """
    Remove the specified image for both channels and add a zero-only image 
    at the last time point to maintain the shape.
    
    Args:
        data (numpy.ndarray): The input data of shape (cells, time, channels, height, width).
        cell_index (int): The index of the cell to modify.
        time_index (int): The time point to remove.
        
    Returns:
        numpy.ndarray: The modified data.
    """
    # Extract shape information
    _, time_points, channels, height, width = data.shape
    new_data = data.copy()
    # Remove the specified time point
    modified_cell_data = np.delete(new_data[cell_index], time_index, axis=0)  # Shape: (1079, 2, 20, 20)

    # Create a zero-only image
    zero_image = np.zeros((1, channels, height, width))  # Shape: (1, 2, 20, 20)

    # Append the zero-only image at the last time point
    modified_cell_data = np.concatenate([modified_cell_data, zero_image], axis=0)  # Back to (1080, 2, 20, 20)

    # Assign the modified data back
    new_data[cell_index] = modified_cell_data

    return new_data


#### Cell Clock Normalisation ####

def normalize_cell_time(data):
    """
    Normalize cell time by extracting feature values at 0%, 10%, ..., 100% 
    of each cell's lifetime.
    
    Parameters:
    - data: numpy array of shape (cells, time points, features)
      (assumes the first feature is used to determine lifetime)
    
    Returns:
    - normalized_data: numpy array of shape (cells, 11, features)
    """
    num_cells, num_timepoints, num_features = data.shape
    normalized_data = np.zeros((num_cells, 11, num_features))

    for i in range(num_cells):
        # Determine lifetime (number of nonzero values in the first feature)
        lifetime = np.count_nonzero(data[i, :, 0])

        if lifetime == 0:
            continue  # Skip cells that are all zero

        # Compute the indices corresponding to 0%, 10%, ..., 100% of the lifetime
        indices = np.round(np.linspace(0, lifetime - 1, 11)).astype(int)

        # Extract feature values at these indices
        normalized_data[i] = data[i, indices, :]

    return normalized_data


def normalize_cell_time_images(images):
    """
    Normalize cell time by extracting images at 0%, 10%, ..., 100% 
    of each cell's lifetime.
    
    Parameters:
    - images: numpy array of shape (cells, time, height, width)
      (assumes a cell's lifetime is determined by nonzero pixel values)
    
    Returns:
    - normalized_images: numpy array of shape (cells, 11, height, width)
    """
    num_cells, num_timepoints, height, width = images.shape
    normalized_images = np.zeros((num_cells, 11, height, width))

    for i in range(num_cells):
        # Determine lifetime (number of nonzero frames)
        lifetime = np.count_nonzero(np.sum(images[i], axis=(1, 2)))  # Sum over spatial dims to check for zeros
        #print(lifetime)
        if lifetime == 0:
            continue  # Skip cells that are all zero

        # Compute the indices corresponding to 0%, 10%, ..., 100% of the lifetime
        indices = np.round(np.linspace(0, lifetime - 1, 11)).astype(int)

        # Extract images at these indices
        normalized_images[i] = images[i, indices, :, :]

    return normalized_images


