import numpy as np
import pandas as pd
import glob
import tifffile as tiff


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


# def process_all_replicates(tracks_br1, tracks_br2_ tracks_br3, sub_size):
#     "SHOULD I MAKE A FUNCTION THAT PROCESSES ALL TRACKS AT ONCE?"
#     return images, segmentations, fret, y_train

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

    # Taking only the features we want
    desired_features = [
    "unique_track_id", "xpos", "ypos", "tpos", "cell_av_FRET_norm",
    "INT_av", "Area", "Aspect_Ratio", "mitosis_time", "fate"]

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

