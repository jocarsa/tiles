import cv2
import numpy as np

def rotate_tile(tile, angle):
    """Rotate a tile by a given angle around its center."""
    h, w = tile.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_tile = cv2.warpAffine(tile, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated_tile

# Load the tile
tile_path = 'tile.png'  # Replace with your tile image path
tile = cv2.imread(tile_path)
tile_height, tile_width = tile.shape[:2]

# Frame dimensions and video settings
frame_width, frame_height = 1920, 1080
fps = 30
duration = 10  # Duration in seconds for the video
frame_count = fps * duration

# Video writer
output_path = 'rotating_tiles.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Generate video frames
for frame_idx in range(frame_count):
    # Create a blank frame
    output_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Check if it's time to animate (every 5 seconds)
    if (frame_idx // fps) % 5 == 0 and (frame_idx % fps) < 30:
        # Calculate animation progress (0 to 1)
        progress = (frame_idx % fps) / 30.0
        angle = -90 * progress  # Rotate left by 90 degrees
    else:
        angle = 0  # No rotation for idle frames

    # Apply rotation to each tile
    for y in range(0, frame_height, tile_height):
        for x in range(0, frame_width, tile_width):
            # Calculate the region of the frame to fill
            y_end = min(y + tile_height, frame_height)
            x_end = min(x + tile_width, frame_width)
            
            # Calculate the corresponding region of the tile
            tile_y_end = y_end - y
            tile_x_end = x_end - x
            
            # Extract the portion of the tile that fits the frame
            tile_to_rotate = tile[:tile_y_end, :tile_x_end]
            
            # Rotate the tile if necessary
            if angle != 0:
                rotated_tile = rotate_tile(tile_to_rotate, angle)
            else:
                rotated_tile = tile_to_rotate
            
            # Place the rotated tile into the frame
            output_frame[y:y_end, x:x_end] = rotated_tile

    # Write the frame to the video
    video_writer.write(output_frame)

# Release the video writer
video_writer.release()
print(f"Video saved as {output_path}")
