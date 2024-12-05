import cv2
import numpy as np
import os
import time

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
duration = 60  # Duration in seconds for the video
frame_count = fps * duration

# Ensure output directory exists
output_dir = 'videos'
os.makedirs(output_dir, exist_ok=True)

# Generate video file path with epoch time
epoch_time = int(time.time())
output_path = os.path.join(output_dir, f"{epoch_time}_rotating_tiles.mp4")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Create initial tile grid and store their rotations
num_tiles_x = frame_width // tile_width
num_tiles_y = frame_height // tile_height
tile_rotations = np.zeros((num_tiles_y, num_tiles_x), dtype=float)  # Use float for smooth transitions

# Generate video frames
for frame_idx in range(frame_count):
    # Create a blank frame
    output_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Check if it's time to animate (every 5 seconds)
    is_animation_time = (frame_idx // fps) % 5 == 0
    animation_frame = frame_idx % fps if is_animation_time else 0

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate position in the frame
            y_start = y * tile_height
            x_start = x * tile_width
            y_end = min(y_start + tile_height, frame_height)
            x_end = min(x_start + tile_width, frame_width)

            # Extract portion of the tile that fits in the frame
            tile_to_place = tile[:y_end - y_start, :x_end - x_start]

            # Calculate rotation
            if is_animation_time and animation_frame < 30:
                start_angle = tile_rotations[y, x]
                end_angle = start_angle - 90  # Incrementally subtract 90 degrees
                current_angle = start_angle + (end_angle - start_angle) * (animation_frame / 30.0)
                rotated_tile = rotate_tile(tile_to_place, current_angle)
            else:
                rotated_tile = rotate_tile(tile_to_place, tile_rotations[y, x])

            # Update rotation state after animation
            if is_animation_time and animation_frame == 29:
                tile_rotations[y, x] = (tile_rotations[y, x] - 90) % 360

            # Place the tile into the frame
            output_frame[y_start:y_end, x_start:x_end] = rotated_tile

    # Show the framebuffer
    cv2.imshow('Framebuffer', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the frame to the video
    video_writer.write(output_frame)

# Release the video writer and close display window
video_writer.release()
cv2.destroyAllWindows()

print(f"Video saved as {output_path}")
