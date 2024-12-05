import cv2
import numpy as np
import os
import time
import random

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
output_path = os.path.join(output_dir, f"{epoch_time}_random_wait_rotating_tiles.mp4")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Create initial tile grid and store their rotations and timers
num_tiles_x = frame_width // tile_width
num_tiles_y = frame_height // tile_height
tile_rotations = np.zeros((num_tiles_y, num_tiles_x), dtype=float)  # Store current rotation for each tile
tile_wait_times = np.random.randint(2, 6, size=(num_tiles_y, num_tiles_x))  # Random wait times (2 to 5 seconds)
tile_time_counters = np.zeros((num_tiles_y, num_tiles_x), dtype=int)  # Frame counters for each tile

# Generate video frames
for frame_idx in range(frame_count):
    # Create a blank frame
    output_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    for y in range(num_tiles_y):
        for x in range(num_tiles_x):
            # Calculate position in the frame
            y_start = y * tile_height
            x_start = x * tile_width
            y_end = min(y_start + tile_height, frame_height)
            x_end = min(x_start + tile_width, frame_width)

            # Extract portion of the tile that fits in the frame
            tile_to_place = tile[:y_end - y_start, :x_end - x_start]

            # Check if the tile is animating or waiting
            if tile_time_counters[y, x] < fps * tile_wait_times[y, x]:
                # Tile is waiting, no animation
                rotated_tile = rotate_tile(tile_to_place, tile_rotations[y, x])
                tile_time_counters[y, x] += 1
            elif tile_time_counters[y, x] < fps * tile_wait_times[y, x] + 30:
                # Tile is animating
                animation_frame = tile_time_counters[y, x] - fps * tile_wait_times[y, x]
                start_angle = tile_rotations[y, x]
                end_angle = start_angle - 90
                progress = animation_frame / 30.0
                current_angle = start_angle + (end_angle - start_angle) * progress
                rotated_tile = rotate_tile(tile_to_place, current_angle)
                tile_time_counters[y, x] += 1

                # Update rotation state after animation completes
                if animation_frame == 29:
                    tile_rotations[y, x] = (tile_rotations[y, x] - 90) % 360
                    tile_time_counters[y, x] = 0  # Reset counter to wait again
            else:
                # Reset counter after animation
                tile_time_counters[y, x] = 0

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
