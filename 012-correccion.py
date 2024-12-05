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
    rotated_tile = cv2.warpAffine(tile, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated_tile

# Load the tile
tile_path = 'tile.png'  # Replace with your tile image path
original_tile = cv2.imread(tile_path)

# Frame dimensions and video settings
frame_width, frame_height = 1080, 1080
fps = 30
duration = 100  # Duration in seconds for the video
frame_count = fps * duration

# Ensure output directory exists
output_dir = 'videos'
os.makedirs(output_dir, exist_ok=True)

# Generate video file path with epoch time
epoch_time = int(time.time())
output_path = os.path.join(output_dir, f"{epoch_time}_independent_tiles.mp4")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Generate a uniform random size for all tiles
tile_size = random.randint(8, 32)  # Random size for all tiles
resized_tile = cv2.resize(original_tile, (tile_size, tile_size))

# Calculate the number of tiles in the grid
num_tiles_x = (frame_width // tile_size)
num_tiles_y = (frame_height // tile_size)

num_tiles_x += 1
num_tiles_y += 1

# Create a grid of positions
positions = [
    (y * tile_size, x * tile_size)
    for y in range(num_tiles_y)
    for x in range(num_tiles_x)
]

# Set fixed wait time and animation time
wait_time_frames = fps  # 1 second
animation_time_frames = int(0.5 * fps)  # 0.5 seconds

# Initialize rotation states
tile_rotations = [0] * len(positions)  # Store current rotation for each tile
tile_time_counters = [random.randint(0, wait_time_frames - 1) for _ in positions]  # Start counters randomly
rotation_directions = [random.choice([-90, 90]) for _ in positions]  # Random initial direction for each tile

# Generate video frames
for frame_idx in range(frame_count):
    # Create a blank white frame
    output_frame = np.full((frame_height, frame_width, 3), 255, dtype=np.uint8)

    for idx, (y_start, x_start) in enumerate(positions):
        y_end = min(y_start + tile_size, frame_height)
        x_end = min(x_start + tile_size, frame_width)

        # Extract the visible portion of the tile to fit within the frame
        visible_tile = resized_tile[:y_end - y_start, :x_end - x_start]

        # Correctly center the tile for rotation
        visible_h, visible_w = visible_tile.shape[:2]
        center_tile = (visible_w // 2, visible_h // 2)

        # Determine current angle
        if tile_time_counters[idx] < wait_time_frames:
            # Tile is waiting, no animation
            current_angle = tile_rotations[idx]
            tile_time_counters[idx] += 1
        elif tile_time_counters[idx] < wait_time_frames + animation_time_frames:
            # Tile is animating
            animation_frame = tile_time_counters[idx] - wait_time_frames
            start_angle = tile_rotations[idx]
            end_angle = start_angle + rotation_directions[idx]
            progress = animation_frame / animation_time_frames
            current_angle = start_angle + (end_angle - start_angle) * progress
            tile_time_counters[idx] += 1

            # Update rotation state after animation completes
            if animation_frame == animation_time_frames - 1:
                tile_rotations[idx] = end_angle % 360
                tile_time_counters[idx] = 0  # Reset counter to wait again
                # Randomly choose a new direction for the next rotation
                rotation_directions[idx] = random.choice([-90, 90])
        else:
            # Reset counter after animation
            tile_time_counters[idx] = 0

        # Rotate the original tile by the current angle
        rotation_matrix = cv2.getRotationMatrix2D(center_tile, current_angle, 1.0)
        rotated_tile = cv2.warpAffine(visible_tile, rotation_matrix, (visible_w, visible_h),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Place the tile into the frame
        output_frame[y_start:y_end, x_start:x_end] = rotated_tile[:y_end - y_start, :x_end - x_start]

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