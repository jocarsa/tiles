import cv2
import numpy as np

# Load the 128x128 tile
tile_path = 'tile.png'  # Replace with your tile image path
tile = cv2.imread(tile_path)

# Get tile dimensions
tile_height, tile_width = tile.shape[:2]

# Create an empty frame with the target size (1920x1080)
frame_width, frame_height = 1920, 1080
frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Fill the frame with the tile
for y in range(0, frame_height, tile_height):
    for x in range(0, frame_width, tile_width):
        # Calculate the region of the frame to fill
        y_end = min(y + tile_height, frame_height)
        x_end = min(x + tile_width, frame_width)
        
        # Calculate the corresponding region of the tile
        tile_y_end = y_end - y
        tile_x_end = x_end - x
        
        # Copy the appropriate portion of the tile to the frame
        frame[y:y_end, x:x_end] = tile[:tile_y_end, :tile_x_end]

# Save or display the resulting frame
output_path = 'tiled_frame.png'
cv2.imwrite(output_path, frame)
print(f"Tiled frame saved as {output_path}")

# Optionally display the result
cv2.imshow('Tiled Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
