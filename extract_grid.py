#!/usr/bin/env python3
"""
Script to extract a large TIF file into a 2D grid of 640x640 pixel tiles.
Tiles are saved as JPG files in a "data" folder, named as x_X_y_Y.jpg
where X and Y are the grid coordinates (starting from bottom-left).

Edge tiles are padded with zero values if they don't have enough pixels.
"""

import sys
import os
import numpy as np
from pathlib import Path

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    print("Error: rasterio is required for this script. Please install it with: pip3 install rasterio")
    sys.exit(1)

from PIL import Image


def extract_grid_tiles(tif_path, output_dir="data", tile_size=640):
    """
    Extract the entire TIF image into a grid of tiles.
    
    Args:
        tif_path: Path to the input TIF file
        output_dir: Directory to save the tiles (default: "data")
        tile_size: Size of each tile in pixels (default: 640)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {output_path.absolute()}")
    
    try:
        print(f"Opening TIF file: {tif_path}")
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            num_bands = src.count
            print(f"Image size: {width} x {height} pixels, {num_bands} band(s)")
            
            # Calculate grid dimensions
            num_tiles_x = (width + tile_size - 1) // tile_size  # Ceiling division
            num_tiles_y = (height + tile_size - 1) // tile_size  # Ceiling division
            print(f"Grid size: {num_tiles_x} x {num_tiles_y} tiles")
            print(f"Total tiles to extract: {num_tiles_x * num_tiles_y}")
            print()
            
            # Process each tile
            # Note: In image coordinates, (0,0) is top-left, y increases downward
            # We want grid coordinates where (0,0) is bottom-left, y increases upward
            tile_count = 0
            for grid_y in range(num_tiles_y):
                # Calculate pixel Y position (starting from bottom)
                # grid_y=0 is bottom row, grid_y=num_tiles_y-1 is top row
                # Bottom row starts at: height - tile_size (or 0 if height < tile_size)
                pixel_y_bottom = height - (grid_y + 1) * tile_size
                
                for grid_x in range(num_tiles_x):
                    # Calculate pixel X position (starting from left)
                    pixel_x = grid_x * tile_size
                    
                    # Calculate actual window size (may be smaller at edges)
                    window_width = min(tile_size, width - pixel_x)
                    
                    # For Y coordinate: handle bottom edge padding
                    if pixel_y_bottom < 0:
                        # This tile extends beyond the bottom edge
                        # We'll read from y=0 and pad the top with zeros
                        pixel_y = 0
                        # Read only the part that exists in the image
                        window_height = tile_size + pixel_y_bottom  # pixel_y_bottom is negative
                    else:
                        pixel_y = pixel_y_bottom
                        window_height = min(tile_size, height - pixel_y)
                    
                    # Read the window
                    window = Window(pixel_x, pixel_y, window_width, window_height)
                    
                    # Read the data
                    if num_bands == 1:
                        # Grayscale - convert to RGB
                        data = src.read(1, window=window)
                        rgb_data = np.stack([data, data, data], axis=2)
                    elif num_bands == 3:
                        # RGB
                        rgb_data = np.dstack([src.read(i, window=window) for i in range(1, 4)])
                    elif num_bands >= 4:
                        # RGBA or more - take first 3 bands for RGB
                        rgb_data = np.dstack([src.read(i, window=window) for i in range(1, 4)])
                    else:
                        print(f"Error: Unsupported number of bands: {num_bands}")
                        return False
                    
                    # Convert to uint8 if needed
                    if rgb_data.dtype != np.uint8:
                        # Normalize to 0-255 range
                        if rgb_data.max() > 255:
                            rgb_data = (rgb_data / rgb_data.max() * 255).astype(np.uint8)
                        else:
                            rgb_data = rgb_data.astype(np.uint8)
                    
                    # Pad if necessary (edge tiles)
                    if rgb_data.shape[0] < tile_size or rgb_data.shape[1] < tile_size:
                        # Create a zero-padded array
                        padded_data = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        # Place the actual data in the appropriate position
                        # For bottom-left origin:
                        # - Y padding: pad at top (data goes to bottom of tile)
                        # - X padding: pad at right (data goes to left of tile)
                        data_height, data_width = rgb_data.shape[0], rgb_data.shape[1]
                        y_start = tile_size - data_height  # Place data at bottom
                        x_start = 0  # Place data at left
                        padded_data[y_start:y_start+data_height, x_start:x_start+data_width] = rgb_data
                        rgb_data = padded_data
                    
                    # Save as JPG
                    img = Image.fromarray(rgb_data, 'RGB')
                    filename = f"x_{grid_x}_y_{grid_y}.jpg"
                    filepath = output_path / filename
                    img.save(filepath, 'JPEG', quality=95)
                    
                    tile_count += 1
                    if tile_count % 10 == 0:
                        print(f"Processed {tile_count}/{num_tiles_x * num_tiles_y} tiles...", end='\r')
            
            print(f"\nSuccessfully extracted {tile_count} tiles to {output_path.absolute()}")
            return True
            
    except FileNotFoundError:
        print(f"Error: File not found: {tif_path}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python3 extract_grid.py <input.tif> [output_dir] [tile_size]")
        print("\nExample:")
        print("  python3 extract_grid.py west1_zoom_21.tif")
        print("  python3 extract_grid.py west1_zoom_21.tif data 640")
        print("\nDefault values:")
        print("  output_dir: data")
        print("  tile_size: 640 pixels")
        sys.exit(1)
    
    tif_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data"
    tile_size = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    
    # Check if input file exists
    if not os.path.exists(tif_path):
        print(f"Error: Input file does not exist: {tif_path}")
        sys.exit(1)
    
    # Extract the grid
    success = extract_grid_tiles(tif_path, output_dir, tile_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

