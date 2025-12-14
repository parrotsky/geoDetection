#!/usr/bin/env python3
"""
Script to extract a 640x640 pixel tile from a large TIF file and save it as JPG.

Based on the log file, the TIF file is:
- Size: 75520 x 56832 pixels
- Format: GeoTIFF with JPEG compression
- Coordinate system: WGS 84

This script uses rasterio for efficient reading of large geospatial TIF files.
"""

import sys
import os
import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not available, falling back to PIL with increased limits")
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None  # Disable decompression bomb protection


def extract_tile_rasterio(tif_path, output_path, x_offset=0, y_offset=0, tile_size=640):
    """
    Extract a tile using rasterio (efficient for large geospatial files).
    """
    try:
        print(f"Opening TIF file with rasterio: {tif_path}")
        with rasterio.open(tif_path) as src:
            width = src.width
            height = src.height
            num_bands = src.count
            print(f"Image size: {width} x {height} pixels, {num_bands} band(s)")
            
            # Validate offsets
            if x_offset < 0 or y_offset < 0:
                print("Error: Offsets must be non-negative")
                return False
            
            if x_offset + tile_size > width or y_offset + tile_size > height:
                print(f"Error: Tile would extend beyond image boundaries")
                print(f"  Image: {width}x{height}, Requested: {x_offset}+{tile_size} x {y_offset}+{tile_size}")
                return False
            
            # Read only the window we need (memory efficient)
            print(f"Extracting tile at position ({x_offset}, {y_offset}) with size {tile_size}x{tile_size}")
            window = Window(x_offset, y_offset, tile_size, tile_size)
            
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
            
            # Save using PIL
            from PIL import Image
            img = Image.fromarray(rgb_data, 'RGB')
            print(f"Saving to: {output_path}")
            img.save(output_path, 'JPEG', quality=95)
            print(f"Successfully saved {tile_size}x{tile_size} tile to {output_path}")
            return True
            
    except Exception as e:
        print(f"Error with rasterio: {str(e)}")
        return False


def extract_tile_pil(tif_path, output_path, x_offset=0, y_offset=0, tile_size=640):
    """
    Extract a tile using PIL (fallback method).
    """
    try:
        print(f"Opening TIF file with PIL: {tif_path}")
        with Image.open(tif_path) as img:
            width, height = img.size
            print(f"Image size: {width} x {height} pixels")
            
            # Validate offsets
            if x_offset < 0 or y_offset < 0:
                print("Error: Offsets must be non-negative")
                return False
            
            if x_offset + tile_size > width or y_offset + tile_size > height:
                print(f"Error: Tile would extend beyond image boundaries")
                print(f"  Image: {width}x{height}, Requested: {x_offset}+{tile_size} x {y_offset}+{tile_size}")
                return False
            
            # Extract the tile (left, top, right, bottom)
            print(f"Extracting tile at position ({x_offset}, {y_offset}) with size {tile_size}x{tile_size}")
            tile = img.crop((x_offset, y_offset, x_offset + tile_size, y_offset + tile_size))
            
            # Convert to RGB if necessary (JPG doesn't support transparency)
            if tile.mode in ('RGBA', 'LA', 'P'):
                # Create a white background for transparent images
                rgb_tile = Image.new('RGB', tile.size, (255, 255, 255))
                if tile.mode == 'P':
                    tile = tile.convert('RGBA')
                rgb_tile.paste(tile, mask=tile.split()[-1] if tile.mode in ('RGBA', 'LA') else None)
                tile = rgb_tile
            elif tile.mode != 'RGB':
                tile = tile.convert('RGB')
            
            # Save as JPG with good quality
            print(f"Saving to: {output_path}")
            tile.save(output_path, 'JPEG', quality=95)
            print(f"Successfully saved {tile_size}x{tile_size} tile to {output_path}")
            return True
            
    except FileNotFoundError:
        print(f"Error: File not found: {tif_path}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def extract_tile(tif_path, output_path, x_offset=0, y_offset=0, tile_size=640):
    """
    Extract a tile from a TIF file and save it as JPG.
    Uses rasterio if available, otherwise falls back to PIL.
    """
    if HAS_RASTERIO:
        return extract_tile_rasterio(tif_path, output_path, x_offset, y_offset, tile_size)
    else:
        return extract_tile_pil(tif_path, output_path, x_offset, y_offset, tile_size)


def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 3:
        print("Usage: python3 extract_tile.py <input.tif> <output.jpg> [x_offset] [y_offset] [tile_size]")
        print("\nExample:")
        print("  python3 extract_tile.py west1_zoom_21.tif tile.jpg")
        print("  python3 extract_tile.py west1_zoom_21.tif tile.jpg 0 0 640")
        print("  python3 extract_tile.py west1_zoom_21.tif tile.jpg 1000 2000 640")
        print("\nDefault values:")
        print("  x_offset: 0 (top-left corner)")
        print("  y_offset: 0 (top-left corner)")
        print("  tile_size: 640 pixels")
        sys.exit(1)
    
    tif_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Parse optional arguments
    x_offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    y_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tile_size = int(sys.argv[5]) if len(sys.argv) > 5 else 640
    
    # Check if input file exists
    if not os.path.exists(tif_path):
        print(f"Error: Input file does not exist: {tif_path}")
        sys.exit(1)
    
    # Extract the tile
    success = extract_tile(tif_path, output_path, x_offset, y_offset, tile_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

