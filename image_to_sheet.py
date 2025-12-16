#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image to Sheet Converter
Converts images containing text/tables to Excel (XLS) or CSV files.
Supports Uzbek Cyrillic (using Russian OCR model) and English.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pandas as pd
import easyocr
from PIL import Image


class ImageToSheetConverter:
    """Converts images with text/tables to spreadsheet formats."""
    
    def __init__(self):
        """Initialize OCR reader with Russian and English language support."""
        print("Initializing OCR reader (this may take a moment on first run)...")
        # Russian model handles Cyrillic script (including Uzbek Cyrillic)
        # English model handles Latin script
        self.reader = easyocr.Reader(['ru', 'en'], gpu=False)
        print("OCR reader initialized successfully!")
    
    def preprocess_image(self, image_path: str, handwritten: bool = True) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy, optimized for handwritten text.
        
        Args:
            image_path: Path to the image file
            handwritten: Whether the image contains handwritten text
            
        Returns:
            Preprocessed image as numpy array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Resize if image is too large (improves OCR speed and accuracy)
        # For handwritten text, keep higher resolution
        height, width = img.shape[:2]
        max_dimension = 2500 if handwritten else 2000
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast for handwritten text
        if handwritten:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Apply denoising with better parameters for handwritten text
        if handwritten:
            # More aggressive denoising for handwritten text
            denoised = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        else:
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Apply thresholding - different strategies for handwritten vs printed
        if handwritten:
            # For handwritten text, use adaptive threshold with larger block size
            adaptive_thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 5  # Larger block size for handwritten
            )
        else:
            adaptive_thresh = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        processed = adaptive_thresh
        
        # Apply morphological operations - lighter for handwritten text
        if handwritten:
            # Lighter morphological operations to preserve handwritten character details
            kernel = np.ones((1, 1), np.uint8)
        else:
            kernel = np.ones((2, 2), np.uint8)
        
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed
    
    def extract_text(self, image_path: str, preprocess: bool = True, min_confidence: float = 0.2, handwritten: bool = True) -> List[Tuple]:
        """
        Extract text from image using OCR, optimized for handwritten text.
        
        Args:
            image_path: Path to the image file
            preprocess: Whether to preprocess the image
            min_confidence: Minimum confidence threshold for text detection (lower for handwritten)
            handwritten: Whether the image contains handwritten text
            
        Returns:
            List of tuples containing (bbox, text, confidence)
        """
        if preprocess:
            img = self.preprocess_image(image_path, handwritten=handwritten)
            # EasyOCR expects RGB, so convert back
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        
        print("Extracting text from image...")
        if handwritten:
            print("Using handwritten text optimization...")
        
        # Use paragraph=False for better table detection
        # detail=1 to get bounding boxes
        # For handwritten text, use more lenient thresholds
        if handwritten:
            results = self.reader.readtext(
                img_rgb,
                paragraph=False,
                detail=1,
                width_ths=0.5,  # More lenient for handwritten
                height_ths=0.5,  # More lenient for handwritten
                allowlist=None,  # Allow all characters
                blocklist=''  # Don't block any characters
            )
        else:
            results = self.reader.readtext(
                img_rgb,
                paragraph=False,
                detail=1,
                width_ths=0.7,
                height_ths=0.7
            )
        
        # Filter by confidence (lower threshold for handwritten)
        filtered_results = [
            (bbox, text, conf) for (bbox, text, conf) in results 
            if conf >= min_confidence
        ]
        
        # Post-process text to correct common OCR errors
        processed_results = []
        for (bbox, text, conf) in filtered_results:
            # Clean numbers in the text
            cleaned_text = self.post_process_text(text, is_number_column=False)
            processed_results.append((bbox, cleaned_text, conf))
        
        print(f"Found {len(processed_results)} text elements (filtered from {len(results)})")
        
        return processed_results
    
    def is_cyrillic(self, text: str) -> bool:
        """
        Check if text contains Cyrillic characters.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains Cyrillic characters
        """
        cyrillic_pattern = re.compile(r'[\u0400-\u04FF]')
        return bool(cyrillic_pattern.search(text))
    
    def clean_number(self, text: str) -> str:
        """
        Clean and correct common OCR mistakes in numbers.
        Handles handwritten text OCR errors.
        
        Args:
            text: Text that may contain numbers
            
        Returns:
            Cleaned text with corrected numbers
        """
        if not text:
            return text
        
        # Common OCR mistakes for handwritten numbers
        corrections = {
            # Common misreadings
            'O': '0',  # Letter O -> Zero
            'o': '0',
            'I': '1',  # Letter I -> One (in some contexts)
            'l': '1',  # Lowercase L -> One
            'S': '5',  # Letter S -> Five
            's': '5',
            'Z': '2',  # Letter Z -> Two
            'z': '2',
            'G': '6',  # Letter G -> Six
            'g': '6',
            'B': '8',  # Letter B -> Eight
            'b': '8',
            # Cyrillic characters that might be misread as numbers
            'О': '0',  # Cyrillic O -> Zero
            'о': '0',
            'З': '3',  # Cyrillic З -> Three
            'з': '3',
        }
        
        # Extract potential numbers from text
        cleaned = text
        
        # Try to find and correct number patterns
        # Look for sequences that might be numbers
        number_pattern = re.compile(r'[0-9OoIlSsZzGgBbОоЗз]+')
        
        def correct_number(match):
            num_str = match.group()
            corrected = ''
            for char in num_str:
                if char in corrections:
                    corrected += corrections[char]
                elif char.isdigit():
                    corrected += char
                else:
                    # Keep non-number characters
                    corrected += char
            return corrected
        
        cleaned = number_pattern.sub(correct_number, cleaned)
        
        return cleaned
    
    def extract_and_validate_numbers(self, text: str) -> str:
        """
        Extract numbers from text and validate/correct them.
        
        Args:
            text: Text containing numbers
            
        Returns:
            Text with corrected numbers
        """
        if not text:
            return text
        
        # First clean the text
        cleaned = self.clean_number(text)
        
        # Extract number sequences
        # Pattern for numbers: digits, dots, commas, spaces, slashes (for dates/fractions)
        number_pattern = re.compile(r'[\d\s\.\,\/\-]+')
        
        def validate_number(match):
            num_str = match.group().strip()
            # Remove spaces and validate
            num_clean = re.sub(r'\s+', '', num_str)
            
            # Check if it's a valid number format
            if re.match(r'^\d+([\.\,]\d+)?([\/\-]\d+)?$', num_clean):
                return num_clean
            return num_str
        
        result = number_pattern.sub(validate_number, cleaned)
        
        return result
    
    def post_process_text(self, text: str, is_number_column: bool = False) -> str:
        """
        Post-process OCR text to correct common errors.
        
        Args:
            text: Raw OCR text
            is_number_column: Whether this is likely a number column
            
        Returns:
            Post-processed text
        """
        if not text:
            return text
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # If it's a number column, apply number-specific cleaning
        if is_number_column:
            text = self.extract_and_validate_numbers(text)
        
        # Common character corrections for Cyrillic
        cyrillic_corrections = {
            # Common OCR mistakes
            '0': 'О',  # Zero might be Cyrillic O in some contexts (but we want to keep numbers)
            # Add more as needed
        }
        
        return text.strip()
    
    def organize_text_into_table(self, ocr_results: List[Tuple]) -> List[List[str]]:
        """
        Organize OCR results into a table structure based on Y-coordinates.
        Uses adaptive thresholding and column clustering for better accuracy.
        
        Args:
            ocr_results: List of OCR results (bbox, text, confidence)
            
        Returns:
            List of rows, where each row is a list of cell values
        """
        if not ocr_results:
            return []
        
        # Extract text with bounding box information
        text_items = []
        y_coordinates = []
        
        for (bbox, text, confidence) in ocr_results:
            # Calculate bounding box properties
            y_coords = [point[1] for point in bbox]
            x_coords = [point[0] for point in bbox]
            
            top_y = min(y_coords)
            bottom_y = max(y_coords)
            left_x = min(x_coords)
            right_x = max(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            center_x = sum(x_coords) / len(x_coords)
            height = bottom_y - top_y
            
            text_items.append({
                'text': text.strip(),
                'y': center_y,
                'top_y': top_y,
                'bottom_y': bottom_y,
                'x': center_x,
                'left_x': left_x,
                'right_x': right_x,
                'height': height,
                'confidence': confidence
            })
            y_coordinates.append(center_y)
        
        # Calculate adaptive Y threshold based on median text height
        heights = [item['height'] for item in text_items]
        median_height = np.median(heights) if heights else 20
        y_threshold = max(median_height * 0.5, 15)  # Adaptive threshold
        
        # Sort by Y coordinate first
        text_items.sort(key=lambda x: x['y'])
        
        # Group items into rows using clustering approach
        rows = []
        current_row = []
        
        for item in text_items:
            if not current_row:
                current_row.append(item)
            else:
                # Check if this item is on the same row
                # Use both center Y and top/bottom overlap
                avg_y = sum(i['y'] for i in current_row) / len(current_row)
                
                # Check Y distance
                y_distance = abs(item['y'] - avg_y)
                
                # Also check if bounding boxes overlap vertically
                overlaps = any(
                    not (item['bottom_y'] < i['top_y'] or item['top_y'] > i['bottom_y'])
                    for i in current_row
                )
                
                if y_distance <= y_threshold or overlaps:
                    current_row.append(item)
                else:
                    # Finalize current row
                    current_row.sort(key=lambda x: x['x'])
                    rows.append([i['text'] for i in current_row])
                    current_row = [item]
        
        # Don't forget the last row
        if current_row:
            current_row.sort(key=lambda x: x['x'])
            rows.append([i['text'] for i in current_row])
        
        # Now align columns using clustering
        # Find common X positions (columns) across rows
        if rows:
            rows = self._align_columns(rows, text_items)
        
        return rows
    
    def _align_columns(self, rows: List[List[str]], text_items: List[dict]) -> List[List[str]]:
        """
        Align columns by clustering X positions across all rows.
        
        Args:
            rows: List of rows (each row is a list of cell texts)
            text_items: Original text items with position info
            
        Returns:
            Aligned rows with consistent column structure
        """
        if not rows:
            return rows
        
        # Collect all X positions
        x_positions = []
        text_to_x = {}
        
        for item in text_items:
            x_positions.append(item['x'])
            text_to_x[item['text']] = item['x']
        
        # Cluster X positions to find column centers
        x_positions = sorted(set(x_positions))
        
        # Group nearby X positions (within threshold)
        column_centers = []
        x_threshold = np.median([item['height'] for item in text_items]) * 0.8 if text_items else 30
        
        for x in x_positions:
            # Check if this X is close to any existing column center
            assigned = False
            for i, center in enumerate(column_centers):
                if abs(x - center) <= x_threshold:
                    # Update center to average
                    column_centers[i] = (column_centers[i] + x) / 2
                    assigned = True
                    break
            
            if not assigned:
                column_centers.append(x)
        
        column_centers = sorted(column_centers)
        
        # Rebuild rows with aligned columns
        aligned_rows = []
        for row in rows:
            aligned_row = [''] * len(column_centers)
            
            for cell_text in row:
                if cell_text in text_to_x:
                    cell_x = text_to_x[cell_text]
                    # Find closest column
                    closest_col = min(
                        range(len(column_centers)),
                        key=lambda i: abs(column_centers[i] - cell_x)
                    )
                    # If multiple items map to same column, concatenate
                    if aligned_row[closest_col]:
                        aligned_row[closest_col] += ' ' + cell_text
                    else:
                        aligned_row[closest_col] = cell_text
            
            aligned_rows.append(aligned_row)
        
        return aligned_rows
    
    def convert_to_dataframe(self, table_data: List[List[str]]) -> pd.DataFrame:
        """
        Convert table data to pandas DataFrame with number cleaning.
        
        Args:
            table_data: List of rows (each row is a list of cells)
            
        Returns:
            pandas DataFrame
        """
        if not table_data:
            return pd.DataFrame()
        
        # Find maximum number of columns
        max_cols = max(len(row) for row in table_data) if table_data else 0
        
        # Pad rows to have the same number of columns
        padded_data = []
        for row in table_data:
            # Post-process each cell, especially numbers
            processed_row = []
            for cell in row:
                # Check if cell looks like a number
                is_number = bool(re.search(r'[\dOoIlSsZzGgBb]', cell))
                cleaned_cell = self.post_process_text(cell, is_number_column=is_number)
                processed_row.append(cleaned_cell)
            
            padded_row = processed_row + [''] * (max_cols - len(processed_row))
            padded_data.append(padded_row)
        
        # Create DataFrame
        # Use first row as header if it looks like a header
        if len(padded_data) > 1:
            # Check if first row might be a header (all non-empty, shorter text)
            first_row = padded_data[0]
            is_header = all(cell.strip() for cell in first_row[:3]) and \
                       all(len(cell) < 50 for cell in first_row)
            
            if is_header:
                df = pd.DataFrame(padded_data[1:], columns=first_row)
            else:
                df = pd.DataFrame(padded_data)
        else:
            df = pd.DataFrame(padded_data)
        
        # Try to convert number columns to numeric type
        for col in df.columns:
            # Check if column contains mostly numbers
            numeric_count = 0
            total_count = 0
            for val in df[col]:
                if pd.notna(val) and str(val).strip():
                    total_count += 1
                    # Check if it's a number (after cleaning)
                    cleaned = self.extract_and_validate_numbers(str(val))
                    if re.match(r'^[\d\.\,\/\-]+$', cleaned.replace(' ', '')):
                        numeric_count += 1
            
            # If more than 50% are numbers, try to convert
            if total_count > 0 and numeric_count / total_count > 0.5:
                # Try to convert to numeric
                for idx in df.index:
                    val = df.at[idx, col]
                    if pd.notna(val) and str(val).strip():
                        cleaned = self.extract_and_validate_numbers(str(val))
                        # Remove spaces and try to parse
                        cleaned = cleaned.replace(' ', '').replace(',', '.')
                        try:
                            # Try to convert to float or int
                            if '.' in cleaned or '/' in cleaned:
                                num_val = float(cleaned.split('/')[0]) if '/' in cleaned else float(cleaned)
                            else:
                                num_val = int(cleaned)
                            df.at[idx, col] = num_val
                        except (ValueError, AttributeError):
                            pass  # Keep original if conversion fails
        
        return df
    
    def convert_image_to_sheet(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        format: str = 'xlsx',
        preprocess: bool = True,
        min_confidence: float = 0.2,
        handwritten: bool = True
    ) -> str:
        """
        Convert image to spreadsheet file.
        
        Args:
            image_path: Path to input image
            output_path: Path to output file (optional)
            format: Output format ('xlsx' or 'csv')
            preprocess: Whether to preprocess image
            
        Returns:
            Path to the created output file
        """
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Determine output path
        if output_path is None:
            base_name = Path(image_path).stem
            if format.lower() == 'xlsx':
                output_path = f"{base_name}.xlsx"
            else:
                output_path = f"{base_name}.csv"
        
        # Extract text from image
        ocr_results = self.extract_text(
            image_path, 
            preprocess=preprocess, 
            min_confidence=min_confidence,
            handwritten=handwritten
        )
        
        if not ocr_results:
            print("Warning: No text detected in the image!")
            # Create empty file
            df = pd.DataFrame()
        else:
            # Print detected text for debugging (limit output)
            print("\nDetected text (showing first 20 items):")
            for i, (bbox, text, confidence) in enumerate(ocr_results[:20]):
                lang = "Cyrillic" if self.is_cyrillic(text) else "Latin"
                print(f"  [{lang}] {text[:50]}... (confidence: {confidence:.2f})")
            if len(ocr_results) > 20:
                print(f"  ... and {len(ocr_results) - 20} more items")
            
            # Organize into table structure
            print("\nOrganizing text into table structure...")
            table_data = self.organize_text_into_table(ocr_results)
            print(f"Created {len(table_data)} rows")
            
            # Convert to DataFrame
            df = self.convert_to_dataframe(table_data)
        
        # Save to file
        print(f"\nSaving to {output_path}...")
        if format.lower() == 'xlsx':
            df.to_excel(output_path, index=False, engine='openpyxl')
        else:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"Successfully created {output_path}")
        return output_path


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Convert images with text/tables to Excel (XLS) or CSV files. '
                    'Supports Uzbek Cyrillic and English text.'
    )
    parser.add_argument(
        'image',
        type=str,
        help='Path to the input image file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to the output file (default: same as input with .xlsx or .csv extension)'
    )
    parser.add_argument(
        '-f', '--format',
        type=str,
        choices=['xlsx', 'csv'],
        default='xlsx',
        help='Output format: xlsx or csv (default: xlsx)'
    )
    parser.add_argument(
        '--no-preprocess',
        action='store_true',
        help='Disable image preprocessing (may be faster but less accurate)'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.2,
        help='Minimum confidence threshold for OCR (0.0-1.0, default: 0.2 for handwritten)'
    )
    parser.add_argument(
        '--printed',
        action='store_true',
        help='Use printed text optimization instead of handwritten (may improve speed)'
    )
    
    args = parser.parse_args()
    
    try:
        converter = ImageToSheetConverter()
        converter = ImageToSheetConverter()
        converter.convert_image_to_sheet(
            image_path=args.image,
            output_path=args.output,
            format=args.format,
            preprocess=not args.no_preprocess,
            min_confidence=args.min_confidence,
            handwritten=not args.printed
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

