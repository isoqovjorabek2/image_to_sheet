# Image to Sheet Converter

Convert images containing text and tables into Excel (XLS) or CSV spreadsheets. This application supports:

- **Uzbek Cyrillic text** (using Russian OCR model)
- **English text** (using English OCR model)
- Automatic language detection and processing
- Table structure detection and organization

## Features

- 📸 Extract text from images using OCR
- ✍️ **Optimized for handwritten text** with specialized preprocessing
- 🔢 **Accurate number recognition** with post-processing and validation
- 🔤 Support for Cyrillic (Uzbek) and Latin (English) scripts
- 📊 Automatic table structure detection
- 💾 Export to Excel (.xlsx) or CSV formats
- 🖼️ Advanced image preprocessing for better OCR accuracy
- 🎯 Confidence scoring and error correction for detected text
- 🔧 Automatic number cleaning and character correction

## Installation

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

**Note:** On first run, EasyOCR will download the language models (Russian and English). This may take a few minutes and requires an internet connection.

## Usage

### Command Line Interface

Basic usage:

```bash
python image_to_sheet.py path/to/image.jpg
```

This will create an Excel file (`image.xlsx`) in the same directory as the input image.

### Options

- **Specify output file:**
  ```bash
  python image_to_sheet.py image.jpg -o output.xlsx
  ```

- **Export as CSV:**
  ```bash
  python image_to_sheet.py image.jpg -f csv
  ```

- **Disable image preprocessing (faster but less accurate):**
  ```bash
  python image_to_sheet.py image.jpg --no-preprocess
  ```

- **Adjust confidence threshold (lower for handwritten text):**
  ```bash
  python image_to_sheet.py image.jpg --min-confidence 0.15
  ```

- **Use printed text optimization (if text is printed, not handwritten):**
  ```bash
  python image_to_sheet.py image.jpg --printed
  ```

### Full Command Syntax

```bash
python image_to_sheet.py <image_path> [OPTIONS]

Options:
  -o, --output PATH        Path to output file (default: same as input with .xlsx extension)
  -f, --format FORMAT      Output format: xlsx or csv (default: xlsx)
  --no-preprocess          Disable image preprocessing
  --min-confidence FLOAT   Minimum confidence threshold (0.0-1.0, default: 0.2 for handwritten)
  --printed                Use printed text optimization instead of handwritten
  -h, --help              Show help message
```

## Examples

### Example 1: Convert image to Excel

```bash
python image_to_sheet.py document.png
# Creates: document.xlsx
```

### Example 2: Convert image to CSV with custom output name

```bash
python image_to_sheet.py table.jpg -o my_table.csv -f csv
```

### Example 3: Process image without preprocessing

```bash
python image_to_sheet.py image.png --no-preprocess -o result.xlsx
```

## Opening XLSX Files

After creating an XLSX file, you can open it in several ways:

### Option 1: Double-click (Windows/Mac)
Simply double-click the `.xlsx` file and it will open with your default spreadsheet application (Excel, LibreOffice Calc, etc.).

### Option 2: Use the helper script
```bash
# Open with default application
python open_xlsx.py image.xlsx

# Or display contents in terminal
python open_xlsx.py image.xlsx -d
```

### Option 3: Manual opening
- **Windows**: Right-click → Open with → Microsoft Excel (or LibreOffice Calc)
- **Mac**: Right-click → Open With → Excel/Numbers
- **Linux**: Right-click → Open With → LibreOffice Calc

### Option 4: Online
Upload the file to [Google Sheets](https://sheets.google.com) or [Office Online](https://www.office.com) to view it in your browser.

## How It Works

1. **Image Loading**: The application loads the input image
2. **Preprocessing** (optional): Applies denoising, grayscale conversion, and thresholding to improve OCR accuracy
3. **OCR Processing**: Uses EasyOCR with Russian and English language models to extract text
4. **Language Detection**: Automatically detects Cyrillic vs Latin characters
5. **Table Organization**: Groups detected text into rows and columns based on spatial coordinates
6. **Export**: Saves the organized data to Excel or CSV format

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)
- And other formats supported by OpenCV

## Language Support

- **Cyrillic Script**: Detected using Russian OCR model
  - Uzbek (Cyrillic)
  - Russian
  - Other Cyrillic-based languages
- **Latin Script**: Detected using English OCR model
  - English
  - Other Latin-based languages

## Requirements

- Python 3.7 or higher
- See `requirements.txt` for package dependencies

## Handwritten Text Support

This application is **optimized for handwritten text**, especially numbers:

- **Specialized preprocessing**: Enhanced contrast and denoising for handwritten characters
- **Number accuracy**: Post-processing corrects common OCR mistakes (O→0, I→1, S→5, etc.)
- **Lower confidence threshold**: Default 0.2 (instead of 0.3) to capture more handwritten text
- **Character correction**: Automatically fixes common misreadings in numbers
- **Number validation**: Detects and validates number patterns, converts to numeric types in Excel

### Tips for Better Handwritten Text Recognition

1. **Image Quality**: Use high-resolution images (at least 300 DPI)
2. **Contrast**: Ensure good contrast between text and background
3. **Lighting**: Avoid shadows and glare
4. **Confidence Threshold**: Lower the threshold (e.g., `--min-confidence 0.15`) if text is not detected
5. **Preprocessing**: Keep preprocessing enabled (default) for best results

## Troubleshooting

### OCR Not Detecting Text

- Ensure the image has good contrast and resolution
- Try enabling preprocessing (default behavior)
- Check that text is clearly visible and not too small
- For handwritten text, lower the confidence threshold: `--min-confidence 0.15`
- Ensure image is not too blurry or low resolution

### Numbers Not Accurate

- The application includes automatic number correction
- Common mistakes are automatically fixed (O→0, I→1, etc.)
- Check the output Excel file - numbers should be converted to numeric type
- If numbers are still wrong, the image quality may need improvement

### Cyrillic Text Not Detected Correctly

- The application uses Russian OCR model for Cyrillic text
- Ensure the image quality is good
- Some Cyrillic characters may be confused with similar-looking characters

### Performance Issues

- First run will download language models (~500MB total)
- Processing large images may take time
- Use `--no-preprocess` for faster processing (may reduce accuracy)

## License

MIT License - see LICENSE file for details
