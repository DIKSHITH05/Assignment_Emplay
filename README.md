# RFP/Bid Document Extractor

## Overview
This project extracts structured information from Request for Proposals (RFP) and Bid documents using Language Models. The system processes PDF and HTML files and outputs structured JSON data with 20 predefined fields.

## Quick Start

## Option 1: Automated Setup (Recommended)

### Linux/macOS

```bash
chmod +x setup.sh
./setup.sh
```

### Windows (PowerShell)

1. Open PowerShell in the `setup` folder.
2. Allow script execution for this session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

3. Run the setup script:

```powershell
.\setup.ps1
```

This script will:
1. Create virtual environment
2. Install all dependencies
3. Run extraction for Bid1 and Bid2
4. Verify results

### Option 2: Manual Setup
```bash
# Step 1: Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Setup OpenRouter API key (recommended)
# Option A: Create .env file
cp env.example .env
# Edit .env and add your API key

# Option B: Export directly
export OPENROUTER_API_KEY="your_key_here"
```

## Usage Examples

### Extract from Bid1 folder:
```bash
python enhanced_bid_extractor.py --input Bid1/
```

### Extract from Bid2 folder:
```bash
python enhanced_bid_extractor.py --input Bid2/
```

### Use OpenRouter (RECOMMENDED):
```bash
python enhanced_bid_extractor.py --input Bid1/ --openrouter
```

### Verify extraction accuracy:
```bash
python verify_extraction_accuracy.py
```

## Output
JSON files are saved to the `output/` folder with the following naming:
- `output/Bid1_extracted.json`
- `output/Bid2_extracted.json`

## Extracted Fields (20 Total)
1. Bid_Number
2. Title
3. Due_Date
4. Bid_Submission_Type
5. Term_of_Bid
6. Pre_Bid_Meeting
7. Installation
8. Bid_Bond_Requirement
9. Delivery_Date
10. Payment_Terms
11. Any_Additional_Documentation_Required
12. MFG_for_Registration
13. Contract_or_Cooperative_to_use
14. Model_no
15. Part_no
16. Product
17. contact_info
18. company_name
19. Bid_Summary
20. Product_Specification

## How It Works
1. Reads PDF and HTML files from the input folder
2. Extracts all text content from documents
3. Uses Language Models to identify and structure information
4. Maps extracted data to 20 predefined fields
5. Outputs structured JSON files

## Technical Details
- Primary Extraction: OpenRouter API (GPT-4o-mini)
- Fallback: Regex pattern matching
- Output Format: JSON with structured data
- Supported Formats: PDF, HTML, TXT

## Dependencies
- pdfplumber: PDF text extraction
- beautifulsoup4: HTML parsing
- openai: API integration
- langchain: LLM framework
- See requirements.txt for complete list

## Troubleshooting
If you encounter issues:
1. Make sure virtual environment is activated
2. Check that all dependencies are installed
3. Verify OpenRouter API key is set
4. Run verification script to check results

