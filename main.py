#!/usr/bin/env python3
"""
Medical Report OCR Parser with Ollama DeepSeek
Extract text from medical reports and convert to JSON using local Ollama
"""

import os
import cv2
import numpy as np
import json
import requests
from pathlib import Path
import glob
from datetime import datetime
import pytesseract
from tqdm import tqdm
import traceback

# ====== CONFIGURATION VARIABLES ======
INPUT_FOLDER = "./input_images"  # Folder containing medical report images
OUTPUT_FOLDER = "./output"       # Where to save results
MAX_IMAGES = 0                   # 0 = process all images, else limit to this number
OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
OLLAMA_MODEL = "llama3.2:3b"  # Model to use
DEBUG_MODE = True                # Enable detailed debugging output
# =====================================

class MedicalReportOCR:
    def __init__(self, ollama_url=OLLAMA_BASE_URL, model_name=OLLAMA_MODEL):
        """Initialize OCR processor and Ollama client"""
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Test Ollama connection
        try:
            response = requests.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                print(f"✅ Connected to Ollama at {ollama_url}")
                
                # Check if model is available
                models = [model['name'] for model in response.json().get('models', [])]
                if model_name in models:
                    print(f"✅ Model {model_name} is available")
                else:
                    print(f"⚠️  Model {model_name} not found. Available models: {models}")
                    print(f"   Run: ollama pull {model_name}")
            else:
                print(f"❌ Failed to connect to Ollama at {ollama_url}")
        except Exception as e:
            print(f"❌ Ollama connection error: {e}")
            print("   Make sure Ollama is running: ollama serve")
    
    def preprocess_image(self, image_path):
        """Preprocess image for better OCR results"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(sharpened)
            
            if DEBUG_MODE:
                print(f"   🖼️  Image preprocessing completed successfully")
            
            return enhanced
        except Exception as e:
            print(f"   ❌ Image preprocessing error: {e}")
            raise
    
    def extract_text_tesseract(self, image_path):
        """Extract text using Tesseract"""
        try:
            if DEBUG_MODE:
                print(f"   🔍 Starting Tesseract OCR extraction...")
            
            processed_img = self.preprocess_image(image_path)
            
            # Get text with confidence
            data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
            
            extracted_texts = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:
                    text = data['text'][i].strip()
                    if text:
                        extracted_texts.append({
                            'text': text,
                            'confidence': int(data['conf'][i])
                        })
            
            # Combine all text
            full_text = ' '.join([item['text'] for item in extracted_texts])
            
            if DEBUG_MODE:
                print(f"   📝 OCR extracted {len(extracted_texts)} text blocks")
                print(f"   📏 Full text length: {len(full_text)} characters")
                if len(full_text) > 0:
                    preview = full_text[:200] + "..." if len(full_text) > 200 else full_text
                    print(f"   👀 Text preview: {preview}")
            
            return full_text, extracted_texts
            
        except Exception as e:
            print(f"   ❌ OCR extraction failed: {e}")
            if DEBUG_MODE:
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            return "", []
    
    def generate_json_with_ollama(self, extracted_text, image_filename):
        """Use Ollama DeepSeek to convert extracted text to structured JSON"""
        
        if DEBUG_MODE:
            print(f"   🤖 Starting Ollama processing...")
            print(f"   📊 Input text length: {len(extracted_text)} characters")
        
        # Truncate text if too long to avoid token limits
        max_text_length = 8000  # Adjust based on your model's context window
        if len(extracted_text) > max_text_length:
            extracted_text = extracted_text[:max_text_length] + "\n[TEXT TRUNCATED DUE TO LENGTH]"
            if DEBUG_MODE:
                print(f"   ✂️  Text truncated to {max_text_length} characters")
        
        prompt = f"""You are an expert medical report parser. I have extracted text from a medical report image using OCR. Please analyze this text and convert it into a well-structured JSON format.

The extracted text from the medical report is:
{extracted_text}

Please create a comprehensive JSON structure that includes:

1. **hospital_info**: Hospital name, address, phone, website, etc.
2. **patient_info**: Patient details like name, age, gender, ID, etc.
3. **doctor_info**: Referring doctor, consultant, pathologist, etc.
4. **report_info**: Report type, dates (collection, report), sample info, etc.
5. **test_results**: Array of all tests with:
   - test_name
   - result_value  
   - reference_range
   - unit
   - status (normal/abnormal if determinable)
6. **additional_info**: Any notes, interpretations, or other relevant information

Guidelines:
- Extract ALL available information from the text
- If a field is not found, include it with null value
- For test results, try to identify patterns like "TEST_NAME VALUE RANGE UNIT"
- Preserve exact values and ranges as found in the text
- Clean up obvious OCR errors where possible
- Make the JSON as comprehensive and accurate as possible

Return ONLY the JSON structure, no additional text or explanations."""

        try:
            if DEBUG_MODE:
                print(f"   📡 Sending request to Ollama API...")
                print(f"   🔗 URL: {self.ollama_url}/api/generate")
                print(f"   🏷️  Model: {self.model_name}")
            
            # Call Ollama API
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 2048
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=120
            )
            
            if DEBUG_MODE:
                print(f"   📥 Ollama response status: {response.status_code}")
                print(f"   📏 Response content length: {len(response.text)} characters")
            
            if response.status_code != 200:
                error_msg = f'Ollama API error: HTTP {response.status_code}'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                    print(f"   📄 Response headers: {dict(response.headers)}")
                    print(f"   📄 Response content: {response.text[:500]}...")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text,
                    'status_code': response.status_code
                }
            
            # Parse the response
            try:
                result = response.json()
                if DEBUG_MODE:
                    print(f"   ✅ Successfully parsed Ollama response JSON")
                    print(f"   🔑 Response keys: {list(result.keys())}")
            except json.JSONDecodeError as e:
                error_msg = f'Failed to parse Ollama response as JSON: {str(e)}'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                    print(f"   📄 Raw response: {response.text[:1000]}...")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': response.text
                }
            
            json_text = result.get('response', '').strip()
            
            if DEBUG_MODE:
                print(f"   📏 JSON text length: {len(json_text)} characters")
                if len(json_text) == 0:
                    print(f"   ⚠️  WARNING: Empty response from Ollama!")
                    print(f"   🔍 Full Ollama result: {result}")
                else:
                    preview = json_text[:300] + "..." if len(json_text) > 300 else json_text
                    print(f"   👀 JSON preview: {preview}")
            
            if not json_text:
                return {
                    'success': False,
                    'error': 'Empty response from Ollama',
                    'raw_response': json_text,
                    'full_ollama_result': result
                }
            
            # Clean up the response to get just the JSON
            original_json_text = json_text
            
            if json_text.startswith('```json'):
                json_text = json_text[7:]
                if DEBUG_MODE:
                    print(f"   🧹 Removed ```json prefix")
            elif json_text.startswith('```'):
                json_text = json_text[3:]
                if DEBUG_MODE:
                    print(f"   🧹 Removed ``` prefix")
            
            if json_text.endswith('```'):
                json_text = json_text[:-3]
                if DEBUG_MODE:
                    print(f"   🧹 Removed ``` suffix")
            
            json_text = json_text.strip()
            
            if DEBUG_MODE and json_text != original_json_text:
                print(f"   🧹 Cleaned JSON text length: {len(json_text)} characters")
            
            # Try to find JSON in the response if it's not at the beginning
            if not json_text.startswith('{') and not json_text.startswith('['):
                # Look for JSON patterns in the text
                import re
                json_match = re.search(r'(\{.*\})', json_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                    if DEBUG_MODE:
                        print(f"   🔍 Extracted JSON from response using regex")
                else:
                    if DEBUG_MODE:
                        print(f"   ⚠️  No JSON structure found in response!")
                        print(f"   📄 Full cleaned text: {json_text}")
            
            # Parse and return JSON
            try:
                parsed_json = json.loads(json_text)
                if DEBUG_MODE:
                    print(f"   ✅ Successfully parsed JSON structure")
                    print(f"   🔑 JSON keys: {list(parsed_json.keys()) if isinstance(parsed_json, dict) else 'Not a dict'}")
            except json.JSONDecodeError as e:
                error_msg = f'JSON parsing error: {str(e)}'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                    print(f"   📄 JSON text that failed to parse: {json_text[:500]}...")
                    print(f"   🔧 JSON error position: line {e.lineno}, column {e.colno}")
                
                return {
                    'success': False,
                    'error': error_msg,
                    'raw_response': json_text,
                    'original_response': original_json_text,
                    'json_error_details': {
                        'line': e.lineno,
                        'column': e.colno,
                        'message': e.msg
                    }
                }
            
            # Add metadata
            parsed_json['_metadata'] = {
                'source_image': image_filename,
                'extraction_method': 'tesseract_ollama_deepseek',
                'processing_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name
            }
            
            if DEBUG_MODE:
                print(f"   🎉 JSON processing completed successfully!")
            
            return {
                'success': True,
                'json_data': parsed_json,
                'raw_response': json_text
            }
            
        except requests.RequestException as e:
            error_msg = f'Ollama request error: {str(e)}'
            if DEBUG_MODE:
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
        except Exception as e:
            error_msg = f'Unexpected error in Ollama processing: {str(e)}'
            if DEBUG_MODE:
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'error': error_msg,
                'raw_response': None
            }
    
    def process_image(self, image_path):
        """Process a single medical report image"""
        image_filename = os.path.basename(image_path)
        print(f"📄 Processing: {image_filename}")
        
        try:
            # Extract text using Tesseract
            extracted_text, extraction_details = self.extract_text_tesseract(image_path)
            
            if not extracted_text.strip():
                error_msg = 'No text extracted from image'
                if DEBUG_MODE:
                    print(f"   ❌ {error_msg}")
                
                return {
                    'success': False,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'error': error_msg
                }
            
            print(f"   📝 Extracted {len(extraction_details)} text blocks")
            
            # Generate structured JSON using Ollama
            ollama_result = self.generate_json_with_ollama(extracted_text, image_filename)
            
            if ollama_result['success']:
                if DEBUG_MODE:
                    print(f"   ✅ Successfully generated JSON structure")
                
                return {
                    'success': True,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'extracted_text': extracted_text,
                    'extraction_details': extraction_details,
                    'structured_json': ollama_result['json_data'],
                    'ollama_raw_response': ollama_result['raw_response']
                }
            else:
                if DEBUG_MODE:
                    print(f"   ❌ Failed to generate JSON: {ollama_result['error']}")
                
                return {
                    'success': False,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'error': ollama_result['error'],
                    'extracted_text': extracted_text,
                    'ollama_raw_response': ollama_result.get('raw_response'),
                    'ollama_error_details': ollama_result
                }
                
        except Exception as e:
            error_msg = f'Processing error: {str(e)}'
            if DEBUG_MODE:
                print(f"   ❌ {error_msg}")
                print(f"   🔧 Full error traceback:")
                traceback.print_exc()
            
            return {
                'success': False,
                'image_path': image_path,
                'image_filename': image_filename,
                'error': error_msg
            }

def get_image_files(folder_path):
    """Get all image files from the specified folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(image_files)

def save_raw_text(extracted_text, image_filename, text_output_dir):
    """Save raw extracted text to a .txt file"""
    base_name = os.path.splitext(image_filename)[0]
    txt_filename = f"{base_name}.txt"
    txt_filepath = os.path.join(text_output_dir, txt_filename)
    
    try:
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        return True, txt_filename
    except Exception as e:
        if DEBUG_MODE:
            print(f"   ❌ Error saving raw text: {e}")
        return False, str(e)

def main():
    """Main processing function"""
    
    # Validate input folder
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Input folder not found: {INPUT_FOLDER}")
        print(f"   Create the folder and add medical report images")
        return
    
    # Create output directories
    json_output_dir = os.path.join(OUTPUT_FOLDER, "json")
    text_output_dir = os.path.join(OUTPUT_FOLDER, "text")
    debug_output_dir = os.path.join(OUTPUT_FOLDER, "debug")
    
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(text_output_dir, exist_ok=True)
    if DEBUG_MODE:
        os.makedirs(debug_output_dir, exist_ok=True)
        print(f"📂 Debug output directory: {debug_output_dir}")
    
    print(f"📂 JSON output directory: {json_output_dir}")
    print(f"📂 Text output directory: {text_output_dir}")
    
    # Get image files
    image_files = get_image_files(INPUT_FOLDER)
    
    if not image_files:
        print(f"❌ No image files found in: {INPUT_FOLDER}")
        print("   Supported formats: JPG, JPEG, PNG, BMP, TIFF")
        return
    
    # Limit number of images if specified
    if MAX_IMAGES > 0:
        image_files = image_files[:MAX_IMAGES]
        print(f"📊 Processing limited to {MAX_IMAGES} images")
    
    print(f"📊 Found {len(image_files)} image(s) to process")
    
    # Initialize OCR processor
    try:
        ocr_processor = MedicalReportOCR()
    except Exception as e:
        print(f"❌ Failed to initialize OCR processor: {str(e)}")
        if DEBUG_MODE:
            traceback.print_exc()
        return
    
    # Process each image
    successful_count = 0
    failed_count = 0
    text_saved_count = 0
    
    for i, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        print(f"\n{'='*50}")
        print(f"Processing {i+1}/{len(image_files)}")
        print(f"{'='*50}")
        
        # Process image
        result = ocr_processor.process_image(image_path)
        
        # Generate output filenames
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_filename = f"{base_name}_extracted.json"
        json_filepath = os.path.join(json_output_dir, json_filename)
        
        # Save raw extracted text regardless of JSON processing success
        if 'extracted_text' in result and result['extracted_text'].strip():
            text_success, text_result = save_raw_text(
                result['extracted_text'], 
                result['image_filename'], 
                text_output_dir
            )
            
            if text_success:
                print(f"   📝 Raw text saved: {text_result}")
                text_saved_count += 1
            else:
                print(f"   ❌ Failed to save raw text: {text_result}")
        
        if result['success']:
            # Save structured JSON
            try:
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(result['structured_json'], f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Successfully saved: {json_filename}")
                
                # Display summary of extracted data
                json_data = result['structured_json']
                hospital_name = json_data.get('hospital_info', {})
                if isinstance(hospital_name, dict):
                    hospital_name = hospital_name.get('hospital_name', 'N/A')
                
                patient_name = json_data.get('patient_info', {})
                if isinstance(patient_name, dict):
                    patient_name = patient_name.get('name', 'N/A')
                
                test_results = json_data.get('test_results', [])
                test_count = len(test_results) if isinstance(test_results, list) else 0
                
                print(f"   📋 Hospital: {hospital_name}")
                print(f"   👤 Patient: {patient_name}")
                print(f"   🧪 Tests found: {test_count}")
                
                successful_count += 1
                
            except Exception as e:
                print(f"   ❌ Failed to save JSON: {str(e)}")
                if DEBUG_MODE:
                    traceback.print_exc()
                failed_count += 1
        else:
            # Save detailed error information
            error_data = {
                'error': result['error'],
                'image_path': result['image_path'],
                'extracted_text': result.get('extracted_text', ''),
                'timestamp': datetime.now().isoformat(),
                'debug_info': result.get('ollama_error_details', {})
            }
            
            error_filename = f"{base_name}_error.json"
            error_filepath = os.path.join(json_output_dir, error_filename)
            
            try:
                with open(error_filepath, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)
                print(f"   💾 Error details saved to: {error_filename}")
            except Exception as e:
                print(f"   ❌ Failed to save error details: {str(e)}")
            
            # Save debug information if enabled
            if DEBUG_MODE and 'ollama_error_details' in result:
                debug_filename = f"{base_name}_debug.json"
                debug_filepath = os.path.join(debug_output_dir, debug_filename)
                
                try:
                    with open(debug_filepath, 'w', encoding='utf-8') as f:
                        json.dump(result['ollama_error_details'], f, indent=2, ensure_ascii=False)
                    print(f"   🔧 Debug info saved to: {debug_filename}")
                except Exception as e:
                    print(f"   ❌ Failed to save debug info: {str(e)}")
            
            print(f"   ❌ Failed to process: {result['error']}")
            failed_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {successful_count} images")
    print(f"❌ Failed to process: {failed_count} images")
    print(f"📝 Raw text files saved: {text_saved_count}")
    print(f"📂 Output folder: {OUTPUT_FOLDER}")
    
    if DEBUG_MODE:
        print(f"🔧 Debug mode was enabled - check debug folder for detailed logs")
    
    return successful_count, failed_count, text_saved_count

if __name__ == "__main__":
    print("🚀 Starting Medical Report OCR Processing with Ollama DeepSeek")
    print(f"📂 Input folder: {INPUT_FOLDER}")
    print(f"🔢 Max images: {'All' if MAX_IMAGES == 0 else MAX_IMAGES}")
    print(f"🤖 Using: Tesseract OCR + Ollama {OLLAMA_MODEL}")
    print(f"🔧 Debug mode: {'Enabled' if DEBUG_MODE else 'Disabled'}")
    
    try:
        successful, failed, text_saved = main()
    except KeyboardInterrupt:
        print("\n⏸️  Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if DEBUG_MODE:
            print("🔧 Full error traceback:")
            traceback.print_exc()