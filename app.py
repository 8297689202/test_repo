import streamlit as st
import requests
import json
import numpy as np
from PIL import Image
import pypdfium2 as pdfium
from paddleocr import PaddleOCR
import time
import re
import pdfplumber
import io
import base64

# Your existing URL and headers
url = "https://client-api-uat01.vfsai.com/chats/"
headers = {
    'Content-Type': 'application/json',
    'x-app-id': '9f5d2237b22effbf96bf513df5cf8b29',
    'authorization': 'Access 05c718d678a1f1f5c2f2b4b1b1a1ae5dd3331a6b08c594d67b7de6b4d20dd7e3ae3c30bed253a93e2dbc32a9ebfa80e00b903e1ffc9e174dd3a327ce8d0daa56ec40e1332b7b771a6945fb2a519a80a5'
}
PROMPT_TEMPLATE = '''You are an expert in analyzing travel documents and decoding flight information. Extract all relevant information from the provided travel document and return the data in a json format. Follow these guidelines:

1. Travel Number:
   - The "travel_number" refers to the unique booking identifier.
   - Valid labels/identifiers (in priority order):
     1. "Airline PNR" or "PNR:" or "PNR -"
     2. "Confirmation Number:"
     3. "Reservation Code:"
     4. "Check-in Reference:" (when not marked as agency/GDS reference)
     5. "GDS PNR" (when explicitly marked as ticket PNR, not agency reference)

   - Format Requirements:
     * Must be between 5 and 7 characters long (inclusive)
     * Must be a standalone identifier
     * Can contain letters and numbers
     * May contain a hyphen (-)
     * Special Rules for Confirmation Numbers:
       - When extracting from "Confirmation Number:" label:
         * Take ONLY the standalone alphanumeric string immediately after the label
         * Must be exactly 5-6 characters long
         * Extract ONLY the first complete identifier
         * Examples:
           - "Confirmation Number: ABC12D" -> extract "ABC12D"
           - "Confirmation Number: ABC12D 99" -> extract "ABC12D"
           - "Confirmation Number: ABC12D99" -> extract "ABC12D"
       - Specifically for Confirmation Numbers:
         * DO NOT include any digits or characters that appear after the first complete identifier
         * Stop extraction at the first complete valid identifier (5-6 characters)
         * Ignore any trailing numbers or text even if connected
         * Examples of INCORRECT extraction:
           - "Confirmation Number: ABC12D99" -> "ABC12D99" (wrong)
           - "Confirmation Number: ABC12D456" -> "ABC12D456" (wrong)
         * Examples of CORRECT extraction:
           - "Confirmation Number: ABC12D99" -> "ABC12D" (correct)
           - "Confirmation Number: ABC12D456" -> "ABC12D" (correct)
     * For Check-in References in complex layouts:
       - Example 1: "Agency Reference: 123456 Check-in Reference: ABC789(XY)" -> extract "ABC789"
       - Example 2: "Check-in Reference: DEF456(ZZ) / OTHER123" -> extract "DEF456"
     * When references appear in header-style format:
       Example:
       Agency Reference:  Check-in Reference:  Travel Agency:
       ABC123            XYZ789(DL)           AGENCY NAME
       -> extract "XYZ789"
     * Common formats:
       - After "PNR:" or "PNR -" or "Airline PNR:"
       - After "Confirmation Number:"
       - In a clearly labeled PNR field
       - In a dedicated reference field

   - Identification Rules:
     * Check document sections for explicit PNR labels first
     * Look for standalone PNR formats
     * For Check-in References, ignore any text in parentheses (e.g., from "ABC123(XY)" take only "ABC123")
     * When multiple identifiers exist:
       - Airline PNR takes highest priority
       - Confirmation Number second priority
       - GDS PNR (when explicitly marked as ticket PNR) third priority
       - Check-in Reference (when standalone) fourth priority
       - If multiple references exist, use ONLY the highest priority one

   - **IMPORTANT: NEVER extract:**
     * ANY number labeled as "Agency Reference:" or "Agency Ref:"
     * ANY reference that starts with "Agency"
     * ANY reference from a travel agency section
     * Booking References longer than 7 characters
     * Reference numbers containing spaces
     * E-ticket numbers (usually 10+ digits)
     * Flight numbers (airline code + numbers)
     * Order IDs (usually longer than 7 characters)
     * Cart Information numbers
     * Any part of a longer string
     * GDS References when marked as agency reference

   Validation Examples:
   CORRECT to extract:
   - Airline PNR formats:
     * "PNR: ABC123"
     * "Airline PNR - XYZ45P"
     * "PNR: KLM789"

   - Confirmation Number formats:
     * "Confirmation Number: N7K42L"
     * "Confirmation #: RT567K"

   - Check-in Reference formats:
     * "Check-in Ref: H8M92P"
     * "Check-in Reference: ABC123(XY)" -> extract "ABC123"
     * "Reference: WQ456T" (when not in agency section)

   INCORRECT to extract:
   - Agency references (never extract these):
     * "Agency Reference: ABC123"
     * "Agency Ref: XYZ789"
     * "GDS Reference: XY789Z" (when in agency section)
     * Any number after "Agency Reference:"

   - Other invalid formats:
     * "Booking: REF123456789"
     * "Reference: BOOK987654321"
     * "Flight numbers: BA 123"
     * "E-ticket: 1234567890123"
     * "Order ID: 34728581279"

   VALIDATION STEPS:
   1. First scan for explicit "Agency Reference" labels and EXCLUDE these numbers
   2. Then scan for explicit PNR labels in priority order
   3. Check each potential number against format requirements
   4. For Confirmation Numbers:
      a) First identify the label "Confirmation Number:"
      b) Extract only the first 5-6 characters after the label
      c) Stop extraction after getting valid identifier
      d) Ignore any trailing numbers or text
   5. For Check-in References:
      a) First identify the label "Check-in Reference:"
      b) Extract the immediately following alphanumeric string
      c) If it contains parentheses, take only the part before them
      d) Ignore any text after forward slashes (/)
   6. Verify it's not in exclusion list
   7. Confirm it's a complete standalone identifier
   8. For multiple valid numbers:
      a) Use Airline PNR if available
      b) Use Confirmation Number if no Airline PNR
      c) Use GDS PNR if clearly marked as ticket PNR
      d) Use Check-in Reference only if standalone and not agency-related

   Note: Always prioritize explicitly labeled identifiers over inferred ones. If in doubt, prefer not to extract rather than extract incorrectly.

2. Flight Name:
   - flight_name must ALWAYS include both airline name and flight number
   - Format as "AIRLINE NAME XX NNN" where:
     * AIRLINE NAME is the complete carrier name (e.g., "China Eastern Airlines", "Emirates")
     * XX is the airline code (e.g., "MU", "EK")
     * NNN is the flight number
   - Always extract the full airline name when available in the document
   - Common airline codes and names:
     * MU = China Eastern Airlines
     * EK = Emirates
     * CZ = China Southern Airlines
     * CA = Air China
     * QR = Qatar Airways
   - Examples of correct flight names:
     * "China Eastern Airlines MU 245"
     * "Emirates EK 302"
     * "Qatar Airways QR 545"
   - Do NOT abbreviate airline names
   - Do NOT omit airline name from the flight_name field

# Modify the Example Cases to include full airline names:
   Case 1 - Standard Format:
   ```
   British Airways BA 123
   LONDON (LHR) Terminal 3
   10:15 Mon

   PARIS (CDG) Terminal 2B
   13:30 Mon
   ```
   → Flight name: British Airways BA 123
   → Origin: LHR (10:15 is earlier)
   → Destination: CDG (13:30 is later)

# Update the Expected JSON Format example:
Expected JSON Format:
{
    "passenger_names": ["LASTNAME FIRSTNAME MIDDLENAME"],
    "flights": [
        {
            "passenger_names": ["LASTNAME FIRSTNAME MIDDLENAME"],
            "flight_origin": "XXX",
            "flight_destination": "YYY",
            "travel_number": "XXXXXX",
            "date_of_travel": "DD MMM YYYY HH:mm - DD MMM YYYY HH:mm",
            "flight_name": "FULL AIRLINE NAME XX NNN",  # e.g., "China Eastern Airlines MU 245"
            "notes": "via ZZZ"
        }
    ]
}

# Add to VALIDATION CHECKLIST:
1. For Each Segment:
   □ Identified complete flight name (airline name + code + number)
   □ Verified airline name is included in flight_name field
   □ Found exactly two timestamps
   ...

# Add to Common Errors to Avoid:
2. Common Errors to Avoid:
   □ Don't omit airline name from flight_name
   □ Don't abbreviate airline names
   □ Don't use only flight code and number


 3. Flight Segments and Directions:
   - A flight segment is defined by a unique flight number
   - CRITICAL: Follow these steps IN ORDER for EVERY segment:
     1. Find the flight number
     2. Locate ALL timestamps in that segment
     3. Match each timestamp with its associated city/airport
     4. EARLIER timestamp's city = ORIGIN (Departure)
     5. LATER timestamp's city = DESTINATION (Arrival)

   IMPORTANT RULES:
   * Direction is determined ONLY by timestamps, not text order
   * Ignore airline routes or typical flight patterns
   * Each segment must be processed independently
   * Text order of cities is irrelevant
   * Only timestamp association determines origin/destination

   Edge Cases and Examples:

   Case 1 - Standard Format:
   ```
   FL 123
   LONDON (LHR) Terminal 3
   10:15 Mon

   PARIS (CDG) Terminal 2B
   13:30 Mon
   ```
   → Origin: LHR (10:15 is earlier)
   → Destination: CDG (13:30 is later)

   Case 2 - Reversed Text Order:
   ```
   FL 456
   DUBAI (DXB)
   Arrives: 05:30 Wed

   SINGAPORE (SIN)
   Departs: 22:45 Tue
   ```
   → Origin: SIN (22:45 is earlier)
   → Destination: DXB (05:30 is later)
   Note: Text order is DXB→SIN but timestamps show SIN→DXB

   Case 3 - Overnight Flight:
   ```
   FL 789
   TOKYO (HND)
   23:50 Thu

   SEOUL (ICN)
   01:30 Fri +1
   ```
   → Origin: HND (23:50 is earlier)
   → Destination: ICN (01:30 next day is later)

   Case 4 - Mixed Format with Connection:
   ```
   FL 234
   Departure: DELHI (DEL) 08:55
   Via: MUMBAI (BOM)
   Arrival: 10:45 | Departure: 11:30
   Final: BANGKOK (BKK) 16:20
   ```
   → Should be processed as one segment
   → Origin: DEL (08:55 is first time)
   → Destination: BKK (16:20 is final time)
   → Note: Include "via BOM" in notes

   Case 5 - Unusual Format:
   ```
   FL 567 from NEW YORK (JFK)
   Time of departure: 19:15
   Operating carrier info...
   Destination information:
   MIAMI (MIA) scheduled arrival: 22:30
   ```
   → Origin: JFK (19:15 is earlier)
   → Destination: MIA (22:30 is later)

   Case 6 - Reverse Route with Different Dates:
   ```
   FL 890
   FRANKFURT (FRA) Terminal 1
   Wed, 25 Nov 21:15

   SINGAPORE (SIN) Terminal 3
   Thu, 26 Nov 05:40 +1
   ```
   → Origin: FRA (21:15 is earlier)
   → Destination: SIN (05:40 next day is later)

   Case 7 - Complex Multi-City Format:
   ```
   Journey Segment: FL 111
   Arrival City: BANGKOK (BKK)
   Arrival Time: 18:55
   Departure Point: DUBAI (DXB)
   Departure Time: 14:35
   ```
   → Origin: DXB (14:35 is earlier)
   → Destination: BKK (18:55 is later)
   Note: Despite "Arrival City" being listed first

VALIDATION CHECKLIST:
1. For Each Segment:
   □ Identified flight number
   □ Found exactly two timestamps
   □ Matched each timestamp with correct city
   □ Confirmed earlier time = origin
   □ Confirmed later time = destination
   □ Checked for date changes (+1)
   □ Verified chronological sequence

2. Common Errors to Avoid:
   □ Don't assume first city listed is origin
   □ Don't assume second city listed is destination
   □ Don't rely on words like "from" or "to"
   □ Don't use airline route patterns
   □ Don't assume connection cities are destinations

3. Quality Checks:
   □ Every origin has a departure time
   □ Every destination has an arrival time
   □ All times follow chronological order
   □ Date changes are properly noted
   □ Layovers are noted but don't create new segments

3. Origins and Destinations:
  - For each flight segment:
    * Read the city/airport codes in PAIRS (departure and arrival)
    * Each pair represents one segment of the journey
  - Route Continuity Rules:
    * Outbound segments should connect logically
    * Return segments may start from a different city than final outbound destination
    * Check connecting times between segments
  - Multiple-Leg Journeys:
    * Verify if cities appear as both arrival and departure points
    * Connect segments based on chronological order and flight numbers
    * Pay attention to dates to distinguish outbound vs return segments

    NOTE:
    - Different flight numbers mean different segments.
    - Return journey may follow a different route than outbound.
    - Connecting cities will appear twice: once as arrival, once as departure.
    - "+1 Day(s)" indicates overnight flight, not direction change.


4. Dates:
   - Date format must be "DD MMM YYYY HH:mm - DD MMM YYYY HH:mm"
   - Include both departure and arrival times
   - For multi-day flights, show both dates with times
   - For flights with layovers, show initial departure and final arrival times

5. Passengers:
   - Names must be in UPPERCASE with proper spacing
   - Format as "LASTNAME FIRSTNAME MIDDLENAME" if available
   - Remove salutations (Mr, Mrs, etc.)
   - If multiple tickets show identical itineraries for different passengers:
     * List all passengers under each flight segment
     * Combine identical flights for multiple passengers into single segments

NOTES:
- Different flight numbers mean different segments
- Return journey may follow a different route than outbound
- A gap between arrival and departure cities in consecutive flights is valid
- "+1 Day(s)" indicates overnight flight, not direction change
- Focus on extracting segments exactly as shown, without assumptions about continuity
- Use dates and times as primary way to sequence flights

Expected JSON Format:
{
    "passenger_names": ["LASTNAME FIRSTNAME MIDDLENAME"],
    "flights": [
        {
            "passenger_names": ["LASTNAME FIRSTNAME MIDDLENAME"],
            "flight_origin": "XXX",
            "flight_destination": "YYY",
            "travel_number": "XXXXXX",
            "date_of_travel": "DD MMM YYYY HH:mm - DD MMM YYYY HH:mm",
            "flight_name": "Airline XX NNN",
            "notes": "via ZZZ" // for flights with layovers
        }
    ]
}

Text to analyze: '''

def convert_pdf_to_images(pdf_content):
    """Convert PDF to images using pypdfium2."""
    try:
        # Load PDF from bytes
        pdf = pdfium.PdfDocument(pdf_content)
        images = []

        # Convert each page to image
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            # Render page to image (scale=2 gives better quality, similar to dpi=200)
            pil_image = page.render(
                scale=2.0,            # Increase for better quality (like DPI)
                rotation=0            # No rotation
            ).to_pil()
            images.append(pil_image)

        return images
    except Exception as e:
        print(f"Error converting PDF to images: {str(e)}")
        return None

def process_with_paddleocr(pdf_content):
    """Process PDF using PaddleOCR with pdfium conversion."""
    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=False,
            show_log=False
        )

        # Convert PDF to images using pdfium
        images = convert_pdf_to_images(pdf_content)
        if not images:
            return None

        extracted_text = []

        for image in images:
            # Convert PIL image to numpy array
            img_array = np.array(image)

            # Perform OCR
            result = ocr.ocr(img_array)

            # Extract text from OCR results
            page_text = []
            for line in result:
                for word_info in line:
                    if isinstance(word_info, list) and len(word_info) >= 2:
                        text = word_info[1][0]  # Extract text
                        confidence = word_info[1][1]  # Extract confidence
                        if confidence > 0.5:  # Filter low confidence results
                            page_text.append(text)

            # Join text with proper spacing
            page_content = ' '.join(page_text)
            if page_content.strip():
                extracted_text.append(page_content)

        return '\n\n'.join(extracted_text)

    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        return None

def validate_extracted_text(text):
    """Validate if the extracted text contains enough meaningful content."""
    if not text:
        return False

    # Check minimum length
    if len(text.strip()) < 100:
        return False

    # Check for key flight document indicators
    key_terms = ['flight', 'passenger', 'date', 'departure', 'arrival',
                 'pnr', 'booking', 'ticket', 'airline']
    found_terms = sum(1 for term in key_terms if term.lower() in text.lower())
    if found_terms < 3:
        return False

    # Check text quality
    readable_chars = sum(c.isalnum() or c.isspace() for c in text)
    if readable_chars / len(text) < 0.65:  # Slightly relaxed threshold
        return False

    return True


def hybrid_process_pdf(file_content):
    """Hybrid PDF processing using PDFPlumber and PaddleOCR with pdfium."""
    try:
        # First attempt with PDFPlumber
        with io.BytesIO(file_content) as pdf_file:
            try:
                with pdfplumber.open(pdf_file) as pdf:
                    extracted_text = ""

                    for page in pdf.pages:
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                extracted_text += "\n".join([" ".join(filter(None, row))
                                                          for row in table]) + "\n\n"

                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n\n"

                    if validate_extracted_text(extracted_text):
                        return enhanced_clean_and_preprocess_text(extracted_text)

            except Exception as e:
                pass

        # Fallback to OCR
        ocr_text = process_with_paddleocr(file_content)
        if ocr_text and validate_extracted_text(ocr_text):
            return enhanced_clean_and_preprocess_text(ocr_text)

        return None

    except Exception as e:
        return None


def enhanced_clean_and_preprocess_text(text):
    """Enhanced text cleaning with better formatting preservation."""
    # Remove URLs and web artifacts
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove page indicators and artifacts
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'about:blank', '', text)

    # Handle date formats
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1 \2 \3', text)

    # Clean table artifacts
    text = re.sub(r'\|\s+\|', ' ', text)
    text = re.sub(r'[-_]{3,}', '', text)

    # Basic cleanup (from your existing function)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\r', '\n')

    # Fix flight numbers and codes
    text = re.sub(r'([A-Z]{2})\s*(\d{1,4}(?:[A-Z])?)', r'\1 \2', text)
    text = re.sub(r'([A-Z]{3})\s*\(', r'\1 (', text)
    text = re.sub(r'\)\s*([A-Z]{3})', r') \1', text)

    # Fix date and time formats
    text = re.sub(r'(\d{2}):(\d{2})\s*(?:hrs?|AM|PM)?',
                  r'\1:\2',
                  text,
                  flags=re.IGNORECASE)
    text = re.sub(r'(\d{2})\s*(\w{3})\s*(\d{4})', r'\1 \2 \3', text)

    # Handle flight segments better
    text = re.sub(r'(Flight|Flight:)\s*',
                  'Flight: ',
                  text,
                  flags=re.IGNORECASE)
    text = re.sub(r'(Terminal|Terminal:)\s*',
                  'Terminal: ',
                  text,
                  flags=re.IGNORECASE)

    # Clean special characters but preserve important ones
    text = re.sub(r'[^a-zA-Z0-9\s\(\)\/\-:,\.]', ' ', text)

    # Normalize multiple newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()


def format_for_frontend(json_output):
    """Format JSON output for display with passenger count and improved error handling."""
    try:
        # First try to parse the input directly as JSON
        try:
            data = json.loads(json_output)
        except json.JSONDecodeError:
            # Clean the output of any markdown formatting
            clean_json = json_output.replace('```json', '').replace('```', '').strip()
            try:
                data = json.loads(clean_json)
            except json.JSONDecodeError:
                # If still fails, try to extract JSON from markdown format
                json_match = re.search(r'{.*}', clean_json, re.DOTALL)
                if not json_match:
                    return f"Error: Could not find valid JSON data in the output\nOriginal output: {json_output}"
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    return f"Error parsing JSON: {str(e)}\nOriginal JSON output: {json_output}"

        # Rest of the function remains the same
        passenger_names = data.get('passenger_names', [])
        if isinstance(passenger_names, str):
            passenger_names = [passenger_names]

        passenger_count = len(passenger_names)
        passenger_names_str = ','.join(passenger_names)

        header = [
            f"Number of Passengers: {passenger_count}",
            f"Passengers: {passenger_names_str}",
            ""
        ]

        formatted_flights = []
        for idx, flight in enumerate(data['flights'], 1):
            flight_passengers = flight.get('passenger_names', [])
            if not flight_passengers:
                flight_passengers = passenger_names

            if isinstance(flight_passengers, str):
                flight_passengers = [flight_passengers]

            flight_passengers_str = ','.join(flight_passengers)

            formatted_flight = [
                f"Flight {idx}:",
                f"- Passengers: {flight_passengers_str}",
                f"- Flight origin: {flight['flight_origin']}",
                f"- Flight destination: {flight['flight_destination']}",
                f"- Travel number/PNR: {flight['travel_number']}",
                f"- Date of travel: {flight['date_of_travel']}",
                f"- Flight name: {flight['flight_name']}"
            ]

            if flight.get('notes'):
                if flight['notes'].strip():
                    formatted_flight.append(f"- Notes: {flight['notes']}")

            formatted_flights.append("\n".join(formatted_flight))

        all_sections = header + formatted_flights
        return "\n\n".join(all_sections)

    except Exception as e:
        return f"Error formatting output: {str(e)}\nOriginal JSON output: {json_output}"


def display_pdf(pdf_content):
    """Display PDF in Streamlit."""
    try:
        base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def main():
    st.set_page_config(page_title="Flight Document Analyzer", layout="wide")

    st.title("✈️ Flight Document Analyzer")

    uploaded_file = st.file_uploader("Upload flight document (PDF)", type=['pdf'])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📄 Uploaded Document")
            file_content = uploaded_file.read()
            display_pdf(file_content)

        with col2:
            with st.spinner('Processing...'):
                extracted_text = hybrid_process_pdf(file_content)

                if extracted_text:
                    data = {
                        "question": PROMPT_TEMPLATE + extracted_text
                    }

                    response = requests.post(url, headers=headers, data=json.dumps(data))

                    if response.status_code in [200, 201]:
                        try:
                            json_response = json.loads(response.text)
                            json_output = json_response["messages"][0]["answer"]
                            formatted_output = format_for_frontend(json_output)

                            # Show exact formatted output including "Processed Flight Information:"
                            st.text("Processed Flight Information:")
                            st.text(formatted_output)

                        except Exception as e:
                            st.error("Error processing response")
                    else:
                        st.error("Unable to process document")
                else:
                    st.error("Could not extract information from document")

if __name__ == "__main__":
    main()