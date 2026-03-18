"""
Extract district-level data from PBS Census 2023 Table 12 PDF.
Handles PBS's space-separated number format (e.g., '3 6.09' = 36.09).
"""
import pdfplumber
import pandas as pd
import re
import os

pdf_path = os.path.join(os.path.dirname(__file__), "table_12_punjab_districts.pdf")

def parse_literate_pct_line(line):
    """
    Parse the 'Literate %' line from PBS PDF.
    Format example: 'Literate % 3 6.09 4 3.68 2 8.18 3 5.59 2 7.39 3 5.48 1 8.86 5 .26 5 7.42 6 4.14 5 0.60 4 1.41'
    Values are: Total, Male, Female, Transgender, Rural_Total, Rural_Male, Rural_Female, Rural_Trans, Urban_Total, Urban_Male, Urban_Female, Urban_Trans
    Numbers have spaces in them: '3 6.09' = 36.09, '5 .26' = 5.26
    """
    parts = line.replace('Literate %', '').strip()
    
    # Strategy: find all decimal numbers. Each percentage value contains exactly one decimal point.
    # Split by decimal points to reconstruct values.
    # The pattern is: digits (possibly with spaces) followed by .digits
    # e.g., "3 6.09 4 3.68" -> we need to find values like "36.09", "43.68"
    
    # Remove all spaces and then re-parse won't work because "3 6.09" becomes "36.09" but 
    # "5 .26" becomes "5.26" - actually that works!
    # Wait - if we remove ALL spaces, "3 6.09 4 3.68" becomes "36.0943.68" which is wrong.
    
    # Better approach: Find each decimal number pattern allowing for internal spaces
    # A value is: optional_digit_group SPACE decimal_point digits
    # More precisely: (digits)(space)(digits).(digits) OR (digits).(digits)
    
    # Let's use a different approach: split on patterns that look like boundaries between numbers
    # Each number ends with .\d+ and the next number starts with a digit
    
    # First, collapse multiple spaces
    parts = re.sub(r'\s+', ' ', parts)
    
    # Pattern: capture sequences that form a decimal number
    # A number is: (one or more groups of digits possibly separated by single spaces) . (digits)
    # The decimal point is the anchor
    
    values = []
    # Find all occurrences of decimal numbers with potential space before the decimal or within integer part
    # Match pattern: digits (space digits)* space? . digits
    pattern = r'(\d[\d\s]*?\.[\d]+)'
    matches = re.findall(pattern, parts)
    
    for m in matches:
        # Remove internal spaces to get the actual number
        clean = m.replace(' ', '')
        try:
            values.append(float(clean))
        except ValueError:
            values.append(None)
    
    # Also handle edge cases like just "-" or single digit percentages
    return values


def parse_population_line(line, prefix):
    """Parse a population line, returning the first (TOTAL) value."""
    parts = line.replace(prefix, '').strip()
    parts = re.sub(r'\s+', ' ', parts)
    
    # Population numbers have commas and spaces: "1 ,891,721" = 1891721
    # Find the first large number
    # Remove the prefix, then grab everything up to the pattern break
    # Numbers look like: digit(s) ,digit(s),digit(s) or digit(s) space ,digit(s)
    
    # Simple approach: remove all spaces around commas, then split
    cleaned = re.sub(r'\s*,\s*', '', parts)
    # Now we have groups of digits possibly separated by spaces
    # The first group should be the total population
    nums = cleaned.split()
    
    # Reconstruct: take digits until we hit a gap that's too big
    # Actually, after removing commas, "1 891721 963665 ..." - first token is partial
    # Better: remove ALL spaces, then split by reasonable boundaries
    
    all_digits = re.sub(r'\s+', '', cleaned)
    # This gives us something like "18917219636659279381181365225..."
    # That's not useful. Let me try a different approach.
    
    # Go back to original: find comma-separated numbers with spaces
    raw = line.replace(prefix, '').strip()
    # Pattern for PBS numbers: digits, optionally followed by (space?,comma,space?,digits)+
    # e.g., "1 ,891,721" or "9 63,665"
    pattern = r'(\d[\d\s]*(?:\s*,\s*\d{3})*)'
    matches = re.findall(pattern, raw)
    
    results = []
    for m in matches:
        clean = m.replace(' ', '').replace(',', '')
        try:
            results.append(int(clean))
        except ValueError:
            pass
    
    return results


# Extract all text from PDF
all_text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            all_text += text + "\n"

# Parse district data
lines = all_text.split('\n')
districts_data = {}
current_district = None
in_district = False

for i, line in enumerate(lines):
    line = line.strip()
    
    # Check for district header
    district_match = re.match(r'^([A-Z][\sA-Z\.]+?)\s+DISTRICT\s*$', line)
    if district_match:
        current_district = district_match.group(1).strip()
        in_district = True
        if current_district not in districts_data:
            districts_data[current_district] = {}
        continue
    
    # Tehsil header = stop
    if re.match(r'^[A-Z][\sA-Z\.]+?\s+TEHSIL\s*$', line) or \
       re.match(r'^[A-Z][\sA-Z\.]+?\s+CITY\s+TEHSIL\s*$', line):
        in_district = False
        continue
    
    if not in_district or current_district is None:
        continue
    
    d = districts_data[current_district]
    
    # === Literacy Rate ===
    if line.startswith('Literate %'):
        values = parse_literate_pct_line(line)
        if len(values) >= 3 and 'literacy_rate' not in d:
            d['literacy_rate'] = values[0]
            d['male_literacy'] = values[1]
            d['female_literacy'] = values[2]
        if len(values) >= 7 and 'rural_literacy' not in d:
            d['rural_literacy'] = values[4]
            d['urban_literacy'] = values[8] if len(values) > 8 else None
    
    # === Population >=5 ===
    if line.startswith('Population >=5') or line.startswith('Population >= 5'):
        prefix = 'Population >=5' if 'Population >=5' in line else 'Population >= 5'
        nums = parse_population_line(line, prefix)
        if nums and 'population_5plus' not in d:
            d['population_5plus'] = nums[0]
    
    # === Population >=10 ===
    if line.startswith('Population >=10') or line.startswith('Population >=1 0'):
        prefix = 'Population >=10' if 'Population >=10' in line else 'Population >=1 0'
        nums = parse_population_line(line, prefix)
        if nums and 'population_10plus' not in d:
            d['population_10plus'] = nums[0]
    
    # === Enrolment Primary (total) ===
    if line.startswith('Enrolment Primary'):
        nums = parse_population_line(line, 'Enrolment Primary')
        if nums and 'enrolment_primary' not in d:
            d['enrolment_primary'] = nums[0]
    
    # === Enrolment Middle ===
    if line.startswith('Enrolment Middle'):
        nums = parse_population_line(line, 'Enrolment Middle')
        if nums and 'enrolment_middle' not in d:
            d['enrolment_middle'] = nums[0]
    
    # === Enrolment Matric ===
    if line.startswith('Enrolment Matric'):
        nums = parse_population_line(line, 'Enrolment Matric')
        if nums and 'enrolment_matric' not in d:
            d['enrolment_matric'] = nums[0]
            
    # === Enrolment Intermediate ===
    if line.startswith('Enrolment Intermidiate') or line.startswith('Enrolment Intermediate'):
        key = 'Enrolment Intermidiate' if 'Intermidiate' in line else 'Enrolment Intermediate'
        nums = parse_population_line(line, key)
        if nums and 'enrolment_intermediate' not in d:
            d['enrolment_intermediate'] = nums[0]
    
    # === Enrolment Graduation & above ===
    if line.startswith('Enrolment Graduation'):
        nums = parse_population_line(line, 'Enrolment Graduation above')
        if nums and 'enrolment_graduation_above' not in d:
            d['enrolment_graduation_above'] = nums[0]
    
    # === Out of School Children (5-16) ===
    if line.startswith('Out of School Children'):
        nums = parse_population_line(line, 'Out of School Children (5-16)')
        if nums and 'out_of_school_5_16' not in d:
            d['out_of_school_5_16'] = nums[0]
    
    # === Never to School (all) ===
    if line.startswith('Never to School (all)'):
        nums = parse_population_line(line, 'Never to School (all)')
        if nums and 'never_attended_school' not in d:
            d['never_attended_school'] = nums[0]
    
    # === Drop Out (5-16) ===
    if line.startswith('Drop Out'):
        nums = parse_population_line(line, 'Drop Out (5 - 16)')
        if nums and 'dropout_5_16' not in d:
            d['dropout_5_16'] = nums[0]
    
    # === Literate >=10 (absolute number) ===
    if line.startswith('Literate >=10') or line.startswith('Literate >=1 0'):
        prefix = 'Literate >=10' if 'Literate >=10' in line else 'Literate >=1 0'
        nums = parse_population_line(line, prefix)
        if nums and 'literate_10plus' not in d:
            d['literate_10plus'] = nums[0]
    
    # === Ever Attended ===
    if line.startswith('Ever Attended'):
        nums = parse_population_line(line, 'Ever Attended')
        if nums and 'ever_attended' not in d:
            d['ever_attended'] = nums[0]

# Build DataFrame
rows = []
for district_name, data in districts_data.items():
    row = {'district': district_name}
    row.update(data)
    rows.append(row)

df = pd.DataFrame(rows)

# Print results
print(f"\nFound {len(df)} districts:\n")
for _, row in df.iterrows():
    print(f"  {row['district']}: Literacy={row.get('literacy_rate', 'N/A')}%, "
          f"Male={row.get('male_literacy', 'N/A')}%, Female={row.get('female_literacy', 'N/A')}%")

# Save to CSV
output_path = os.path.join(os.path.dirname(__file__), "pbs_census_2023_extracted.csv")
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFull dataset:")
print(df.to_string())
