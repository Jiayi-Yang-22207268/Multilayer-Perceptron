import csv

# Define attribute names (features first, then letter at the end)
attributes = [
    'x-box', 'y-box', 'width', 'high', 'onpix',
    'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar',
    'x2ybr', 'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx',
    'letter'
]

input_file = 'letter-recognition.data'
output_file = 'letter-recognition.csv'

with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    
    # Write header row
    writer.writerow(attributes)
    
    # Process each line
    for line in infile:
        line = line.strip()
        if line:
            values = line.split(',')
            # Original format: letter,feature1,feature2,...,feature16
            # New format: feature1,feature2,...,feature16,letter
            letter = values[0]
            features = values[1:]
            writer.writerow(features + [letter])

print(f"Converted {input_file} to {output_file}")
