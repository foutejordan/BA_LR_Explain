import re

pattern = re.compile(r'(?<!])\n')

with open("../vec_vox1.txt", 'r') as file:
    lines = file.readlines()

# Join lines into a single string
text = ''.join(lines)

# Apply the replacement
text = pattern.sub('', text)

# Write the modified text into a new file
with open("vox1-transformed.txt", 'w') as new_file:
    new_file.write(text)
