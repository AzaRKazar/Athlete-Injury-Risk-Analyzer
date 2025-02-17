import os 
from markitdown import MarkItDown

folder = 'injuries/football/raw/2023'
files = os.listdir(folder)

for file in files:
    if file.endswith('.pdf'):
        print(f'Processing {file}...')
        path = os.path.join(folder, file)
        filename = os.path.splitext(file)[0]
        md = MarkItDown()
        result = md.convert(path)
        with open(f'injuries/football/transformed/2023/{filename}.md', 'w') as f:
            f.write(result.text_content)
        print(f'Finished processing {file}.')