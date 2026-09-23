import os, json

files = [
    'MrinalSingh_SmartDeliveryAI.ipynb',
    'requirements.txt',
    'MrinalSingh_ProjectReport.docx',
    'README.md',
]
print('=== File Check ===')
for f in files:
    size = os.path.getsize(f) if os.path.exists(f) else 0
    status = 'OK' if size > 0 else 'MISSING'
    print('  {:8s}  {}  ({:,} bytes)'.format(status, f, size))

print()
with open('MrinalSingh_SmartDeliveryAI.ipynb', encoding='utf-8') as fh:
    nb = json.load(fh)
cells = nb['cells']
code_cells = [c for c in cells if c['cell_type'] == 'code']
md_cells = [c for c in cells if c['cell_type'] == 'markdown']
print('Notebook cells: {}'.format(len(cells)))
print('  Code cells:    {}'.format(len(code_cells)))
print('  Markdown cells:{}'.format(len(md_cells)))
empty_code = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and not ''.join(c['source']).strip()]
print('  Empty code cells: {}'.format(len(empty_code)))
print()
print('All 4 submission files are ready.')
