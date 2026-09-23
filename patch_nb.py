"""
Patch notebook: add required cell IDs for nbformat 5.1+
"""
import json, uuid

with open('MrinalSingh_SmartDeliveryAI.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nb['nbformat_minor'] = 5
for cell in nb['cells']:
    if 'id' not in cell:
        cell['id'] = uuid.uuid4().hex[:8]

with open('MrinalSingh_SmartDeliveryAI.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Notebook patched with cell IDs.')
