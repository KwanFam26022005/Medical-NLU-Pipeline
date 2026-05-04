import json

with open('d:/Chatbot-Y tế/train_4_2_2026.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('d:/Chatbot-Y tế/full_extract.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell.get('source', []))
        ctype = cell['cell_type']
        out.write(f'=== CELL {i} ({ctype}) ===\n')
        out.write(src[:300] + '\n')
        if ctype == 'code':
            for j, output in enumerate(cell.get('outputs', [])):
                if output['output_type'] == 'stream':
                    text = ''.join(output['text'])
                    out.write(f'  [OUTPUT {j} stream, len={len(text)}]\n')
                    if len(text) > 2000:
                        out.write(text[:500] + '\n...[TRUNCATED]...\n' + text[-1500:] + '\n')
                    else:
                        out.write(text + '\n')
                elif output['output_type'] in ('execute_result', 'display_data'):
                    data = output.get('data', {})
                    if 'text/plain' in data:
                        t = ''.join(data['text/plain'])
                        if len(t) < 500:
                            out.write(f'  [OUTPUT {j} display]: {t}\n')
        out.write('\n')

print("Done")
