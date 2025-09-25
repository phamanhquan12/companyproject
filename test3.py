import pickle
import os
import pymupdf4llm
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
import re
from IPython.display import display, Markdown
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed', 'luong.pkl')
load_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'luong.pdf')

content = pymupdf4llm.to_markdown(load_path, show_progress=True, table_strategy='lines_strict', page_chunks=True)

print(content[0]['metadata'])

print(content[0]['tables'])

print()
print()
text = content[0]['text']
text = text.replace('*', '')
text = text.replace('_', '')
text = re.sub(r'<br\s*/?>', ' ', text)
    
# 2. Chuẩn hóa các khoảng trắng ngang (space, tab) thành một dấu cách duy nhất
text = re.sub(r'[ \t]+', ' ', text)

# 3. Chuẩn hóa các dòng trống liên tiếp (3+ dòng mới) thành một dòng trống duy nhất (2 dòng mới)
text = re.sub(r'\n{3,}', '\n\n', text)
    
# 2. Consolidate multiple spaces, newlines, and tabs into a single space.

print(text)


