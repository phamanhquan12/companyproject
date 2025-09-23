import pickle
import os
from IPython.display import display, Markdown
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed', 'luong.pkl')
print(path)
with open(path, 'rb+') as f:
    content = pickle.load(f)


print(display(Markdown(content[0].page_content)))
print(None + 1)