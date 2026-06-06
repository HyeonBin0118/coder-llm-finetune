import pathlib
import re

p = pathlib.Path(r'C:\Users\VisualS2\anaconda3\envs\finetune_env\Lib\site-packages\trl\chat_template_utils.py')
content = p.read_text(encoding='utf-8')

# read_text() 호출을 전부 read_text(encoding='utf-8')로 교체
content = re.sub(r'\.read_text\(\)', '.read_text(encoding="utf-8")', content)

p.write_text(content, encoding='utf-8')
print('패치 완료')