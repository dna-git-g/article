import argparse
import os
import re

def replace_citations(match):
    new_str = ""
    citations = []

    for citation in match.group(1).split(","):
        citations.append(f"@{citation.strip()}")
    
    return f"[{'; '.join(citations)}]"

parser = argparse.ArgumentParser()

parser.add_argument("file_path")

args = parser.parse_args()

if not os.path.exists(args.file_path):
    print(f"Файл '{args.file_path}' не найден.")
    os._exit(0)

with open(args.file_path, "r+") as file:
    content = file.read()
    text_replaced = re.sub(r"\\citep{0,1}\{([^}]+)\}", replace_citations, content)
    file.seek(0)
    file.truncate()
    file.write(text_replaced)