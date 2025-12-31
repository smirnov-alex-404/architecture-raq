import os
from json import load

input_folder = 'star_wars_articles'
output_folder = 'knowledge_base'

TERMS_MAP = load(open('terms_map.json'))



def replace_terms_in_text(text: str):
    for term, new_term in TERMS_MAP.items():
        text = text.replace(term, new_term)
    return text


def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            with open(input_path, 'r', encoding='utf-8') as f:
                original_content = f.read().lower()
            modified_content = replace_terms_in_text(original_content)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"Processed: {filename}")


if __name__ == '__main__':
    main()
