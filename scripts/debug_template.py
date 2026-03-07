
import os

file_path = r"d:\renhangxi_tiktok_bysj\templates\prediction\dashboard.html"
base_path = r"d:\renhangxi_tiktok_bysj\templates\base.html"

def check_file(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("File not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if '{% verbatim' in content:
        print("FOUND {% verbatim %}!")
    else:
        print("No {% verbatim %} found.")
        
    if '{{ theme }}' in content:
        print("FOUND {{ theme }} usage.")
    else:
        print("No {{ theme }} usage found.")
        
    # Check for unclosed tags or weird characters
    lines = content.splitlines()
    for i, line in enumerate(lines):
        if '{{ theme' in line and '}}' not in line:
            print(f"Suspicious line {i+1}: {line.strip()}")
            if i + 1 < len(lines):
                print(f"Next line: {lines[i+1].strip()}")

check_file(file_path)
check_file(base_path)
