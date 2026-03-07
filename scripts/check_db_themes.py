
import os
import sys
import django

# Setup Django environment
sys.path.append(os.getcwd())
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')
django.setup()

from renhangxi_tiktok_bysj.douyin_hangxi.models import Video

def check_themes():
    print("Checking themes...")
    themes = list(Video.objects.values_list('theme_label', flat=True).distinct())
    print(f"Found {len(themes)} themes.")
    for t in themes:
        print(f"Theme: '{t}'")
        if '{{' in t or '}}' in t:
            print(f"WARNING: Suspicious theme label found: {t}")

if __name__ == '__main__':
    check_themes()
