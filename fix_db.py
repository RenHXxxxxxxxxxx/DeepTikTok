import os
import sys
import django

sys.path.append(os.path.join(os.path.dirname(__file__), 'renhangxi_tiktok_bysj'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')
django.setup()

from douyin_hangxi.models import Comment
from douyin_hangxi.views import extract_semantic_features, calculate_refined_sentiment, clean_text_nuclear

def run():
    comments = Comment.objects.all()
    count = 0
    updated = []
    
    for c in comments:
        cleaned_raw = clean_text_nuclear(c.content)
        target_text = extract_semantic_features(cleaned_raw)

        if not target_text.strip() or len(target_text.strip()) < 1:
            continue

        score, label = calculate_refined_sentiment(target_text)

        c.sentiment_score = float(score)
        c.sentiment_label = str(label)
        c.content_clean = target_text
        
        updated.append(c)
        count += 1
        
        if len(updated) >= 500:
            Comment.objects.bulk_update(updated, ['sentiment_score', 'sentiment_label', 'content_clean'])
            updated = []
            print(f"Updated {count} comments...")
            
    if updated:
        Comment.objects.bulk_update(updated, ['sentiment_score', 'sentiment_label', 'content_clean'])
        print(f"Updated {count} comments total.")

if __name__ == '__main__':
    run()
