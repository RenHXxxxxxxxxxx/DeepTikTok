import os
import sys
import django
import json
from unittest.mock import MagicMock

# 1. Setup Django Environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "renhangxi_tiktok_bysj.settings")
django.setup()

from data.spyder_unified import DouyinUnifiedPipeline
from renhangxi_tiktok_bysj.douyin_hangxi.views import get_global_status

def verify():
    print("🚀 Starting Verification...")

    # Mock Request
    request = MagicMock()
    request.session = {'active_theme': 'TestTheme'}
    request.method = 'GET'
    request.user.is_authenticated = True # Mock login

    # Test 1: No Spider Running
    print("\n[Test 1] Checking status when NO spider is running...")
    response = get_global_status(request)
    data = json.loads(response.content)
    print(f"Response: {data}")
    
    if data['spider_running'] is False:
        print("✅ PASS: spider_running is False")
    else:
        print("❌ FAIL: spider_running should be False")

    # Test 2: Spider Running
    print("\n[Test 2] Starting Spider Instance...")
    try:
        spider = DouyinUnifiedPipeline()
        # Simulate progress
        spider.progress_stats = {'current': 5, 'total': 10, 'stage': 'testing'}
        
        print("Spider started. Checking status...")
        response = get_global_status(request)
        data = json.loads(response.content)
        print(f"Response: {data}")
        
        if data['spider_running'] is True:
            print("✅ PASS: spider_running is True")
        else:
            print("❌ FAIL: spider_running should be True")
            
        if data['spider_progress']['current'] == 5:
            print("✅ PASS: Progress current is 5")
        else:
            print(f"❌ FAIL: Progress mismatch {data['spider_progress']}")

        # Test 3: Stop Spider
        print("\n[Test 3] Closing Spider...")
        spider.close()
        
        print("Spider closed. Checking status...")
        response = get_global_status(request)
        data = json.loads(response.content)
        print(f"Response: {data}")
        
        if data['spider_running'] is False:
            print("✅ PASS: spider_running is False after close")
        else:
            print("❌ FAIL: spider_running should be False after close")

    except Exception as e:
        print(f"❌ Exception during test: {e}")
        try:
            spider.close()
        except: pass

if __name__ == "__main__":
    verify()
