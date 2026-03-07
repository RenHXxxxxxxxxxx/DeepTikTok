# 1. Setup Django Environment
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "renhangxi_tiktok_bysj.settings")
# django.setup()

import sys
import os
from django.conf import settings

# *MATCH VIEWS.PY IMPORT STRATEGY*
if os.path.join(settings.BASE_DIR, 'data') not in sys.path:
    sys.path.insert(0, os.path.join(settings.BASE_DIR, 'data'))
from spyder_unified import DouyinUnifiedPipeline

from renhangxi_tiktok_bysj.douyin_hangxi.views import get_global_status
from unittest.mock import MagicMock
import json


print("Starting Verification inside Shell...")

# Mock Request
request = MagicMock()
request.session = {'active_theme': 'TestTheme'}
request.method = 'GET'
request.user.is_authenticated = True

# Test 1: No Spider Running
print("\n[Test 1] Checking status when NO spider is running...")
# Ensure instance is None
DouyinUnifiedPipeline.instance = None
response = get_global_status(request)
data = json.loads(response.content)
print(f"Response: {data}")

if data['spider_running'] is False:
    print("PASS: spider_running is False")
else:
    print("FAIL: spider_running should be False")

# Test 2: Spider Running
print("\n[Test 2] Mocking Spider Instance...")
# Manually mock the instance to avoid launching browser
mock_spider = MagicMock()
mock_spider.progress_stats = {'current': 42, 'total': 100, 'stage': 'shell_testing'}
DouyinUnifiedPipeline.instance = mock_spider

print("Spider mocked. Checking status...")
response = get_global_status(request)
data = json.loads(response.content)
print(f"Response: {data}")

if data['spider_running'] is True:
    print("PASS: spider_running is True")
else:
    print("FAIL: spider_running should be True")

if data['spider_progress']['current'] == 42:
    print("PASS: Progress current is 42")
else:
    print(f"FAIL: Progress mismatch {data['spider_progress']}")

# Test 3: Stop Spider
print("\n[Test 3] Clearing Spider Instance...")
DouyinUnifiedPipeline.instance = None

print("Spider cleared. Checking status...")
response = get_global_status(request)
data = json.loads(response.content)

if data['spider_running'] is False:
    print("PASS: spider_running is False after clear")
else:
    print("FAIL: spider_running should be False after clear")

print("\nVerification Complete!")
