import os
import sys
import django
from django.test import TestCase

# 将项目根目录加入 python 搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'renhangxi_tiktok_bysj.settings')
django.setup()

from renhangxi_tiktok_bysj.douyin_hangxi.models import Video

class TestDjangoDatabase(TestCase):
    
    def setUp(self):
        # 测试前准备：插入一条假数据以验证 DB 连通性
        self.test_video = Video.objects.create(
            video_id="test_aweme_123", # 注意：原来写的是 aweme_id，现在是 video_id
            desc="Test Description",
            nickname="author_123", # 原来写的是 author_id, 现在似乎是 nickname
            digg_count=1000,
            comment_count=50,
            share_count=10
        )
        
    def test_database_connection(self):
        # 测试是否能正常读写 sqlite3 数据库
        video = Video.objects.get(video_id="test_aweme_123")
        self.assertEqual(video.digg_count, 1000)
        self.assertEqual(video.desc, "Test Description")

    def tearDown(self):
        # 测试后清理假数据
        self.test_video.delete()
