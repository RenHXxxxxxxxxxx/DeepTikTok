"""
*补充采集评论脚本*
*从已有视频 CSV 中读取视频 ID，只采集评论*
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.spyder_unified import run_comment_only_service

if __name__ == "__main__":
    print(" *启动评论补充采集...*")
    print(" *视频源: douyin_video_非遗手工艺.csv*")
    print(" *每视频最多采集: 50 条评论*")
    print("=" * 50)
    
    run_comment_only_service(
        video_csv_path=r"D:\renhangxi_tiktok_bysj\data\douyin_video_非遗手工艺.csv",
        theme_name="非遗手工艺",
        max_comments=50
    )
    
    print("\n *评论采集完成！*")

