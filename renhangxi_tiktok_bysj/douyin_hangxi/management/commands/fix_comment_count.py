from django.core.management.base import BaseCommand
from django.db.models import Count
from douyin_hangxi.models import Comment, Video


class Command(BaseCommand):
    help = '*从Comment表重新统计评论数并更新Video表的comment_count字段*'

    def handle(self, *args, **options):
        # *从数据库统计每个视频的实际评论数*
        comment_stats = Comment.objects.values('video_id').annotate(
            actual_count=Count('comment_id')
        )

        # *转换为字典*
        actual_counts = {item['video_id']: item['actual_count'] for item in comment_stats}

        # *更新Video表中的comment_count字段*
        videos = Video.objects.all()
        updated_count = 0
        
        for video in videos:
            actual_count = actual_counts.get(video.video_id, 0)
            if video.comment_count != actual_count:
                self.stdout.write(
                    f"视频 {video.video_id}: 旧值={video.comment_count}, 新值={actual_count}"
                )
                video.comment_count = actual_count
                video.save()
                updated_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f'\n评论数量已从数据库重新统计并更新,共更新 {updated_count} 条记录'
            )
        )
