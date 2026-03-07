from django.contrib import admin
from django.utils.html import format_html
from .models import Video, Comment, AIModelConfig


# 1. 视频数据管理 (修复 E124 冲突版)
@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    # 列表页显示
    list_display = (
        'theme_label', 'short_desc', 'nickname', 'digg_visual',
        'brightness_tag', 'audio_bpm', 'cut_frequency',
        'error_visual', 'create_time'
    )

    # [关键修复] 显式指定详情页链接字段，避开 theme_label
    list_display_links = ('short_desc', 'nickname')

    # 允许在列表页直接修改主题名称
    list_editable = ('theme_label',)

    # 搜索与筛选
    search_fields = ('theme_label', 'desc', 'nickname', 'video_id')
    list_filter = ('theme_label', 'create_time', 'nickname')
    list_per_page = 20

    # 详情页分组布局
    fieldsets = (
        ('基础抓取与主题分类', {
            'fields': ('theme_label', 'video_id', 'nickname', 'desc', 'create_time', 'duration')
        }),
        ('多模态素材上传', {
            'description': '上传视频后点击保存，系统将自动调用 RTX 3060 进行内容分析',
            'fields': ('video_file',)
        }),
        ('核心特征数据 (由 AI 分析器自动填充)', {
            'fields': (
                ('visual_brightness', 'visual_saturation'),
                ('cut_frequency', 'audio_bpm')
            )
        }),
        ('互动统计', {
            'fields': (('digg_count', 'comment_count'), ('collect_count', 'share_count', 'download_count'),
                       'follower_count')
        }),
        ('预测结果对比', {
            'fields': ('predicted_digg_count', 'actual_vs_predicted_error')
        }),
    )

    def short_desc(self, obj):
        if not obj.desc:
            return "无描述"
        return obj.desc[:15] + "..." if len(obj.desc) > 15 else obj.desc

    short_desc.short_description = "视频简介"

    def digg_visual(self, obj):
        if obj.digg_count > 10000:
            color, weight = 'red', 'bold'
        elif obj.digg_count > 1000:
            color, weight = 'orange', 'normal'
        else:
            color, weight = 'gray', 'normal'
        return format_html('<span style="color:{}; font-weight:{};">{}</span>', color, weight, obj.digg_count)

    digg_visual.short_description = "点赞热度"

    def brightness_tag(self, obj):
        val = obj.visual_brightness
        if val is None: return "-"
        return format_html(
            '<div style="width:50px; text-align:center; background:#eee; border-left: 5px solid rgb({},{},{});">{}</div>',
            int(val), int(val), int(val), int(val)
        )

    brightness_tag.short_description = "画面亮度"

    def error_visual(self, obj):
        val = obj.actual_vs_predicted_error
        if val is None: return "未预测"
        color = "#28a745" if abs(val) < 20 else "#dc3545"
        return format_html(
            '<b style="color:{};">{}%</b>', color, val
        )

    error_visual.short_description = "预测偏差"


# 2. 评论舆情管理 (修复 E124 冲突版)
@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ('theme_label', 'short_content', 'nickname', 'ip_label', 'sentiment_colored', 'create_time')

    # [关键修复] 显式指定详情页链接字段，避开 theme_label
    list_display_links = ('short_content', 'nickname')

    list_editable = ('theme_label',)
    search_fields = ('content', 'nickname', 'theme_label')
    list_filter = ('theme_label', 'sentiment_label', 'ip_label')
    list_per_page = 20

    def short_content(self, obj):
        return obj.content[:30] + "..." if len(obj.content) > 30 else obj.content

    short_content.short_description = "评论内容"

    def sentiment_colored(self, obj):
        val = obj.sentiment_score
        if val >= 0.7:
            color, label = '#28a745', '正面'
        elif val <= 0.4:
            color, label = '#dc3545', '负面'
        else:
            color, label = '#ffc107', '中性'
        val_str = "{:.2f}".format(val)
        return format_html(
            '<div style="background-color:{}; color:white; padding:2px 6px; border-radius:4px; text-align:center; width:60px;">{}</div>',
            color, val_str
        )

    sentiment_colored.short_description = "情感评分"


# 3. AI 模型资产管理
@admin.register(AIModelConfig)
class AIModelConfigAdmin(admin.ModelAdmin):
    list_display = ('version_name', 'status_flag', 'create_time', 'description')
    readonly_fields = ('create_time',)

    def status_flag(self, obj):
        if obj.is_active:
            return format_html('<span style="color:green; font-weight:bold;">[当前运行中]</span>')
        return format_html('<span style="color:gray;">未激活</span>')

    status_flag.short_description = "状态"

    def save_model(self, request, obj, form, change):
        if obj.is_active:
            AIModelConfig.objects.filter(is_active=True).exclude(id=obj.id).update(is_active=False)
        super().save_model(request, obj, form, change)