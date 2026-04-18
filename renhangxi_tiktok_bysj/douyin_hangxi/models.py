import os
from django.db import models
from .utils.video_analyzer import VideoContentAnalyzer


# 1. 视频信息表：支持多模态特征与多主题分类
class Video(models.Model):
    # 数据包主题标签，用于区分不同的离线数据来源
    theme_label = models.CharField(max_length=100, default='默认主题', db_index=True, verbose_name="所属主题包")

    # 基础抓取字段
    video_id = models.CharField(max_length=50, primary_key=True, verbose_name="视频ID")
    nickname = models.CharField(max_length=100, verbose_name="作者昵称")
    desc = models.TextField(verbose_name="视频描述", blank=True, null=True)
    create_time = models.DateTimeField(verbose_name="发布时间", blank=True, null=True)
    duration = models.CharField(max_length=20, verbose_name="视频时长", blank=True, null=True)

    # 互动统计数据
    follower_count = models.IntegerField(default=0, verbose_name="粉丝数")
    digg_count = models.IntegerField(default=0, verbose_name="点赞数")
    comment_count = models.IntegerField(default=0, verbose_name="评论数")
    collect_count = models.IntegerField(default=0, verbose_name="收藏数")
    share_count = models.IntegerField(default=0, verbose_name="分享数")
    download_count = models.IntegerField(default=0, verbose_name="下载数")

    # AI 分析提取的多模态特征 (由后台脚本或视频分析器填充)
    # 物理字段名：visual_brightness, visual_saturation, audio_bpm, cut_frequency
    visual_brightness = models.FloatField(null=True, blank=True, verbose_name="画面亮度")
    visual_saturation = models.FloatField(null=True, blank=True, verbose_name="画面饱和度")
    audio_bpm = models.IntegerField(null=True, blank=True, verbose_name="音频节奏BPM")
    cut_frequency = models.FloatField(null=True, blank=True, verbose_name="平均剪辑频率")

    # 视频素材文件
    video_file = models.FileField(upload_to='videos/%Y/%m/%d/', verbose_name="视频原文件", null=True, blank=True)

    # 预测相关字段
    predicted_digg_count = models.IntegerField(null=True, blank=True, verbose_name="预测点赞数")
    actual_vs_predicted_error = models.FloatField(null=True, blank=True, verbose_name="预测偏差率")

    # 异步处理状态跟踪字段
    # 语义: 0=待处理(Pending), 1=处理中(Processing), 2=已完成(Completed), -1=失败(Failed)
    analysis_status = models.IntegerField(default=0, db_index=True, verbose_name="AI分析状态")
    local_temp_path = models.CharField(max_length=255, null=True, blank=True, verbose_name="本地临时路径")

    class Meta:
        db_table = 'tb_video'
        verbose_name = '视频数据管理'
        verbose_name_plural = verbose_name


# 2. 评论舆情表
class Comment(models.Model):
    # 建立与视频的一对多关系，并支持主题标签
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='comments', verbose_name="所属视频")
    theme_label = models.CharField(max_length=100, default='默认主题', verbose_name="所属主题包")

    comment_id = models.CharField(max_length=50, primary_key=True, verbose_name="评论ID")
    nickname = models.CharField(max_length=100, verbose_name="用户昵称")
    content = models.TextField(verbose_name="原始评论内容")
    content_clean = models.TextField(verbose_name="清洗后内容")
    create_time = models.DateTimeField(verbose_name="评论时间")
    ip_label = models.CharField(max_length=50, verbose_name="IP属地")
    digg_count = models.IntegerField(default=0, verbose_name="点赞数")

    # 情感分析结果
    sentiment_score = models.FloatField(default=0.5, verbose_name="情感得分")
    sentiment_label = models.CharField(max_length=10, verbose_name="情感标签")

    hour = models.IntegerField(default=0, verbose_name="发布小时")
    text_len = models.IntegerField(default=0, verbose_name="评论长度")

    class Meta:
        db_table = 'tb_comment'
        verbose_name = '评论舆情管理'
        verbose_name_plural = verbose_name


# 3. AI 模型配置管理表
class AIModelConfig(models.Model):
    version_name = models.CharField(max_length=50, verbose_name="模型版本号")
    model_file = models.FileField(upload_to='ai_models/', verbose_name="模型权重文件")
    scaler_file = models.FileField(upload_to='ai_models/', verbose_name="数据标准化文件")
    is_active = models.BooleanField(default=False, verbose_name="当前使用中")
    description = models.TextField(blank=True, verbose_name="备注说明")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        db_table = 'tb_ai_model_config'
        verbose_name = 'AI预测模型管理'
        verbose_name_plural = verbose_name

# 4. 创作者配置表
from django.contrib.auth.models import User

class CreatorConfig(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='config', verbose_name="关联用户")
    llm_api_key = models.CharField(max_length=255, blank=True, null=True, verbose_name="大模型API Key")
    # 建议在生产环境中对llm_api_key进行加密存储，如使用Fernet等加密库
    llm_model_name = models.CharField(max_length=50, default="ernie-4.0-8k", verbose_name="大模型名称")
    
    class Meta:
        db_table = 'tb_creator_config'
        verbose_name = '创作者配置'
        verbose_name_plural = verbose_name
