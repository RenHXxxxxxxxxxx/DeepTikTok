from django.urls import path
from django.contrib.auth import views as auth_views
# 因为此文件在 douyin_hangxi 文件夹内，所以 from . 可以正确找到同目录的 views.py
from . import views

urlpatterns = [
    # 首页看板
    path('', views.dashboard, name='dashboard'),

    # 资产精炼工厂与仓库管理
    path('data/warehouse/', views.data_warehouse, name='data_warehouse'),
    path('api/import_data/', views.run_clean_data_api, name='import_data_api'),
    path('api/launch_spider/', views.launch_spider_api, name='launch_spider_api'),
    path('api/launch_comment_only/', views.launch_comment_only_api, name='launch_comment_only_api'),
    path('api/spider_status/', views.get_spider_status_api, name='get_spider_status_api'),
    path('api/switch_theme/', views.switch_theme, name='switch_theme'),
    path('api/delete_theme/', views.delete_theme, name='delete_theme'),

    # 数据明细列表
    path('data/videos/', views.video_list, name='video_list'),
    path('data/comments/', views.comment_list, name='comment_list'),

    # 可视化洞察图表
    path('charts/user/', views.chart_user, name='chart_user'),
    path('charts/content/', views.chart_content, name='chart_content'),
    path('charts/sentiment/', views.chart_sentiment, name='chart_sentiment'),

    # AI 爆款预测实验室
    path('predict/', views.predict_page, name='predict_page'),
    path('predict/api/', views.predict_api, name='predict_api'),

    # 账户认证系统
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('register/', views.register, name='register'),

    path('api/recalculate_sentiment/', views.recalculate_sentiment_api, name='recalculate_sentiment_api'),
    # AI 分析队列状态查询接口
    path('api/get_analysis_status/', views.get_analysis_status_api, name='analysis_status_api'),
    # 全局系统状态查询接口
    path('api/global-status/', views.get_global_status, name='global_status'),
    # 模型重训练 API (Retrain Model Trigger)
    path('api/retrain_model/', views.retrain_model_api, name='retrain_model_api'),
]