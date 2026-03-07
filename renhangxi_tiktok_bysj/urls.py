from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    # 如果文件夹在内层，请尝试改为：
    path('', include('renhangxi_tiktok_bysj.douyin_hangxi.urls')),
]