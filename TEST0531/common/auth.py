# common/auth.py
import logging
import re
from locust.exception import CatchResponseError

def login_user(locust_instance, username="renhangxi", password="your_password"):
    """
    【进阶版】Django 系统登录鉴权模块
    具备容错机制、CSRF 深度提取与严格的登录状态断言。
    """
    login_url = "/login/"
    csrf_token = ""

    # ==========================================
    # 步骤 1：安全获取 CSRF Token (带异常重试与解析兜底)
    # ==========================================
    try:
        # 发起 GET 请求以获取初始 Cookie 和页面
        with locust_instance.client.get(login_url, catch_response=True, timeout=10) as get_resp:
            if get_resp.status_code != 200:
                get_resp.failure(f"无法访问登录页，状态码: {get_resp.status_code}")
                return False

            # 策略 A：从 Cookie 中提取
            csrf_token = locust_instance.client.cookies.get("csrftoken", "")
            
            # 策略 B (兜底)：如果 Cookie 中没有，尝试从 HTML 表单的 hidden 字段中正则提取
            if not csrf_token:
                match = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', get_resp.text)
                if match:
                    csrf_token = match.group(1)
                    logging.info("已通过 HTML 解析出兜底 CSRF Token。")
                
            if not csrf_token:
                get_resp.failure("致命错误：无法在 Cookie 或 HTML 中找到 CSRF Token。")
                return False
                
            get_resp.success()
            
    except Exception as e:
        logging.error(f"登录前置请求发生异常: {str(e)}")
        return False

    # ==========================================
    # 步骤 2：构造高仿真的请求头与 Payload
    # ==========================================
    # 动态拼接完整的 Referer，防止 Django 的 CsrfViewMiddleware 拦截
    referer_url = f"{locust_instance.host.rstrip('/')}{login_url}"
    
    headers = {
        "X-CSRFToken": csrf_token,
        "Referer": referer_url,
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "Mozilla/5.0 (Locust Load Test Tool) AppleWebKit/537.36"
    }
    
    data = {
        "username": username,
        "password": password,
        "csrfmiddlewaretoken": csrf_token  # 很多 Django 原生表单强制要求 body 中也要有此字段
    }

    # ==========================================
    # 步骤 3：执行登录并进行严格的状态断言
    # ==========================================
    try:
        # Django 默认在登录成功后会进行 302 重定向，Locust 会自动跟随，最终返回 200
        # 因此不仅要看状态码，还要严格检查 cookie 或响应体中的标识
        with locust_instance.client.post(login_url, data=data, headers=headers, catch_response=True, timeout=15) as post_resp:
            
            # 检查是否成功获得了 Django 的用户会话标识
            has_session = "sessionid" in locust_instance.client.cookies
            
            # 检查响应体中是否包含常见的错误提示（防止返回 200 但其实密码错了）
            has_error_msg = "用户名或密码错误" in post_resp.text or "Invalid" in post_resp.text

            if (post_resp.status_code in [200, 302]) and has_session and not has_error_msg:
                logging.info(f"✅ 账号 [{username}] 鉴权成功，已获取有效 Session。")
                post_resp.success()
                return True
            else:
                error_reason = f"状态码:{post_resp.status_code}, 包含session:{has_session}"
                logging.error(f"❌ 账号 [{username}] 鉴权失败！详情: {error_reason}")
                post_resp.failure(f"鉴权失败: {error_reason}")
                return False
                
    except CatchResponseError as ce:
        logging.error(f"登录断言失败: {str(ce)}")
        return False
    except Exception as e:
        logging.error(f"登录 POST 请求发生网络异常: {str(e)}")
        return False