import time
import csv
import os
import sys
import requests
import urllib3
import re
import random
import queue
import threading
import logging
import pandas as pd
from tqdm import tqdm
from DrissionPage import ChromiumPage, ChromiumOptions

# *[Async Persistence] 配置日志记录器*
logger = logging.getLogger(__name__)

# *================================================================*
# *1. 全局配置与入口 (Global Config & Entry)*
# *================================================================*

# *项目根目录自动定位*
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# *[Privacy Preserving] 全球 UA 资源池 (User-Agent Global Pool)*
GLOBAL_UA_POOL = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
    'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
]

# *[Rate Limiting] 全局限流与熔断状态*
GLOBAL_RATE_LIMIT_STRIKES = 0
GLOBAL_SCROLL_COUNT = 0

def get_random_ua():
    return random.choice(GLOBAL_UA_POOL)

def get_gaussian_delay():
    """*多模态高斯延迟模型 (Multi-modal Gaussian Delay Model)*"""
    global GLOBAL_SCROLL_COUNT
    GLOBAL_SCROLL_COUNT += 1
    if GLOBAL_SCROLL_COUNT % 50 == 0:
        # *深度阅读模式*
        delay = random.gauss(20, 5)
    else:
        # *正常滚动模式*
        delay = random.gauss(3, 0.5)
    return max(1.0, delay)

# *屏蔽 SSL 警告*
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# *================================================================*
# *[Fail-Fast] 网络超时配置 (Network Timeout Configuration)*
# *================================================================*
NETWORK_CONFIG = {
    'page_load_timeout': 15,      # *页面加载超时 (秒)*
    'listen_wait_timeout': 10,    # *监听器等待超时 (秒) - 抖音搜索 API 首次加载较慢*
    'max_consecutive_timeouts': 3, # *连续超时阈值，触发浏览器恢复*
    'recovery_wait': 2,           # *恢复后等待时间 (秒)*
    'max_zero_growth_cycles': 5,  # *[Fix 1] 连续零增长阈值 (Zero-Growth Sentinel)*
    'max_empty_responses': 10,    # *[Fix 3] 最大连续空响应阈值*
}

# *================================================================*
# *[DOM Sentinels] 页面探测关键字 (Multilingual DOM Text Sentinel)*
# *================================================================*
END_OF_FEED_KEYWORDS = ['没有更多了', '暂时没有更多了', '已展示所有结果', 'End of results']

# *全局 CSV IO 锁，保证数据不出现错行*
CSV_FILE_LOCK = threading.Lock()

# *导入统一持久化管理器*
sys.path.append(PROJECT_ROOT)
try:
    from services.data_manager import UnifiedPersistenceManager
except ImportError:
    try:
        from data_manager import UnifiedPersistenceManager
    except ImportError:
        print("⚠️ Database Manager path still disconnected.")


class DouyinUnifiedPipeline:
    instance = None  # *[Global Access] 单例引用*


    def __init__(self):
        # *[Global Access] 注册延迟到初始化成功后*
        self.progress_stats = {'current': 0, 'total': 0, 'stage': 'init'}
        
        # *初始化浏览器引擎 (单例)*
        try:
            # *================================================================*
            # *[资源优化] 浏览器配置 (Browser Resource Optimization)*
            # *================================================================*
            co = ChromiumOptions()
            
            # *自动寻找系统浏览器，无需手动定位路径*
            # *将所有登录信息、缓存保存在项目内部的 crawler/browser_files 下*
            data_path = os.path.join(PROJECT_ROOT, 'crawler', 'browser_files')
            co.set_paths(user_data_path=data_path)
            
            # *[Speed + Bandwidth] 阻止图片请求，减少网络开销*
            # co.no_imgs()
            
            # *临时开启图片加载，确保二维码可见*
            co.set_pref('profile.managed_default_content_settings.images', 1) 
            co.set_argument('--disable-blink-features=AutomationControlled')
            co.set_argument(f'--user-agent={get_random_ua()}')
            
            # *[CPU Optimization] 静音播放，避免音频解码消耗 CPU*
            co.mute(True)
            
            # *[Stability] 长时间任务稳定性增强参数*
            co.set_argument('--no-sandbox')        # *容器/虚拟化环境兼容*
            co.set_argument('--disable-gpu')       # *禁用 GPU 加速，避免驱动问题*
            
            # *================================================================*
            # *[Standardization Patch] 批量任务环境标准化 (Batch Environment Standardization)*
            # *================================================================*
            # *Audit 结论: 批量任务通过 Django 背景线程 (threading.Thread) 异步触发，*
            # *此时 OS 会将 Chromium 分配在无显式焦点的后台，使其以 "headless-like" 的默认极小视口 (如 800x600) 启动。*
            # *这会触发目标站点的响应式布局 (Responsive UI)，导致 DOM 结构与单任务前台标准视口不同。*
            # *修复: 强制固定视口为 1920x1080 并最大化，确保无论前后台启动，DOM 树 100% 绝对一致。*
            co.set_argument('--window-size=1920,1080')
            co.set_argument('--start-maximized')
            
            # *[CRITICAL] 显式保持 Headful 模式 (安全考量，不使用 headless)*
            # *DrissionPage 默认即为 Headful，此处无需额外设置*
            
            # *使用优化后的配置初始化浏览器*
            self.page = ChromiumPage(addr_or_opts=co)
            
            # *[Login Intercept] 更加智能的登录检测*
            self.page.get('https://www.douyin.com/')
            # *检查是否真的需要登录*
            if self.page.ele('text:登录', timeout=2):
                print("⏳ *[Login Intercept] 检测到未登录，请在浏览器中完成操作...*")
                
                # *循环检查状态，而非死等单一点*
                for _ in range(120): # *最多等 10 分钟 (120 * 5s)*
                    # *标志 A: 登录按钮消失了*
                    login_btn_gone = not self.page.ele('text:登录', timeout=1)
                    # *标志 B: 出现了“发布视频”按钮 (data-e2e="upload-video") 或用户头像*
                    logged_in_marker = self.page.ele('xpath://a[contains(@href, "/upload")]', timeout=1)
                    
                    if login_btn_gone or logged_in_marker:
                        print("✅ *[Login Intercept] 检测到登录态已激活！*")
                        break
                    
                    time.sleep(2)
                    # *每隔 15 秒如果还没动静，尝试刷新一下页面辅助同步*
                    if _ % 5 == 0 and _ > 0:
                        print("🔄 *[Login Intercept] 正在刷新页面以同步登录状态...*")
                        self.page.refresh()
            else:
                print("✅ *[Login Intercept] 已处于登录状态，直接跳过。*")
                
            print("✅ *[Login Intercept] 登录检测完成！您的登录信息已本地化存储。*")
            try:
                self.page.set.imgs(False) # *动态关闭图片，节省带宽*
            except:
                pass # *如果库版本不支持动态切换，不影响主流程*

            # *[Fail-Fast] 配置页面加载超时 (防止无限阻塞)*
            try:
                self.page.set.timeouts(page_load=NETWORK_CONFIG['page_load_timeout'] * 1000)  # *毫秒*
                print(f"⏱️ *[Fail-Fast] 页面超时已配置: {NETWORK_CONFIG['page_load_timeout']}s*")
            except Exception as timeout_cfg_err:
                print(f"⚠️ *[Fail-Fast] 超时配置失败 (可忽略): {timeout_cfg_err}*")
        except Exception as e:
            print(f"❌ *浏览器初始化失败: {e}*")
            # *[Zombie Singleton Fix] 初始化失败时清除单例引用*
            DouyinUnifiedPipeline.instance = None
            raise e
        
        # *全局去重集合*
        self.existing_video_ids = set()
        self.existing_comment_ids = set()
        
        # *当前任务上下文*
        self.current_theme = ""
        self.video_save_path = ""
        self.comment_save_path = ""
        self.media_dir = ""
        
        # *[Fail-Fast] 连续超时计数器 (用于熔断恢复)*
        self._consecutive_timeout_count = 0

        # *初始化持久化管理器*
        try:
            self.db_manager = UnifiedPersistenceManager()
        except:
            self.db_manager = None

        # *================================================================*
        # *[Async Persistence] 数据库异步写入队列 (Producer-Consumer Pattern)*
        # *================================================================*
        # *Rationale: maxsize=100 防止数据库过慢导致内存泄漏*
        self._db_queue = queue.Queue(maxsize=100)
        
        # *启动后台工作线程 (daemon=True 确保主程序退出时自动清理)*
        self._db_thread = threading.Thread(target=self._async_db_worker, daemon=True)
        self._db_thread.start()
        print("🔄 *[Async Persistence] 数据库异步写入线程已启动*")

        # *[Delayed Registration] 初始化完全成功后才注册单例*
        DouyinUnifiedPipeline.instance = self

    # *================================================================*
    # *[Async Persistence] 数据库后台工作线程 (Consumer Thread)*
    # *================================================================*
    
    def _async_db_worker(self):
        """
        *异步数据库写入工作线程 (Consumer)*
        *从队列中获取数据并批量写入数据库，与爬虫主循环解耦*
        """
        while True:
            try:
                # *阻塞等待队列数据 (data_buffer, theme)*
                data_buffer, theme = self._db_queue.get()
                
                # *执行数据库写入 (同步阻塞，但在独立线程中)*
                if self.db_manager:
                    try:
                        self.db_manager.save_comment_batch(data_buffer, theme)
                    except Exception as db_err:
                        # *[Robust] 捕获所有数据库错误，防止线程崩溃*
                        logger.error(f"*[Async DB] 数据库写入失败: {db_err}*")
                        print(f"❌ *[Async DB] 数据库写入失败: {db_err}*")
                
                # *标记任务完成 (用于 queue.join() 同步)*
                self._db_queue.task_done()
                
            except Exception as e:
                # *[Safety Net] 捕获队列操作异常，保持线程存活*
                logger.error(f"*[Async DB] Worker 异常: {e}*")
                print(f"⚠️ *[Async DB] Worker 异常 (已容错): {e}*")

    # *================================================================*
    # *[Fail-Fast] 网络超时恢复引擎 (Network Timeout Recovery)*
    # *================================================================*
    
    def _handle_network_timeout(self, context="unknown"):
        """
        *处理网络超时，触发恢复机制*
        *当连续超时次数超过阈值时，刷新页面或重启浏览器*
        :param context: *超时发生的上下文描述*
        :return: *True=已恢复可继续, False=恢复失败*
        """
        self._consecutive_timeout_count += 1
        print(f"⚠️ *[Fail-Fast] 网络超时 ({context}), 连续超时: {self._consecutive_timeout_count}/{NETWORK_CONFIG['max_consecutive_timeouts']}*")
        
        if self._consecutive_timeout_count >= NETWORK_CONFIG['max_consecutive_timeouts']:
            print(f"🔄 *[Fail-Fast] 触发恢复机制: 连续超时达阈值*")
            try:
                # *策略1: 尝试刷新当前页面*
                self.page.refresh()
                time.sleep(NETWORK_CONFIG['recovery_wait'])
                self._consecutive_timeout_count = 0  # *重置计数*
                print(f"✅ *[Fail-Fast] 页面刷新成功，继续执行*")
                return True
            except Exception as refresh_err:
                print(f"⚠️ *[Fail-Fast] 刷新失败: {refresh_err}, 尝试导航到空白页*")
                try:
                    # *策略2: 导航到空白页解除阻塞*
                    self.page.get('about:blank')
                    time.sleep(1)
                    self._consecutive_timeout_count = 0
                    print(f"✅ *[Fail-Fast] 空白页导航成功，继续执行*")
                    return True
                except Exception as nav_err:
                    print(f"❌ *[Fail-Fast] 恢复失败: {nav_err}*")
                    return False
        return True
    
    def _reset_timeout_counter(self):
        """*网络请求成功时重置超时计数器*"""
        self._consecutive_timeout_count = 0


    # *================================================================*
    # *2. 基础工具链 (Utils)*
    # *================================================================*
    
    def _normalize_desc(self, desc):
        # *清洗文本描述：压缩空格与换行，并替换半角逗号为全角*
        if not desc: return ""
        desc = desc.replace('\n', ' ').replace('\r', ' ').replace(',', '，')
        return re.sub(r'\s+', ' ', desc).strip()

    def _get_time_str(self, timestamp):
        # *转换 UNIX 时间戳为标准时间*
        if not timestamp: return "未知时间"
        try:
            return time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(int(timestamp)))
        except:
            return str(timestamp)

    def _format_duration(self, ms):
        # *转换毫秒时长为 MM:SS*
        if not ms: return "00:00"
        try:
            sec = int(ms) // 1000
            return "{:02d}:{:02d}".format(sec // 60, sec % 60)
        except:
            return "00:00"

    def _ensure_directories(self, theme):
        # *动态构建目录结构*
        data_dir = os.path.join(PROJECT_ROOT, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # *[Producer Mode] 持久化待处理视频目录*
        # *不再使用临时目录，Consumer 进程会从此处读取并处理*
        pending_dir = os.path.join(PROJECT_ROOT, "media", "pending_videos")
        if not os.path.exists(pending_dir):
            os.makedirs(pending_dir)
        self.pending_dir = pending_dir
        
        # *设置 pending_dir 后立即执行垃圾回收，清理孤立片段*
        try:
            import glob
            part_files = glob.glob(os.path.join(self.pending_dir, "*.part"))
            for pf in part_files:
                try:
                    os.remove(pf)
                    print(f"🧹 [Garbage Collection] Removed orphaned file: {os.path.basename(pf)}")
                except Exception as e:
                    print(f"⚠️ [Garbage Collection] Could not remove {os.path.basename(pf)}: {e}")
        except Exception as e:
            print(f"⚠️ [Garbage Collection] Setup error: {e}")
            
        # *主题专属视频存放目录 (保留用于其他用途)*
        theme_video_dir = os.path.join(PROJECT_ROOT, "media", "videos", theme)
        if not os.path.exists(theme_video_dir):
            os.makedirs(theme_video_dir)
            
        return theme_video_dir

    def _save_to_csv(self, data_list, filename, headers):
        # *通用的 CSV 追加写入逻辑*
        if not data_list: return
        file_exists = os.path.isfile(filename)
        try:
            with CSV_FILE_LOCK:
                with open(filename, 'a', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_NONNUMERIC)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(data_list)
                    
                    # *[CRITICAL FIX] 强制操作系统将缓冲区落盘，防止崩溃时数据丢失*
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            print(f"❌ *持久化失败: {e}*")

    # *================================================================*
    # *3. 模块 A: 视频采集与下载 (Video Harvester)*
    # *================================================================*

    def _download_video_stream(self, url, video_id):
        # *[Producer Mode] 流式下载引擎：原子写入策略 (Atomic Write Strategy)*
        # *修复竞态条件：先下载到 .part 临时文件，完成后原子重命名*
        if not url: return ""
        
        # *路径定义 (Path Definitions)*
        file_name = f"{video_id}.mp4"
        final_path = os.path.join(self.pending_dir, file_name)
        temp_path = os.path.join(self.pending_dir, f"{video_id}.mp4.part")
        
        # *幂等性检查 (Idempotency Check)*
        # *若最终文件已存在，直接返回路径（断点续传/去重）*
        if os.path.exists(final_path):
            try:
                existing_size_mb = os.path.getsize(final_path) / (1024 * 1024)
                print(f"[Crawler] ♻️  Already exists: {final_path} (Size: {existing_size_mb:.2f}MB)")
            except:
                pass
            return final_path

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.douyin.com/',
        }
        
        # *[Verbose Logging] 下载开始*
        print(f"[Crawler] 📥 Started download: {video_id}")

        global GLOBAL_RATE_LIMIT_STRIKES
        max_wait = 300 # *最大退避 5 分钟*
        
        # *Exponential Backoff Retry Loop*
        for retry_n in range(4):
            # *Circuit Breaker Check*
            if GLOBAL_RATE_LIMIT_STRIKES >= 3:
                print("🛑 *[Circuit Breaker] 连续 3 次 429 触发全局冬眠 15 分钟*")
                time.sleep(15 * 60)
                GLOBAL_RATE_LIMIT_STRIKES = 0 # *重置*
                
            headers['User-Agent'] = get_random_ua()
            
            try:
                with requests.get(url, headers=headers, stream=True, timeout=20, verify=False) as r:
                    if r.status_code == 200:
                        GLOBAL_RATE_LIMIT_STRIKES = 0 # *请求成功重置熔断计数*
                        downloaded_size = 0
                        MAX_SIZE = 300 * 1024 * 1024  # *300MB 熔断阈值*

                        # *写入临时文件 (Write to temp_path, NOT final_path)*
                        with open(temp_path, 'wb') as f:
                            try:
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    if chunk: 
                                        f.write(chunk)
                                        downloaded_size += len(chunk)
                                        # *300MB 熔断保护：优雅截断*
                                        if downloaded_size > MAX_SIZE:
                                            try:
                                                size_mb = downloaded_size / (1024 * 1024)
                                                print(f"✂️ *[加速] 视频过大 ({size_mb:.2f}MB > 300MB)，触发截断下载*")
                                            except:
                                                print(f"✂️ *[加速] 视频过大 (>300MB)，触发截断下载*")
                                            break
                            except Exception as write_err:
                                print(f"⚠️ *[下载] 写入流异常 (已容错): {write_err}*")
                                raise write_err  # *重新抛出以触发 finally 清理*
                        break # *退出退避重试循环*
                    elif r.status_code in [429, 503]:
                        if r.status_code == 429:
                            GLOBAL_RATE_LIMIT_STRIKES += 1
                        # *指数退避算法 (Exponential Backoff)*
                        jitter = random.uniform(0.1, 1.0)
                        t_wait = min((2 ** retry_n) + jitter, max_wait)
                        print(f"⚠️ *[Rate Limit] HTTP {r.status_code}，触发指数退避: 休眠 {t_wait:.2f}s*")
                        time.sleep(t_wait)
                    else:
                        print(f"[Crawler] ❌ Download Failed: {url} (HTTP {r.status_code})")
                        break # *其他状态码不重试*
            except Exception as e:
                # *[Verbose Logging] 下载异常*
                print(f"[Crawler] ❌ Request Failed: {url} ({e})")
                time.sleep(2) # *普通失败稍微等待*
                if retry_n == 3:
                     # *如果到达重试上限都没成功，直接跳过*
                     break

        if not os.path.exists(temp_path):
            return "" # *未能成功下载则退出*

        try:
            # *=== [究极优化] 先验证 .part 再提交 (Validate Before Commit) ===*
            # *确保 Worker 永远看不到任何坏文件，消除竞态条件窗口*
            import cv2
            cap = cv2.VideoCapture(temp_path)  # *验证 .part 文件*
            
            if not cap.isOpened():
                # *文件损坏：无法被 OpenCV 打开*
                cap.release()
                print(f"[Crawler] ⚠️ CORRUPT FILE DETECTED (pre-commit): {video_id}")
                print(f"[Crawler] 🗑️  Deleting corrupted .part file: {temp_path}")
                try:
                    os.remove(temp_path)
                except Exception as del_err:
                    print(f"[Crawler] ❌ Failed to delete corrupt file: {del_err}")
                return ""  # *返回空路径，不入库、不采集评论*
            
            # *额外检查：帧数和时长是否有效*
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            if frame_count <= 0 or fps <= 0:
                print(f"[Crawler] ⚠️ INVALID VIDEO (frames={frame_count}, fps={fps}): {video_id}")
                print(f"[Crawler] 🗑️  Deleting invalid .part file: {temp_path}")
                try:
                    os.remove(temp_path)
                except Exception as del_err:
                    print(f"[Crawler] ❌ Failed to delete invalid file: {del_err}")
                return ""
            
            # *验证通过，计算时长*
            duration_sec = frame_count / fps
            
            # *=== [Duration Circuit Breaker] 超长视频熔断 ===*
            # *防止超长视频 (>15分钟) 进入持久化层导致 AI Worker 超时*
            MAX_VIDEO_DURATION = 15 * 60  # *900 秒 = 15 分钟*
            if duration_sec > MAX_VIDEO_DURATION:
                print(f"[Crawler] ⚠️ Video too long ({duration_sec:.1f}s > {MAX_VIDEO_DURATION}s). Deleting...")
                try:
                    os.remove(temp_path)
                    print(f"[Crawler] 🗑️  Deleted oversized video: {video_id}")
                except Exception as del_err:
                    print(f"[Crawler] ❌ Failed to delete oversized video: {del_err}")
                return ""  # *返回空路径，跳过该视频*
            # *=== 熔断检查结束 ===*
            
            # *原子提交 (Atomic Commit)*
            # *只有验证通过的文件才会被重命名为 .mp4，Worker 永远看不到坏文件*
            os.rename(temp_path, final_path)

            # *[Verbose Logging] 下载成功*
            try:
                final_size_mb = downloaded_size / (1024 * 1024)
                print(f"[Crawler] ✅ Verified & Committed: {final_path} (Size: {final_size_mb:.2f}MB, Duration: {duration_sec:.1f}s)")
            except:
                print(f"[Crawler] ✅ Verified & Committed: {final_path}")
            
            return final_path  # *返回绝对路径供 Consumer 使用*
            
        except Exception as e:
            # *[Verbose Logging] 下载异常*
            print(f"[Crawler] ❌ Download Failed Post-Processing: {video_id} ({e})")
        finally:
            # *清理垃圾 (Cleanup & Error Handling)*
            # *若临时文件仍存在（下载失败或崩溃），删除以防止垃圾堆积*
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    print(f"[Crawler] 🧹 Cleaned up orphaned temp file: {temp_path}")
                except Exception as cleanup_err:
                    print(f"⚠️ *[清理] 删除临时文件失败: {cleanup_err}*")
        
        return ""

    # *[Producer Mode] _analyze_and_cleanup 方法已移除*
    # *分析逻辑由独立的 Consumer 进程异步处理*

    def _parse_video_packet(self, item):
        # *解析搜素接口返回的 JSON 包*
        try:
            info = item.get('aweme_info', item)

            # *[新增] 纯视频过滤器 (Video-Only Filter)*
            # *1. 检查 aweme_type: 2=图文, 68=多图PPT. 仅保留视频(0, 4, 51, 61等)*
            aweme_type = info.get('aweme_type', -1)
            if aweme_type in [2, 68]:
                print(f"⚠️ *跳过图文/PPT内容: {info.get('aweme_id')} (Type: {aweme_type})*")
                return None
            
            # *2. 二次检查 images 字段 (防止 aweme_type 漏判)*
            if info.get('images') and len(info.get('images')) > 0:
                print(f"⚠️ *跳过包含图片的内容: {info.get('aweme_id')}*")
                return None

            vid = str(info.get('aweme_id'))
            desc = self._normalize_desc(info.get('desc', ''))
            
            # *[Data Integrity] Ensure Exhaustion Sentinel is not captured*
            for kw in END_OF_FEED_KEYWORDS:
                if kw in desc or kw in str(info.get('author', {}).get('nickname', '')):
                    return None

            # *内存去重*
            if vid in self.existing_video_ids:
                return None

            # *3. 检查视频播放地址*
            video_url = info.get('video', {}).get('play_addr', {}).get('url_list', [None])[0]
            if not video_url:
                print(f"⚠️ *跳过无播放地址的内容: {vid}*")
                return None

            # *[Producer Mode] 触发下载到持久化目录*
            local_path = self._download_video_stream(video_url, vid)
            if not local_path: return None

            self.existing_video_ids.add(vid)
            
            # *[Producer Mode] AI 特征使用占位符值，由 Consumer 异步填充*
            video_data = {
                '视频ID': vid,
                '用户名': info.get('author', {}).get('nickname'),
                '粉丝数量': info.get('author', {}).get('follower_count'),
                '视频描述': desc,
                '发表时间': self._get_time_str(info.get('create_time')),
                '视频时长': self._format_duration(info.get('video', {}).get('duration')),
                '点赞数量': info.get('statistics', {}).get('digg_count'),
                '收藏数量': info.get('statistics', {}).get('collect_count'),
                '评论数量': info.get('statistics', {}).get('comment_count'),
                '分享数量': info.get('statistics', {}).get('share_count'),
                '下载数量': info.get('statistics', {}).get('download_count'),
                'visual_brightness': 0,
                'visual_saturation': 0,
                'cut_frequency': 0,
                'audio_bpm': 0,
                'local_temp_path': local_path
            }

            return video_data
        except:
            return None

    def run_video_crawler(self, task_config, progress_callback=None):
        # *绑定计数器变量，避免 UnboundLocalError*
        empty_response_count = 0
        consecutive_zero_growth_count = 0
        session_collected = 0
        stagnant_bottom_count = 0
        last_page_height = 0
        last_dom_card_count = 0

        # *视频采集主逻辑*
        keyword = task_config['keyword']
        limit = task_config['max_videos']
        
        # *[Status] 初始化进度状态*
        self.progress_stats = {'current': len(self.existing_video_ids), 'total': limit, 'stage': 'video_crawling'}
        
        # *================================================================*
        # *[Dedup] 数据库级去重：从 MySQL 加载已存在的视频 ID*
        # *确保即使 CSV 缺失或不完整，也不会重复下载数据库中已有的视频*
        # *================================================================*
        if self.db_manager:
            try:
                from renhangxi_tiktok_bysj.douyin_hangxi.models import Video
                # *从数据库加载所有视频 ID 到内存集合，实现 O(1) 查找*
                db_ids = list(Video.objects.values_list('video_id', flat=True))
                # *转换为字符串以确保与 API 返回数据匹配*
                self.existing_video_ids.update(str(vid) for vid in db_ids)
                print(f"✅ *[Dedup] 已从数据库加载 {len(db_ids)} 个已存在的视频 ID (合并 CSV + DB)*")
            except Exception as db_load_err:
                print(f"⚠️ *[Dedup] 数据库加载失败 (已容错): {db_load_err}*")
        
        # *[Strategy B] 记录初始采集数量，用于判断是否零收益*
        initial_collected_count = len(self.existing_video_ids)
        zero_yield_retry_count = 0
        
        # *================================================================*
        # *[CRITICAL FIX] 本次会话采集计数器 (Session-Specific Counter)*
        # *BUG 原因: 使用 len(existing_video_ids) 包含历史数据 (如 1418 条)*
        # *导致 1418 < 100 为 False，循环永远不执行*
        # *修复: 单独追踪本次会话新采集的数量*
        # *================================================================*
        # session_collected 已移至方法开头

        # *[进度追踪] 初始化起始时间*
        start_time = time.time()
        
        print(f"\n🎬 *[阶段一] 启动视频采集: {keyword} | 目标: {limit} 条 (已有历史: {initial_collected_count} 条)*")
        
        # *================================================================*
        # *[Strategy A] Pre-emptive Listening (The "Ear First" Rule)*
        # *Action: Move listen.start() BEFORE page.get()*
        # *Why: Ensure we capture network traffic from the very first millisecond*
        # *================================================================*
        self.page.listen.start('aweme/v1/web/search/')
        print(f"👂 *[Strategy A] 网络监听已提前启动 (Pre-emptive Listening)*")

        import urllib.parse
        
        # *[Fail-Fast] 页面导航包裹在 try-except 中*
        encoded_keyword = urllib.parse.quote(keyword.strip())
        search_url = f"https://www.douyin.com/search/{encoded_keyword}?type=video"
        try:
            self.page.get(search_url)
            time.sleep(2)  # *等待可能发生的 WAF 重定向*
            
            # *=== [Human Mimicry Escape Guard] WAF 逃逸补丁 ===*
            try:
                if 'so.douyin.com' in self.page.url:
                    self.page.get('https://www.douyin.com/')
                    time.sleep(2)  # *等待安全主页加载*
                    
                    # *第一次尝试定位物理搜索框，失败则回退至通用 input*
                    search_box = self.page.ele('css:input[data-e2e="searchbar-input"]')
                    if not search_box:
                        search_box = self.page.ele('css:input[type="text"]')
                    
                    if search_box:
                        search_box.clear()
                        search_box.input(keyword)  # *输入未转义的原始 keyword (模拟真人)*
                        
                        # *尝试定位物理搜索按钮，失败则回退至文本匹配*
                        search_btn = self.page.ele('css:button[data-e2e="searchbar-query-button"]')
                        if not search_btn:
                            search_btn = self.page.ele('text:搜索')
                        
                        if search_btn:
                            search_btn.click()
                            time.sleep(3)  # *等待 SPA 框架请求数据并渲染搜索结果瀑布流*
            except Exception:
                pass  # *确保逃逸逻辑发生网络或元素异常时，不会使主控循环崩溃*
            # *=== [Human Mimicry Escape Guard End] ===*
            
            # *=== [Reactive Tab Guard] SPA 路由对齐补丁 ===*
            try:
                # *定位“视频”导航标签页*
                video_tab = self.page.ele('text:视频', timeout=1)
                if video_tab:
                    # *检查标签是否处于激活状态 (依据 aria 属性或其 CSS 类名)*
                    is_aria_selected = video_tab.attr('aria-selected') == 'true'
                    is_class_active = 'active' in str(video_tab.attr('class')).lower()
                    is_parent_active = 'active' in str(video_tab.parent().attr('class')).lower()
                    
                    # *如果“视频”标签未被选中 (由批量并发导致的 SPA 路由漂移)*
                    if not (is_aria_selected or is_class_active or is_parent_active):
                        # *非侵入式修正：主动点击并切换到视频瀑布流*
                        video_tab.click(by_js=True)
                        time.sleep(1.5)  # *等待 SPA 框架完成 DOM 卸载与重绘*
                        
                        # *清理网络监听队列中可能存在的“综合搜索”干扰 API 包*
                        try:
                            self.page.listen.clear()
                        except:
                            pass
            except Exception:
                # *静默容错软降级：遇到任何解析错误直接放行，不影响源主线执行*
                pass
            # *=== [Reactive Tab Guard End] ===*

            # *=== [FIX START] Double-Lock Wait Mechanism ===*
            print("⏳ *[Sync] Waiting for search results to render...*")
            
            # *1. Hard Wait: Allow network request to initiate & old DOM to detach*
            time.sleep(3) 
            
            # *2. Smart Wait: Explicitly wait for ACTUAL VIDEO CONTENT (not skeleton placeholders)*
            # *Selector: 'ul li a[href*="/video/"]' ensures we detect REAL video links*
            try:
                if not self.page.ele('css:ul li a[href*="/video/"]', timeout=10):
                    print("⚠️ *[Warning] Video content did not appear within 10s (skeleton only?)!*")
                    # *3. Fallback: Screenshot for debugging*
                    self.page.get_screenshot(path='error_search_timeout.jpg', full_page=True)
                else:
                    # *4. Visual Confirmation Log: Show detected item count*
                    detected_items = self.page.eles('css:ul li a[href*="/video/"]')
                    print(f"✅ *[Sync] Visual check passed: Found {len(detected_items)} video items.*")
            except Exception as wait_err:
                print(f"⚠️ *[Warning] Smart Wait exception: {wait_err}*")
            # *=== [FIX END] ================================*

            self._reset_timeout_counter()
        except Exception as nav_err:
            print(f"⚠️ *[Fail-Fast] 搜索页导航超时: {nav_err}*")
            if not self._handle_network_timeout("初始搜索页导航"):
                print(f"❌ *[Fail-Fast] 无法恢复，终止视频采集*")
                return []
            # *恢复后重试导航*
            try:
                self.page.get(search_url)
            except Exception as retry_err:
                print(f"❌ *[Fail-Fast] 重试导航失败: {retry_err}，终止采集*")
                return []
        
        # *[Strategy A] (Removed original listen.start position)*

        pbar = tqdm(total=limit, desc="   📹 *视频流抓取中*", unit="条")
        collected_buffer = []

        # *[Fix 4] Bulletproof Exit Strategy: try/finally wrap*
        try:
            # *[CRITICAL FIX] 使用 session_collected 而非 existing_video_ids 长度*
            while session_collected < limit:
                # *================================================================*
                # *[Strategy B] "Minimum Yield" Completion Guard (The "Don't Quit" Rule)*
                # *================================================================*
                if empty_response_count >= NETWORK_CONFIG.get('max_empty_responses', 10):
                    # *计算本次会话实际采集的增量*
                    current_session_collected = len(self.existing_video_ids) - initial_collected_count
                
                    # *如果连续多次空响应，且本次尚未采集到任何数据*
                    if current_session_collected == 0:
                        zero_yield_retry_count += 1
                        if zero_yield_retry_count <= 3:
                            print(f"🔄 *[Strategy B] 零收益保护 (Attempt {zero_yield_retry_count}/3): 触发硬刷新 重置计数*")
                            try:
                                self.page.refresh()
                                time.sleep(5) # *给足时间重新加载*
                                empty_response_count = 0 # *重置空响应计数*
                                continue
                            except Exception as refresh_err:
                                print(f"⚠️ *[Strategy B] 刷新失败: {refresh_err}*")
                        else:
                            print("❌ *[Strategy B] 零收益保护耗尽，放弃该关键词*")
                            break
                    
                    # *Standard timeout handling if we have some data or retries exhausted*
                    print(f"⚠️ *[Fail-Fast] 连续 {NETWORK_CONFIG.get('max_empty_responses', 10)} 次空响应，触发恢复*")
                    if not self._handle_network_timeout("连续空响应"):
                        break
                    empty_response_count = 0
                    continue
                
            # *[BugFix] DrissionPage listener.wait() 存在内部 bug: 'fail' 变量未初始化*
                # *当超时发生时会抛出 UnboundLocalError，需要捕获处理*
                try:
                    res = self.page.listen.wait(timeout=NETWORK_CONFIG['listen_wait_timeout'])
                except UnboundLocalError:
                    # *DrissionPage 库内部 bug：超时时 'fail' 未定义*
                    res = None
                except Exception as listen_err:
                    # *[Fail-Fast] 任何监听异常都记录并继续*
                    print(f"⚠️ *[Fail-Fast] 监听器异常: {listen_err}, 继续尝试...*")
                    res = None
                
                # *[DEBUG] 诊断日志：网络响应状态*
                if res:
                    print(f"🔍 *[DEBUG] 收到网络响应: URL={res.url[:80] if res.url else 'None'}..., has_body={res.response.body is not None}*")
                else:
                    print(f"⚠️ *[DEBUG] 网络监听超时: 第 {empty_response_count+1}/{NETWORK_CONFIG.get('max_empty_responses', 10)} 次空响应*")
                
                if res and res.response.body:
                    try:
                        json_data = res.response.body
                        raw_list = json_data.get('data', []) or json_data.get('aweme_list', [])
                        
                        if raw_list:
                            self._reset_timeout_counter()  # *有数据，重置超时计数*
                            empty_response_count = 0       # *重置空响应计数*
                            zero_yield_retry_count = 0     # *[Strategy B] 成功采集，重置重试计数*
                        
                        pre_parse_count = session_collected  # *[Fix 1] 记录解析前的已采集数量*
                        
                        for item in raw_list:
                            data = self._parse_video_packet(item)
                            if data:
                                if len(data) != 16:
                                    continue
                                collected_buffer.append(data)
                                session_collected += 1  # *[CRITICAL FIX] 递增本次会话计数器*
                                pbar.update(1)
                                
                                # *[Status] 实时更新进度状态*
                                self.progress_stats['current'] = session_collected
                                self.progress_stats['total'] = limit
                                
                                # *[进度回调] 实时汇报视频采集进度到前端*
                                if progress_callback:
                                    progress_callback(
                                        session_collected, 
                                        limit, 
                                        start_time, 
                                        message=f"正在抓取视频: {data.get('视频描述', '')[:20]}..."
                                    )
                                
                                if session_collected >= limit: 
                                    break
                        
                        # *[Fix 1] Zero-Growth Sentinel Logic*
                        new_ids_in_batch = session_collected - pre_parse_count
                        if raw_list and new_ids_in_batch == 0:
                            consecutive_zero_growth_count += 1
                            print(f"⚠️ *[Zero-Growth Sentinel] 收到数据包但无新视频 (连续 {consecutive_zero_growth_count}/{NETWORK_CONFIG['max_zero_growth_cycles']} 次)*")
                            if consecutive_zero_growth_count >= NETWORK_CONFIG['max_zero_growth_cycles']:
                                print(f"🚨 *[Exhaustion_Alert] Stale content detected for theme: {self.current_theme}. Exiting...*")
                                break
                        else:
                            consecutive_zero_growth_count = 0
                        
                        # *增量存盘*
                        if collected_buffer:
                            self._save_to_csv(collected_buffer, self.video_save_path, collected_buffer[0].keys())
                
                            # *[NEW] 轻量级 Batch Commit (DB IOPS Optimization)*
                            if self.db_manager:
                                self.db_manager.save_video_batch(collected_buffer, self.current_theme)
                
                            collected_buffer = []
                
                        self.page.scroll.to_bottom()
                        time.sleep(get_gaussian_delay())
                        
                    except Exception as e:
                        print(f"⚠️ *包解析异常: {e}*")
                else:
                    # *[Fail-Fast] 空响应或超时*
                    empty_response_count += 1
                    
                    # *================================================================*
                    # *[Strategy C] "Eyes Open" DOM Fallback (The Safety Net)*
                    # *Scenario: Network missed but visual elements exist*
                    # *Action: Extract video_id directly from href attributes*
                    # *================================================================*
                    try:
                        # *多选择器优先级队列：尝试多种 CSS 选择器*
                        video_cards = None
                        fallback_selectors = [
                            'css:ul[data-e2e="scroll-list"] li a[href*="/video/"]',
                            'css:ul li a[href*="/video/"]',
                            'css:li.aweme-item a[href*="/video/"]',
                            'css:a[href*="/video/"]',
                        ]
                        
                        for selector in fallback_selectors:
                            try:
                                video_cards = self.page.eles(selector, timeout=0.5)
                                if video_cards and len(video_cards) > 0:
                                    break
                            except:
                                continue
                        
                        if video_cards and len(video_cards) > 0:
                            print(f"⚠️ *[Strategy C] Network MISSED, but {len(video_cards)} Visual Elements detected! Switching to DOM extraction*")
                            salvaged_count = 0
                            
                            for card in video_cards:
                                try:
                                    href = card.attr('href') or ''
                                    
                                    # *从 href 提取 video_id (格式: /video/7123456789)*
                                    import re
                                    match = re.search(r'/video/(\d+)', href)
                                    if not match:
                                        continue
                                        
                                    vid = match.group(1)
                                    
                                    # *内存去重*
                                    if vid in self.existing_video_ids:
                                        continue
                                    
                                    # *尝试提取描述文本 (从 alt 属性或内部文本)*
                                    desc = card.attr('title') or card.attr('alt') or card.text[:100] if hasattr(card, 'text') else ''
                                    desc = self._normalize_desc(desc) if desc else f'DOM_SALVAGED_{vid}'
                                    
                                    # *构建最小化 Fallback Packet (不含下载)*
                                    fallback_data = {
                                        '视频ID': vid,
                                        '用户名': 'DOM_FALLBACK',
                                        '粉丝数量': 0,
                                        '视频描述': desc,
                                        '发表时间': '未知时间',
                                        '视频时长': '00:00',
                                        '点赞数量': 0,
                                        '收藏数量': 0,
                                        '评论数量': 0,
                                        '分享数量': 0,
                                        '下载数量': 0,
                                        'visual_brightness': 0,
                                        'visual_saturation': 0,
                                        'cut_frequency': 0,
                                        'audio_bpm': 0,
                                        'local_temp_path': ''  # *DOM 模式不下载视频*
                                    }
                                    
                                    # *[Data Integrity] Ensure Exhaustion Sentinel is not captured*
                                    skip_flag = False
                                    for kw in END_OF_FEED_KEYWORDS:
                                        if kw in desc:
                                            skip_flag = True
                                            break
                                    if skip_flag or len(fallback_data) != 16:
                                        continue
                                    
                                    # *注册到去重集合*
                                    self.existing_video_ids.add(vid)
                                    collected_buffer.append(fallback_data)
                                    salvaged_count += 1
                                    session_collected += 1  # *[CRITICAL FIX] DOM Fallback 也递增会话计数器*
                                    pbar.update(1)
                                    
                                    # *[Status] 实时更新进度状态*
                                    self.progress_stats['current'] = session_collected
                                    self.progress_stats['total'] = limit
                                    
                                    # *达到上限时提前退出*
                                    if session_collected >= limit:
                                        break
                                        
                                except Exception as card_err:
                                    # *单卡片解析失败，继续下一个*
                                    continue
                            
                            if salvaged_count > 0:
                                print(f"✅ *[Strategy C] DOM Salvage Success: 抢救了 {salvaged_count} 条视频数据*")
                                # *重置计数器，因为实际有数据*
                                empty_response_count = 0
                                zero_yield_retry_count = 0
                                
                                # *增量存盘*
                                if collected_buffer:
                                    self._save_to_csv(collected_buffer, self.video_save_path, collected_buffer[0].keys())
                                    collected_buffer = []
                    except Exception as dom_err:
                        print(f"⚠️ *[Strategy C] DOM Fallback 异常: {dom_err}*")
                
                    try:
                        # *[Fix 2] Multilingual DOM Text Sentinel*
                        feed_exhausted = False
                        for kw in END_OF_FEED_KEYWORDS:
                            try:
                                if self.page.ele(f'text:{kw}', timeout=0.1):
                                    print(f"🚨 *[Exhaustion_Alert] DOM end-of-feed detected: '{kw}' for theme: {self.current_theme}. Exiting...*")
                                    feed_exhausted = True
                                    break
                            except:
                                continue
                        
                        if feed_exhausted:
                            break
                            
                        current_height = self.page.run_js('return document.documentElement.scrollHeight;')
                        current_card_count = len(video_cards) if video_cards else 0
                        
                        if current_height == last_page_height and current_card_count <= last_dom_card_count:
                            stagnant_bottom_count += 1
                        else:
                            stagnant_bottom_count = 0
                            last_page_height = current_height
                            last_dom_card_count = current_card_count
                            
                        if stagnant_bottom_count >= 3:
                            print(f"🎉 *[End of Data] 页面数据已全部见底 (监测到停滞)...*")
                            break
                    except Exception as detect_err:
                        print(f"⚠️ *[Graceful Exit] 停滞检测异常: {detect_err}*")
                
                    self.page.scroll.to_bottom()
        finally:
            pbar.close()
            self.page.listen.stop()
        
        # *返回本次采集到的所有视频 ID，用于下一阶段*

    # *================================================================*
    # *4. 模块 B: 评论挖掘引擎 (Comment Miner)*
    # *================================================================*

    def _parse_comment(self, vid, c_data):
        # *解析单条评论结构*
        cid = str(c_data.get('cid'))
        if not cid or cid in self.existing_comment_ids:
            return None
            
        self.existing_comment_ids.add(cid)
        create_time = c_data.get('create_time', 0)
        
        return {
            '视频ID': vid,
            '评论ID': cid,
            '评论时间': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(create_time)),
            '用户名': c_data.get('user', {}).get('nickname', '未知'),
            '评论内容': c_data.get('text', ''),
            '点赞数': c_data.get('digg_count', 0),
            'IP属地': c_data.get('ip_label', '未知')
        }

    def _activate_comment_box(self):
        # *多向量激活策略：优先级选择器队列*
        is_open = False
        target = None
        
        try:
            # *Step 1: 优先级选择器队列 (Priority Selector Queue)*
            # *策略1: 精确属性匹配*
            try:
                target = self.page.ele('xpath://div[@data-e2e="comment-icon"]', timeout=2)
            except:
                pass
            
            if not target:
                try:
                    target = self.page.ele('xpath://span[contains(@class,"comment-icon")]', timeout=2)
                except:
                    pass
            
            # *策略2: 模糊文本匹配*
            if not target:
                try:
                    target = self.page.ele('xpath://span[contains(text(), "评论")]', timeout=2)
                except:
                    pass
            
            if not target:
                try:
                    target = self.page.ele('text:全部评论', timeout=2)
                except:
                    pass
            
            # *执行点击*
            if target:
                try:
                    self.page.run_js('arguments[0].scrollIntoView({behavior: "auto", block: "center", inline: "center"});', target)
                except:
                    pass
                time.sleep(0.3)
                try:
                    target.click(by_js=True)
                except:
                    target.click()
            else:
                # *兜底方案：坐标点击*
                rect = self.page.rect
                x = int(rect.size[0] * 0.85)
                y = int(rect.size[1] * 0.6)
                self.page.actions.move(x, y).click()
            
            # *Step 2: 视觉确认循环 (Visual Confirmation Loop)*
            timeout_at = time.time() + 3.0  # *3秒超时*
            container = None
            
            while time.time() < timeout_at:
                try:
                    # *优先使用 data-e2e 属性选择器（更稳定）*
                    container = self.page.ele('css:div[data-e2e="comment-list"]', timeout=0.5)
                    if not container:
                        # *备用：class 选择器*
                        container = self.page.ele('.comment-mainContent', timeout=0.5)
                    if container:
                        break
                except:
                    pass
                time.sleep(0.5)
            
            # *Step 3: 视口劫持 (Viewport Hijacking)*
            if container:
                # *强制焦点：将容器居中到视口*
                try:
                    self.page.run_js('arguments[0].scrollIntoView({block: "center", behavior: "smooth"});', container)
                except:
                    pass
                
                # *DrissionPage 锚定*
                try:
                    container.scroll.to_see()
                except:
                    pass
                
                is_open = True
            else:
                # *失败处理：容器未出现*
                print("⚠️ *[Focus Lost] 评论容器未出现，跳过该视频*")
                return False
            
        except Exception as e:
            print(f"⚠️ *[调试] 唤醒评论区操作异常: {e}*")
            return False
        
        return is_open

    def run_comment_crawler(self, task_config, video_data_list, progress_callback=None):
        # *绑定计数器变量，避免 UnboundLocalError*
        empty_response_count = 0
        
        # *评论采集主逻辑：接收视频元数据列表，支持零评过滤*
        limit_per_vid = task_config['max_comments']
        total_vids = len(video_data_list)
        print(f"\n💬 *[阶段二] 启动评论挖掘 | 目标视频数: {total_vids} | 单视频上限: {limit_per_vid}*")
        
        # *[Status] 更新阶段状态*
        self.progress_stats = {'current': 0, 'total': total_vids, 'stage': 'comment_crawling'}

        headers = ['视频ID', '评论ID', '评论时间', '用户名', '评论内容', '点赞数', 'IP属地']
        
        # *读取已有评论 ID 防止跨任务重复*
        if os.path.exists(self.comment_save_path):
            try:
                df = pd.read_csv(self.comment_save_path, usecols=['评论ID'])
                self.existing_comment_ids.update(df['评论ID'].astype(str).tolist())
            except: pass

        self.page.listen.start('comment/list')

        # *性能统计变量*
        start_time = time.time()
        processed_count = 0

        # *[Fix 4] Bulletproof Exit Strategy for Comments*
        try:
            with tqdm(total=total_vids, desc="   📂 *视频遍历中*") as main_bar:
                for idx, v_item in enumerate(video_data_list):
                    vid = str(v_item.get('视频ID'))
                    
                    # *Zero-Comment Skip: 检查元数据中的评论数*
                    try:
                        meta_comment_count = int(v_item.get('评论数量', 0))
                        if meta_comment_count == 0:
                            # *直接跳过，节省时间*
                            processed_count += 1
                            main_bar.update(1)
                            if progress_callback:
                                # *即便跳过也汇报进度*
                                progress_callback(processed_count, total_vids, start_time, message="跳过无评论视频...")
                            
                            # *[Status] 更新进度*
                            self.progress_stats['current'] = processed_count
                            continue
                    except:
                        pass

                    # *汇报进度 (每处理一个视频前)*
                    if progress_callback:
                        progress_callback(processed_count + 1, total_vids, start_time, message=f"正在挖掘视频评论: {vid}")

                    # *[Fail-Fast] 评论页导航包裹在 try-except 中*
                    url = f"https://www.douyin.com/video/{vid}"
                    try:
                        self.page.get(url)
                        self._reset_timeout_counter()  # *导航成功，重置计数*
                    except Exception as nav_err:
                        print(f"⚠️ *[Fail-Fast] 视频页导航超时 ({vid}): {nav_err}, 跳过该视频*")
                        self._handle_network_timeout(f"视频页导航 {vid}")
                        processed_count += 1
                        main_bar.update(1)
                        continue
                    
                    time.sleep(1.5) # *稍微减少等待，加速流程*
                
                    # *激活评论区*
                    # *若激活失败 (如网络原因或视频被删)，则跳过*
                    if not self._activate_comment_box():
                        processed_count += 1
                        main_bar.update(1)
                        continue
                
                    collected_count = 0
                    retry = 0
                
                    # *单视频内的滚动循环*
                    # *Rhythmic Container Scrolling: 节奏化容器滚动*
                    container_ele = None
                    try:
                        # *优先使用 data-e2e 属性选择器（更稳定）*
                        container_ele = self.page.ele('css:div[data-e2e="comment-list"]')
                        if not container_ele:
                            # *备用：class 选择器*
                            container_ele = self.page.ele('.comment-mainContent')
                    except: pass
                
                    # *================================================================*
                    # *Bio-Mimetic Scrolling Loop (高性能评论滚动引擎)*
                    # *策略: Scroll → Instant Listen → Jitter on Data*
                    # *================================================================*

                    # *最多尝试 5 次加载/API 等待*
                    # *[Resilience] 用于 Partial Save 的本地缓冲区 (跨循环持久化)*
                    comment_buffer = []
                
                    while collected_count < limit_per_vid and retry < 5:
                        # *Step 1: 目标滚动 (Targeted Scroll)*
                        # *优先使用容器级滚动，而非页面滚动*
                        if container_ele:
                            try:
                                # *随机步长 (Randomized Step): 模拟人类阅读行为 300-600px 微滚*
                                scroll_pixels = random.randint(300, 600)
                                container_ele.scroll.down(scroll_pixels)
                            except:
                                # *元素若失效则尝试重新获取*
                                try:
                                    # *重新获取容器（优先 data-e2e）*
                                    container_ele = self.page.ele('css:div[data-e2e="comment-list"]')
                                    if not container_ele:
                                        container_ele = self.page.ele('.comment-mainContent')
                                    scroll_pixels = random.randint(300, 600)
                                    container_ele.scroll.down(scroll_pixels)
                                except:
                                    self.page.scroll.down(random.randint(600, 1000))
                        else:
                            self.page.scroll.down(random.randint(600, 1000))
                    
                        # *Step 2: Instant Listen (即时监听)*
                        # *滚动后立即监听网络请求，2秒超时*
                        # *策略: 若有数据立即捕获，无需等待固定时间*
                        try:
                            # *[Fix 3] Config-Driven Consistency*
                            res = self.page.listen.wait(timeout=NETWORK_CONFIG['listen_wait_timeout'])
                        except UnboundLocalError:
                            # *DrissionPage 库内部 bug：超时时 'fail' 未定义*
                            res = None
                        except Exception as listen_err:
                            print(f"⚠️ *[Fail-Fast] 评论监听异常: {listen_err}, 继续尝试...*")
                            res = None

                        # *Step 3: Loop Termination (底部检测)*
                        # *若 res 为空 (2秒内无新数据)，表示已到评论底部*
                        if not res or not res.response.body:
                            retry += 1
                            continue

                        # *Step 4: Bio-Mimetic Jitter (仿生延迟)*
                        # *仅当收到数据时才添加 200-500ms 人类反应时间*
                        # *模拟人类看到新内容后的自然停顿*
                        time.sleep(random.uniform(0.2, 0.5))
                    
                        # *Step 5: Data Processing (数据处理)*
                        # *到达此处时 res 必定有效 (已通过 Step 3 过滤)*
                        try:
                            comments = res.response.body.get('comments', [])
                            has_more = res.response.body.get('has_more', 0)
                        
                            buffer = []
                            for c in comments:
                                item = self._parse_comment(vid, c)
                                if item: buffer.append(item)
                            
                            if buffer:
                                self._save_to_csv(buffer, self.comment_save_path, headers)
                            
                                # *[Resilience] 累积到跨循环缓冲区 (用于 Partial Save)*
                                comment_buffer.extend(buffer)
                            
                                # *[Async Persistence] 将数据推入异步队列，由 Worker 线程处理*
                                # *非阻塞提交，解耦爬取速度与数据库写入速度*
                                if self.db_manager:
                                    try:
                                        self._db_queue.put((list(buffer), self.current_theme), block=True, timeout=5)
                                    except queue.Full:
                                        # *队列已满，直接同步写入 (降级策略)*
                                        logger.warning("*[Async DB] 队列已满，降级为同步写入*")
                                        self.db_manager.save_comment_batch(buffer, self.current_theme)
                                
                                collected_count += len(buffer)
                                retry = 0 # *成功获取数据，重置重试计数*
                            else:
                                retry += 1
                            
                            if has_more == 0: break
                            if collected_count >= limit_per_vid: break # *双重检查*
                        
                        except Exception as parse_err:
                            print(f"⚠️ *[评论解析] 异常: {parse_err}*")
                            retry += 1
                
                        # *================================================================*
                        # *[Resilience] Partial Save 机制 (网络超时/评论底部容错)*
                        # *当 retry >= 5 时，已采集的数据不应丢失*
                        # *================================================================*
                    if retry >= 5 and comment_buffer:
                        print(f"[Resilience] ⚠️ Network timeout/End reached. Attempting partial save of {len(comment_buffer)} comments.")
                        # *最终计数更新 (确保准确反映实际采集量)*
                        collected_count = len(comment_buffer)
                        # *显式推送剩余缓冲区到异步队列 (若尚未推送)*
                        # *注意: 由于循环内已逐批推送，此处仅作为安全网*
                        # *若需要保证完整性，可在此添加额外逻辑*
                        print(f"[Resilience] ✅ Partial save completed. Moving to next video.")
                    
                    processed_count += 1
                    self.progress_stats['current'] = processed_count
                    main_bar.update(1)
        finally:
            self.page.listen.stop()

    def execute_task(self, keyword, max_videos, max_comments, theme_name, progress_callback=None):
        # *执行单次任务的逻辑封装*
        self.current_theme = theme_name
        print(f"\n" + "="*60)
        print(f"📦 *执行任务: {self.current_theme}*")
        print(f"   *关键词: {keyword}*")
        print("="*60)
        
        # *动态路径初始化*
        self.media_dir = self._ensure_directories(self.current_theme)
        self.video_save_path = os.path.join(PROJECT_ROOT, "data", f"douyin_video_{self.current_theme}.csv")
        self.comment_save_path = os.path.join(PROJECT_ROOT, "data", f"douyin_comment_{self.current_theme}.csv")
        
        # *清空单次任务的去重缓存*
        self.existing_video_ids.clear()
        self.existing_comment_ids.clear()
        
        # *加载已完成的视频 (断点续传)*
        if os.path.exists(self.video_save_path):
            try:
                df = pd.read_csv(self.video_save_path, usecols=['视频ID'], dtype=str, on_bad_lines='skip', engine='python', encoding='utf-8-sig')
                self.existing_video_ids.update(df['视频ID'].astype(str).tolist())
            except: pass

        task_config = {
            "keyword": keyword,
            "max_videos": max_videos,
            "max_comments": max_comments
        }
        
        # *执行视频采集 (阶段一通常较快，暂不加细粒度进度)*
        if progress_callback:
            # *通知：开始视频采集*
            progress_callback(0, max_videos, time.time(), message="正在采集视频流...")
            
        # *[关键修复] 传递 progress_callback 到视频采集函数*
        self.run_video_crawler(task_config, progress_callback=progress_callback)
        
        # *准备评论采集元数据*
        target_video_data = []
        if os.path.exists(self.video_save_path):
             try:
                 # *读取完整 CSV 以获取 '评论数量' 元数据*
                 df = pd.read_csv(self.video_save_path, dtype=str, on_bad_lines='skip', engine='python', encoding='utf-8-sig')
                 df = df[df['视频ID'].notna()]
                 df['视频ID'] = df['视频ID'].astype(str).str.strip()
                 if '视频ID' in df.columns:
                     # *转换为字典列表 [{'视频ID': '...', '评论数量': 100}, ...]*
                     target_video_data = df.to_dict('records')
                     print(f"✅ *[评论准备] 成功加载 {len(target_video_data)} 条视频元数据*")
                 else:
                     print(f"⚠️ *[评论准备] CSV 缺少 '视频ID' 列，跳过评论采集*")
             except Exception as csv_err:
                 # *[修复] 不再静默吞掉异常，打印错误信息*
                 print(f"❌ *[评论准备] 读取视频 CSV 失败: {csv_err}*")
        else:
            print(f"⚠️ *[评论准备] 视频 CSV 不存在: {self.video_save_path}*")

        # *执行评论采集*
        if target_video_data:
            if progress_callback:
                progress_callback(0, len(target_video_data), time.time(), message="启动评论挖掘...")
            self.run_comment_crawler(task_config, target_video_data, progress_callback)
        else:
            print("⚠️ *[警告] 未获取到有效视频数据，跳过评论采集阶段*")

        return self.video_save_path, self.comment_save_path

    def close(self):
        # *================================================================*
        
        # *[Global Access] 注销实例*
        DouyinUnifiedPipeline.instance = None
        # *[Safe Shutdown] 优雅关闭 (Graceful Shutdown)*
        # *================================================================*
        
        # *Step 1: 等待异步队列清空 (确保数据完整性)*
        print("⏳ *[Shutdown] Waiting for DB queue to flush...*")
        try:
            self._db_queue.join()  # *阻塞直到所有队列任务完成*
            print("✅ *[Shutdown] DB queue flushed successfully.*")
        except Exception as flush_err:
            print(f"⚠️ *[Shutdown] Queue flush warning: {flush_err}*")
        
        # *Step 2: 关闭浏览器资源*
        # *[Fix 4] 增加 try/except 保护，防止因浏览器进程提前死亡导致异常*
        try:
            self.page.quit()
            print("🏁 *[Shutdown] Browser closed. All resources released.*")
        except Exception as e:
            print(f"⚠️ *[Shutdown] Browser was already closed or crashed: {e}*")

# *================================================================*
# *5. 对外服务接口 (Service Interface)*
# *================================================================*

def run_spider_service(keyword, max_videos, max_comments, theme_name, progress_callback=None):
    """
    *启动爬虫服务的封装函数*
    :param keyword: *搜索关键词*
    :param max_videos: *最大视频抓取数量*
    :param max_comments: *每视频最大评论抓取数量*
    :param theme_name: *任务主题名称 (用于生成文件名)*
    :param progress_callback: *回调函数 (current, total, start_time, message) -> None*
    :return: *返回 (视频CSV路径, 评论CSV路径) 的元组*
    """
    print(f"🚀 *[系统启动] 抖音采集服务 (Service Mode)*")
    
    spider = None
    try:
        spider = DouyinUnifiedPipeline()
        video_path, comment_path = spider.execute_task(keyword, max_videos, max_comments, theme_name, progress_callback)
        print("\n🏁 *[服务结束] 任务执行完毕*")
        return video_path, comment_path
    except Exception as e:
        print(f"❌ *服务执行异常: {e}*")
        raise e
    finally:
        if spider:
            spider.close()

def run_comment_only_service(video_csv_path, theme_name, max_comments, progress_callback=None):
    """
    *仅评论采集服务函数*
    *从已有视频CSV读取视频ID,跳过视频采集阶段,直接进入评论采集*
    :param video_csv_path: *视频CSV文件的绝对路径*
    :param theme_name: *任务主题名称 (用于生成评论文件名)*
    :param max_comments: *每视频最大评论抓取数量*
    :param progress_callback: *回调函数 (current, total, start_time, message) -> None*
    :return: *返回评论CSV路径*
    """
    print(f"🚀 *[系统启动] 仅评论采集服务 (Comment-Only Mode)*")
    
    # *读取视频ID列表*
    if not os.path.exists(video_csv_path):
        print(f"❌ *视频CSV不存在: {video_csv_path}*")
        return None
    
    try:
        # *[The Shield] 救援模式读取CSV，跳过坏行并保持String类型防精度截取*
        df = pd.read_csv(
            video_csv_path, 
            dtype=str, 
            on_bad_lines='skip', 
            engine='python', 
            encoding='utf-8-sig'
        )
        
        # *智能匹配列名 (兼容 视频ID / video_id)*
        id_col = next((c for c in df.columns if c in ['视频ID', 'video_id', 'id']), None)
        cnt_col = next((c for c in df.columns if c in ['评论数量', 'comment_count', 'count']), None)
        
        if not id_col:
            print("❌ *CSV中未找到 ID 列*")
            return None
        
        # *转换为字典列表格式，剔除空 ID*
        video_data_list = []
        for _, row in df.iterrows():
            vid = str(row.get(id_col, '')).strip()
            if vid and vid != 'nan':
                cnt_str = str(row.get(cnt_col, '0')) if cnt_col else '0'
                video_data_list.append({
                    '视频ID': vid,
                    '评论数量': int(float(cnt_str)) if cnt_str.replace('.', '', 1).isdigit() else 0
                })
        
        print(f"📂 *成功抢救出 {len(video_data_list)} 个有效视频 ID（已自动剔除逗号炸弹等坏行）*")
        
        # *[Real-Time Telemetry] 实时更新进度，通知前端数据清洗完成*
        if progress_callback:
            progress_callback(0, len(video_data_list), time.time(), message=f"已抢救 {len(video_data_list)} 个有效记录，分配爬虫资源...")
            
    except Exception as e:
        print(f"❌ *救援读取CSV失败: {e}*")
        return None
    
    spider = None
    try:
        spider = DouyinUnifiedPipeline()
        
        # *设置输出路径*
        spider.current_theme = theme_name
        spider.comment_save_path = os.path.join(PROJECT_ROOT, "data", f"douyin_comment_{theme_name}.csv")
        
        # *任务配置*
        task_config = {
            "keyword": "",  # *评论模式不需要关键词*
            "max_videos": 0,
            "max_comments": max_comments
        }
        
        print(f"\n💬 *启动评论采集 | 主题: {theme_name} | 视频数: {len(video_data_list)}*")
        
        # *直接调用评论采集（传递字典列表）*
        spider.run_comment_crawler(task_config, video_data_list, progress_callback)
        
        print(f"\n✅ *评论已保存至: {spider.comment_save_path}*")
        print("🏁 *[服务结束] 仅评论采集任务执行完毕*")
        
        return spider.comment_save_path
        
    except Exception as e:
        print(f"❌ *评论采集异常: {e}*")
        raise e
    finally:
        if spider:
            spider.close()

if __name__ == "__main__":
    print("="*50)
    print("🕷️ *抖音数据采集引擎 (Interactive Mode)*")
    print("="*50)
    
    try:
        # 1. 关键词
        keyword = input("\n👉 *请输入搜索关键词 (例如 '猫咪'):* ").strip()
        if not keyword:
            print("❌ *关键词不能为空*")
            sys.exit(1)
            
        # 2. 主题名称
        default_theme = keyword
        theme_name = input(f"👉 *请输入任务主题名 (默认 '{default_theme}'):* ").strip()
        if not theme_name:
            theme_name = default_theme
            
        # 3. 视频抓取数
        default_v_max = 10
        v_inp = input(f"👉 *请输入抓取视频数量 (默认 {default_v_max}):* ").strip()
        max_videos = int(v_inp) if v_inp.isdigit() and int(v_inp) > 0 else default_v_max

        # 4. 评论抓取数
        default_c_max = 50
        c_inp = input(f"👉 *请输入每视频评论数量 (默认 {default_c_max}):* ").strip()
        max_comments = int(c_inp) if c_inp.isdigit() and int(c_inp) > 0 else default_c_max

        print(f"\n🚀 *任务已确认*")
        print(f"   *关键词:* {keyword}")
        print(f"   *主题:* {theme_name}")
        print(f"   *目标:* {max_videos} 视频, {max_comments} 评论/视频")
        input("   (按 Enter 开始执行, Ctrl+C 取消...)")
        
        v, c = run_spider_service(keyword, max_videos, max_comments, theme_name)
        print(f"\n✅ *数据已保存:*")
        print(f"   *视频:* {v}")
        print(f"   *评论:* {c}")
        
    except KeyboardInterrupt:
        print("\n❌ *用户取消*")
    except Exception as e:
        print(f"\n❌ *执行异常: {e}*")
    finally:
        input("\nPRESS ENTER TO EXIT...")
