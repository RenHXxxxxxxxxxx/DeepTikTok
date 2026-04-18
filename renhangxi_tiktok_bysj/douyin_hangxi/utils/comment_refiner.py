import pandas as pd
import re
import os
import jieba
from tqdm import tqdm
from snownlp import SnowNLP  # 必须引入情感分析库 (pip install snownlp)


# ======================================================
# 类定义：通用评论洗练引擎 (CommentRefiner V3.0)
# 功能：正则保留关键术语 -> 领域分词 -> 情感计算 -> 停用词过滤 -> 极致去重
# ======================================================

class CommentRefiner:
    def __init__(self, input_path, output_path, stopwords_path):
        """
        初始化清洗器
        :param input_path: 原始评论 CSV 路径
        :param output_path: 洗练后 CSV 路径
        :param stopwords_path: 停用词 TXT 路径
        """
        self.input_path = input_path
        self.output_path = output_path
        self.stopwords_path = stopwords_path

        # 1. 加载停用词
        self.stopwords = self._load_stopwords()

        # 2. 【核心优化】挂载领域专业词典，防止分词碎片化
        self._init_user_dict()

    def _init_user_dict(self):
        """
        初始化领域专用词典
        强制 Jieba 识别这些词为一个整体，不进行拆分
        """
        keywords = [
            # 品牌与产品
            "DeepSeek", "Skechers", "斯凯奇", "D'Lites", "熊猫鞋",
            "GoWalk", "ArchFit", "闪穿", "一脚蹬",
            # 技术术语
            "AI", "GPU", "RTX", "3060", "CUDA", "Python",
            # 电影与热点
            "哪吒", "Nezha", "春节档", "封神"
        ]
        for kw in keywords:
            jieba.add_word(kw)
        print(f" [领域词典] 已挂载 {len(keywords)} 个专业术语 (DeepSeek, Skechers, 3060...)")

    def _load_stopwords(self):
        """加载外部停用词表"""
        stopwords = set()
        if os.path.exists(self.stopwords_path):
            try:
                # 兼容 utf-8 格式加载停用词
                with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            stopwords.add(word)
                print(f" [资源加载] 成功加载 {len(stopwords)} 个停用词。")
            except Exception as e:
                print(f" 停用词加载异常: {e}")
        else:
            print(f" 警告：未找到停用词文件 {self.stopwords_path}，将跳过停用词过滤。")
        return stopwords

    def sanitize_pattern(self, text):
        """
        第一阶段：结构化噪声清理
        【核心修正】：放行英文、数字和关键符号，不再暴力删除
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. 剔除表情 [xxx] (非贪婪匹配)
        text = re.sub(r'\[.*?\]', '', text)

        # 2. 剔除 @用户名 及其后的内容
        text = re.sub(r'@\S+', '', text)

        # 3. 剔除 URL 网页链接
        text = re.sub(r'http\S+', '', text)

        # 4. 【关键修改】保留中文 + 英文 + 数字 + 常用标点
        # 原逻辑 r'[^\u4e00-\u9fa5...]' 会把 DeepSeek 和 3060 删掉
        # 现增加 a-zA-Z0-9
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、“”（）]', ' ', text)

        # 5. 清理多余空格
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def calculate_sentiment(self, text):
        """
        【新增功能】情感倾向计算
        返回: 0.0 (极负面) ~ 1.0 (极正面)
        """
        if not text or len(text.strip()) == 0:
            return 0.5  # 空文本视为中立
        try:
            # SnowNLP 内部会自动处理分词
            s = SnowNLP(text)
            return s.sentiments
        except:
            return 0.5

    def refine_text(self, text):
        """
        第二阶段：分词与语义精炼
        """
        # 获取第一阶段清洗后的干净文本
        clean_text = self.sanitize_pattern(text)
        if not clean_text:
            return ""

        # 使用 jieba 精确分词 (此时已应用 _init_user_dict)
        words = jieba.lcut(clean_text)

        # 【逻辑优化】放宽过滤条件
        refined_words = []
        for w in words:
            # 1. 长度大于1
            # 2. 不在停用词表 (转小写比较，忽略大小写差异)
            if len(w) > 1 and w.lower() not in self.stopwords:
                # 3. 【修改】只要包含有效字符(中文/英文/数字)就保留
                # 这样 "DeepSeek" (纯英) 和 "3060" (纯数) 都能留下来
                if re.search(r'[\u4e00-\u9fa5a-zA-Z0-9]', w):
                    refined_words.append(w)

        return " ".join(refined_words)

    def run(self, min_words=1):
        """
        执行批处理洗练任务
        """
        print(f"\n [引擎启动] 正在执行：正则去噪 -> 领域分词 -> 情感计算 -> 唯一性校验...")

        if not os.path.exists(self.input_path):
            print(f" 运行中止：找不到输入文件 {self.input_path}")
            return

        # 读取原始数据
        # dtype=str 防止纯数字 ID 丢失精度，fillna 防止空值报错
        df = pd.read_csv(self.input_path, dtype={'视频ID': str, '评论ID': str, '评论内容': str})
        df['评论内容'] = df['评论内容'].fillna('')
        initial_count = len(df)

        # 注册进度条
        tqdm.pandas(desc=" 全流程处理中")

        # 1. 清洗 (Sanitize) - 用于肉眼阅读和情感分析
        df['清洗文本'] = df['评论内容'].progress_apply(self.sanitize_pattern)

        # 2. 【新增】情感计算 - 这一步对后续预测热度至关重要
        # 使用 '清洗文本' 计算，保留原句通顺度
        df['sentiment_score'] = df['清洗文本'].progress_apply(self.calculate_sentiment)

        # 3. 分词 (Tokenize) - 用于生成词云和提取关键词特征
        df['refined_content'] = df['评论内容'].progress_apply(self.refine_text)

        # 第三阶段：质量护卫过滤
        # 剔除经过洗练后内容完全消失的记录
        df = df[df['refined_content'].str.strip().astype(bool)]

        # 剔除有效关键词数不足的内容
        df = df[df['refined_content'].apply(lambda x: len(x.split())) >= min_words]

        # 第四阶段：极致物理去重
        # 在同一个视频 ID 下，如果清洗后的关键词特征完全一致，视为重复搬运
        df.drop_duplicates(subset=['视频ID', 'refined_content'], keep='first', inplace=True)

        # 结果导出 (utf-8-sig 确保 Excel 完美打开)
        df.to_csv(self.output_path, index=False, encoding='utf-8-sig')

        final_count = len(df)
        print(f"\n 洗练任务总结：")
        print(f"   - 原始记录：{initial_count} 条")
        print(f"   - 剩余精华：{final_count} 条")
        print(f"   - 剔除杂质：{initial_count - final_count} 条")
        print(f"   -  核心增强：已生成 'sentiment_score' (情感分) 列")
        print(f"   -  核心增强：已保留 'DeepSeek/3060' 等专业术语")
        print(f" 处理完成！精炼后的 CSV 已生成：{self.output_path}\n")


# ======================================================
# 路径自适应运行入口
# 适配嵌套项目结构 D:\renhangxi_tiktok_bysj\renhangxi_tiktok_bysj\...
# ======================================================

if __name__ == "__main__":
    # 1. 定位当前脚本路径
    script_path = os.path.abspath(__file__)

    # 2. 向上跳 4 级获取总根目录 (D:\renhangxi_tiktok_bysj)
    # 你的目录结构较深，这里保持原逻辑
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path))))

    print(f" 目录探测：检测到项目总根目录为 -> {PROJECT_ROOT}")

    # 3. 构造路径
    # [注意]：如果你现在跑的是 'Sikachi' 主题，请手动修改下面的文件名！
    # 比如改为: "douyin_comment_sikachi.csv"
    THEME_NAME = "sikachi"  # <--- 在这里修改主题名即可

    INPUT_FILE = os.path.join(PROJECT_ROOT, "data", f"douyin_comment_{THEME_NAME}.csv")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", f"douyin_comment_refined_{THEME_NAME}.csv")
    STOPWORDS_FILE = os.path.join(PROJECT_ROOT, "data", "hit_stopwords.txt")

    # 4. 实例化并运行
    refiner = CommentRefiner(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        stopwords_path=STOPWORDS_FILE
    )

    # 运行清洗，要求每条评论至少包含 1 个有效关键词
    refiner.run(min_words=1)
