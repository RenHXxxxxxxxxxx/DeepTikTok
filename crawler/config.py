# config.py
class Config:
    # 数据库基础配置
    DB_HOST = "localhost"
    DB_USER = "root"
    DB_PASSWORD = "123456"
    DB_NAME = "renhangxi_tiktok_bysj"
    DB_PORT = 3306

    # 核心改动：项目标识符
    # 当你换了一个主题（比如换成了 哪吒），只需要把这里改为 'nezha'
    # 所有的分析表名就会自动变为 analysis_nezha_...
    PROJECT_TAG = "disney_crazy_animals_city"