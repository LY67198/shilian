import os
from datetime import datetime
from app.core.config import settings
import logging
import logging.handlers
from logging.config import dictConfig
import sys

# 日志目录
script_path = os.path.abspath(__file__)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(script_path), "../.."))

LOG_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d')}.log")
SQLALCHEMY_LOG_FILE = os.path.join(LOG_DIR, f"sqlalchemy_{datetime.now().strftime('%Y%m%d')}.log")
MASTER_PROCESS_FILE = os.path.join(LOG_DIR, "master_process.lock")

# 根据环境设置日志级别
ENV = settings.ENV.lower()
LOG_LEVELS = {
    "development": "DEBUG",
    "testing": "INFO",
    "production": "WARNING"
}
LOG_LEVEL = LOG_LEVELS.get(ENV, "INFO")
SQLALCHEMY_LEVEL = "INFO" if ENV == "production" else "DEBUG"

# 判断是否为主进程
def is_master_process():
    pid = os.getpid()
    worker_id = os.environ.get("UVICORN_WORKER_ID", os.environ.get("UVICORN_PROCESS_ID", None))
    if (worker_id is not None and worker_id == "0") or worker_id is None:
        try:
            if not os.path.exists(MASTER_PROCESS_FILE):
                with open(MASTER_PROCESS_FILE, "w") as f:
                    f.write(str(pid))
                return True
            else:
                with open(MASTER_PROCESS_FILE, "r") as f:
                    master_pid = f.read().strip()
                try:
                    os.kill(int(master_pid), 0)
                    return pid == int(master_pid)
                except OSError:
                    with open(MASTER_PROCESS_FILE, "w") as f:
                        f.write(str(pid))
                    return True
        except Exception:
            return False
    return False

# 创建一个多进程安全的文件处理器
class SafeTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """多进程安全的日志文件处理器"""
    def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, delay=False, utc=False, atTime=None):
        super().__init__(filename, when, interval, backupCount, encoding, delay, utc, atTime)
        self.delay = True
        self.mode = 'a'

    def _open(self):
        return open(self.baseFilename, self.mode, encoding=self.encoding)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console_formatter": {
            "format": "%(asctime)s [%(levelname)s] [PID:%(process)d] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "file_formatter": {
            "format": "%(asctime)s [%(levelname)s] [PID:%(process)d] %(name)s [%(pathname)s:%(lineno)d]: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "console_formatter",
        },
        "file": {
            "class": "app.core.log_config.SafeTimedRotatingFileHandler",
            "level": LOG_LEVEL,
            "formatter": "file_formatter",
            "filename": LOG_FILE,
            "when": "midnight",
            "interval": 1,
            "backupCount": 7,
            "encoding": "utf-8",
        },
        "sqlalchemy_file": {
            "class": "app.core.log_config.SafeTimedRotatingFileHandler",
            "level": SQLALCHEMY_LEVEL,
            "formatter": "file_formatter",
            "filename": SQLALCHEMY_LOG_FILE,
            "when": "midnight",
            "interval": 1,
            "backupCount": 7,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": True,
        },
        "pdfminer": {
            "handlers": ["file"],
            "level": "WARNING",
            "propagate": False,
        },
        "python_multipart": {
            "handlers": ["file"],
            "level": "WARNING",
            "propagate": False,
        },
        "sqlalchemy.engine": {
            "handlers": ["console", "sqlalchemy_file"],
            "level": SQLALCHEMY_LEVEL,
            "propagate": False,
        },
        "uvicorn.access": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
        "uvicorn": {
            "handlers": ["console", "file"],
            "level": LOG_LEVEL,
            "propagate": False,
        },
    },
}

def setup_logging():
    """应用日志配置"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    dictConfig(LOGGING_CONFIG)
    if is_master_process():
        logging.getLogger("app.core.log_config").info("日志系统已初始化（主进程）")


def shutdown_logging():
    """关闭日志处理器"""
    if is_master_process():
        logging.getLogger("app.core.log_config").info("正在关闭日志系统（主进程）")
        try:
            if os.path.exists(MASTER_PROCESS_FILE):
                os.remove(MASTER_PROCESS_FILE)
        except Exception:
            pass
    logging.shutdown()