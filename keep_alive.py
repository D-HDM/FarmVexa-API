import time
import threading
import httpx
from config.settings import settings
from utils.logger import logger


class KeepAlive:
    def __init__(self):
        self.enabled = settings.KEEP_ALIVE_ENABLED
        self.app_url = settings.APP_URL
        self.interval = 600  # 10 minutes

    def start(self):
        if not self.enabled or not self.app_url:
            logger.info("Keep-alive disabled")
            return

        logger.info(f"Keep-alive enabled: {self.app_url}/api/health every {self.interval}s")

        def ping_loop():
            time.sleep(60)
            self.ping()

            while True:
                time.sleep(self.interval)
                self.ping()

        thread = threading.Thread(target=ping_loop, daemon=True)
        thread.start()

    def ping(self):
        try:
            response = httpx.get(f"{self.app_url}/api/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"Keep-alive ping OK")
            else:
                logger.warning(f"Keep-alive ping: {response.status_code}")
        except Exception as e:
            logger.error(f"Keep-alive ping error: {str(e)}")


keep_alive = KeepAlive()