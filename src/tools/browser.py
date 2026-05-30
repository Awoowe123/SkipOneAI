"""Browser tool using Playwright"""
import logging
import asyncio
from typing import Dict, Any, Optional
from src.config import Config

logger = logging.getLogger(__name__)

class BrowserTool:
    """Tool for interacting with web pages via Playwright"""

    async def open_page(self, url: str, js_code: Optional[str] = None) -> Dict[str, Any]:
        """
        Open URL and optionally execute JavaScript

        Args:
            url: URL to open
            js_code: JavaScript code to execute

        Returns:
            Page details and JS result
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            return {
                "success": False,
                "message": "Playwright не установлен. Установите: pip install playwright && playwright install"
            }

        try:
            logger.info(f"Opening browser: {url}")

            async with async_playwright() as p:
                # Launch browser
                # Note: On first run 'playwright install' might be needed
                browser = await p.chromium.launch(
                    headless=Config.browser.headless,
                    executable_path=Config.browser.browser_path if Config.browser.browser_path else None
                )

                context = await browser.new_context(
                    viewport={'width': 1280, 'height': 720},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )

                page = await context.new_page()

                # Navigate
                try:
                    response = await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    status = response.status if response else 0
                except Exception as e:
                    await browser.close()
                    return {"success": False, "error": f"Navigation failed: {e}"}

                page_title = await page.title()

                result_data = {
                    "success": True,
                    "url": url,
                    "title": page_title,
                    "status": status
                }

                # Execute JS if provided
                if js_code:
                    logger.info("Executing JS code...")
                    try:
                        # Wrap in try-catch in JS to prevent crash
                        js_result = await page.evaluate(js_code)
                        result_data["js_result"] = str(js_result)
                    except Exception as e:
                        result_data["js_error"] = str(e)

                # Get basic content (first 500 chars)
                try:
                    content = await page.inner_text('body')
                    result_data["content_preview"] = content[:500] + "..."
                except:
                    pass

                await browser.close()
                return result_data

        except Exception as e:
            logger.error(f"Browser error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

# Schema
BROWSER_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "URL для открытия (http/https)"
        },
        "js_code": {
            "type": "string",
            "description": "JavaScript код для выполнения в консоли браузера. Например: 'document.title' или 'navigator.userAgent'"
        }
    },
    "required": ["url"]
}
