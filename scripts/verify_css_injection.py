"""Verify if inline CSS injection worked on HF Spaces."""

import asyncio

from playwright.async_api import async_playwright


async def verify_css_injection() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        url = "https://huggingface.co/spaces/VibecoderMcSwaggins/antibody-predictor"
        print(f"🌐 Navigating to {url}...")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        print("⏳ Waiting 10s for Gradio...")
        await page.wait_for_timeout(10000)

        print("\n🎨 Checking for <style> tags...")
        style_tags = await page.query_selector_all("style")
        print(f"Found {len(style_tags)} <style> tags")

        if style_tags:
            for i, tag in enumerate(style_tags[:5]):
                content = await tag.inner_text()
                if "gradio-container" in content or "status-card" in content:
                    print(f"  ✅ Tag {i + 1} has our CSS! Preview: {content[:80]}...")

        print("\n🔍 Checking custom classes...")
        for cls in [".status-card", ".header-title", ".gradio-container"]:
            elem = await page.query_selector(cls)
            print(f"  {'✅' if elem else '❌'} {cls}")

        await page.screenshot(
            path="experiments/debug_screenshots/mcp_verification.png", full_page=True
        )
        print("\n📸 Screenshot: experiments/debug_screenshots/mcp_verification.png")

        print("\n🔍 Keeping browser open 15s...")
        await page.wait_for_timeout(15000)
        await browser.close()


asyncio.run(verify_css_injection())
