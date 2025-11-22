"""Verify live HF Space styling (checks inline styles and captures screenshot)."""

import asyncio

from playwright.async_api import async_playwright


async def verify_css_injection() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, slow_mo=500)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})

        url = "https://VibecoderMcSwaggins-antibody-predictor.hf.space"
        print(f"🌐 Navigating to {url}...")
        await page.goto(url, wait_until="domcontentloaded", timeout=60000)

        print("⏳ Waiting 10s for Gradio...")
        await page.wait_for_timeout(10000)

        print("\n🎨 Checking inline styles...")
        title = await page.query_selector("text=Antibody Non-Specificity Predictor")
        if title:
            title_color = await title.evaluate("el => getComputedStyle(el).color")
            print(f"  ✅ Title color: {title_color}")
        else:
            print("  ❌ Title not found")

        status_text = await page.query_selector("text=Ready to Predict")
        if status_text:
            style_info = await status_text.evaluate(
                """
                el => {
                    let node = el;
                    while (node && node !== document.body) {
                        const style = getComputedStyle(node);
                        if (style.backgroundColor && style.borderRadius) {
                            return {
                                bg: style.backgroundColor,
                                color: style.color,
                                border: style.border,
                            };
                        }
                        node = node.parentElement;
                    }
                    return null;
                }
                """
            )
            print(f"  ✅ Status card styles: {style_info}")
        else:
            print("  ❌ Status card not found")

        await page.screenshot(
            path="experiments/debug_screenshots/mcp_verification.png", full_page=True
        )
        print("\n📸 Screenshot: experiments/debug_screenshots/mcp_verification.png")

        print("\n🔍 Keeping browser open 15s...")
        await page.wait_for_timeout(15000)
        await browser.close()


asyncio.run(verify_css_injection())
