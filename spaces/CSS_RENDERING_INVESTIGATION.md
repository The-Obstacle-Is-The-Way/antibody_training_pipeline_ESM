# CSS Rendering Issue - First Principles Investigation

**Date**: 2025-11-22
**Status**: 🚨 CRITICAL - Black rectangles, CSS not loading
**Deployment**: HF Spaces (Gradio 5.0.0)

---

## 🎯 OBSERVED SYMPTOMS

1. **Black rectangles** covering large portions of the UI
2. **Custom CSS not injected** - Playwright confirmed 0 `<style>` tags
3. **Custom classes missing** - `.status-card`, `.header-title`, `.gradio-container` NOT found
4. **Gradio logo visible** - Base Gradio infrastructure loads
5. **Footer text visible** - Some HTML renders correctly
6. **100+ HTTP errors** - 502 errors, HTTP/2 protocol errors, failed module imports

---

## 🧠 FIRST PRINCIPLES ANALYSIS

### What We Know FOR CERTAIN:

1. ✅ **Code is correct locally** - `spaces/app.py` has all the right imports, CSS, gr.Blocks setup
2. ✅ **Deployment successful** - Files pushed to HF Spaces (commit `7010654`)
3. ✅ **Space is "Running"** - HF API shows `"stage": "RUNNING"`
4. ✅ **Gradio 5.0.0 specified** - Matches our development version
5. ❌ **CSS not rendering** - Despite being passed to `gr.Blocks(css=css)`

### The Core Mystery:

**Why does `gr.Blocks(css=css)` work locally but fail on HF Spaces?**

---

## 🔬 WEB SEARCH FINDINGS (Nov 2025)

### Critical Discovery #1: **HF Spaces CSS Loading Limitation**

> **"When loading demos from Spaces, any attributes that apply to the entire Blocks, such as the theme or custom CSS/JS, will not be loaded"**
> — [Gradio Docs: Using Hugging Face Integrations](https://www.gradio.app/guides/using-hugging-face-integrations)

**Implication**: HF Spaces may strip or ignore the `css` parameter in `gr.Blocks()` for security/isolation reasons.

### Critical Discovery #2: **External CSS Loading Bug (Gradio 5+)**

> **"External CSS files do not load in Gradio Blocks interfaces, while inline CSS styles work correctly. This issue did not occur in previous versions."**
> — [Unable to Load External CSS Files in Gradio Blocks · Issue #9613](https://github.com/gradio-app/gradio/issues/9613)

**Implication**: Even if we used external CSS files, they wouldn't load in Gradio 5.0.

### Critical Discovery #3: **Gradio 5 Rendering Bugs**

> **"Bugs after Gradio 5 where some theme classes are not applying properly and some components are rendering bugged"**
> — [[Gradio 5] Some bugs with elements and css · Issue #9671](https://github.com/gradio-app/gradio/issues/9671)

**Implication**: Gradio 5.0.0 has active CSS rendering bugs affecting themes and custom styles.

### Critical Discovery #4: **IFrame CSS Conflicts**

> **"CSS in the parent page can affect embedded Gradio apps, with element selectors like `header { ... }` and `footer { ... }` being most likely to cause issues."**
> — [Sharing Your App - Gradio Docs](https://www.gradio.app/guides/sharing-your-app)

**Implication**: HF Spaces' wrapper page CSS might be conflicting with our app's layout.

### Critical Discovery #5: **IFrame Resizer Issues**

> **"Fix various iFrame related UI issues when deploying to spaces"**
> — [PR #11749 - gradio-app/gradio](https://github.com/gradio-app/gradio/pull/11749)

**Implication**: Active work on iframe rendering bugs suggests ongoing issues with HF Spaces deployment.

---

## 🧪 ROOT CAUSE HYPOTHESIS (Ranked by Likelihood)

### **MOST LIKELY: HF Spaces IFrame CSS Stripping**

**Theory**: HF Spaces embeds Gradio apps in an iframe and strips the `css` parameter for security/isolation.

**Evidence**:
- Official docs state "custom CSS/JS will not be loaded" when loading from Spaces
- Playwright found 0 `<style>` tags despite CSS being defined
- Base Gradio components render (logo, footer) but custom styling doesn't apply

**Test**: Try injecting CSS via `gr.HTML()` components instead of `gr.Blocks(css=css)`

---

### **LIKELY: Gradio 5.0.0 CSS Rendering Bug**

**Theory**: Gradio 5.0.0 has a regression where the `css` parameter doesn't inject styles correctly on HF Spaces.

**Evidence**:
- GitHub Issue #9671 reports CSS bugs in Gradio 5
- Issue #9613 reports external CSS files not loading
- Our Playwright logs show failed module imports suggesting frontend errors

**Test**: Downgrade to Gradio 4.44.0 (last stable 4.x) and redeploy

---

### **POSSIBLE: CSS Syntax/Conflict Issue**

**Theory**: Our CSS has syntax errors or conflicts with HF Spaces' parent page CSS.

**Evidence**:
- IFrame CSS conflicts documented in Gradio docs
- We use broad selectors like `.gradio-container` that might clash

**Test**: Use more specific selectors with `!important` flags

---

### **LESS LIKELY: Build/Deploy Failure**

**Theory**: The app.py didn't actually deploy correctly despite success message.

**Evidence**:
- We fetched app.py from HF Spaces and it matched our local version
- Space shows "Running" status
- Files are present in HF Spaces repo

**Test**: Manual verification via HF Spaces "Files" tab

---

## 🛠️ PROPOSED SOLUTIONS (Prioritized)

### **Solution 1: Inject CSS via gr.HTML() Components** ⭐ RECOMMENDED

**Approach**: Instead of `gr.Blocks(css=css)`, inject CSS using `gr.HTML()` with inline `<style>` tags.

**Rationale**:
- Web search confirms inline CSS in gr.HTML works when `css` parameter doesn't
- This bypasses any HF Spaces CSS stripping
- Follows "inline CSS works correctly" pattern from Issue #9613

**Implementation**:
```python
with gr.Blocks(theme=gr.themes.Soft(), title="Antibody Predictor") as app:
    # Inject CSS via HTML component
    gr.HTML(f\"\"\"
    <style>
    {css}  # Our existing CSS string
    </style>
    \"\"\")

    # Rest of app...
```

**Risk**: Low - documented workaround

---

### **Solution 2: Downgrade to Gradio 4.44.0**

**Approach**: Change `requirements.txt` from `gradio>=5.0.0` to `gradio==4.44.0`

**Rationale**:
- Gradio 5 has documented CSS rendering bugs
- Gradio 4.x is stable and widely used on HF Spaces
- Many production Spaces use Gradio 4.x successfully

**Implementation**:
```txt
# requirements.txt
gradio==4.44.0  # Last stable 4.x release
```

**Risk**: Medium - might lose Gradio 5 features, but we're not using any

---

### **Solution 3: Use External Theme from HF Hub**

**Approach**: Use a pre-built theme from Hugging Face Hub instead of custom CSS

**Rationale**:
- Official HF themes are guaranteed to work on Spaces
- Avoids CSS injection issues entirely

**Implementation**:
```python
import gradio as gr
from gradio.themes.utils import colors, fonts

theme = gr.themes.Soft(
    primary_hue=colors.blue,
    secondary_hue=colors.purple,
    font=[fonts.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=theme) as app:
    ...
```

**Risk**: High - lose custom styling control (no gradient headers, custom cards)

---

### **Solution 4: Use `elem_classes` + CSS File**

**Approach**: Define CSS in separate file, add to git LFS, reference via `elem_classes`

**Rationale**:
- Separates concerns
- Might bypass inline CSS issues

**Implementation**:
```python
with gr.Blocks(theme=gr.themes.Soft(), css="custom.css") as app:
    gr.Markdown("...", elem_classes="header-title")
```

**Risk**: High - Issue #9613 says external CSS files DON'T work in Gradio 5

---

## 📊 DIAGNOSTICS TO RUN

Before implementing fixes, gather more data:

1. **Check HF Spaces Build Logs**
   - Look for Python errors during app initialization
   - Check if CSS file is being read correctly

2. **Inspect Deployed app.py**
   - Verify the deployed file matches our local version
   - Check if HF Spaces modified anything during deployment

3. **Test Gradio Version**
   - Confirm HF Spaces is using Gradio 5.0.0 (not auto-upgrading)

4. **Browser DevTools on Live Space**
   - Check Network tab for failed CSS requests
   - Check Console for JavaScript errors
   - Inspect DOM to see if `<style>` tags exist

---

## 🎯 RECOMMENDED ACTION PLAN

**Phase 1: Quick Fix (15 minutes)**
1. Try Solution 1 (gr.HTML CSS injection)
2. Redeploy to HF Spaces
3. Test if black rectangles disappear

**Phase 2: If Quick Fix Fails (30 minutes)**
1. Downgrade to Gradio 4.44.0 (Solution 2)
2. Redeploy
3. Test rendering

**Phase 3: If Both Fail (1 hour)**
1. Deep dive into HF Spaces build logs
2. Test minimal Gradio Blocks example on HF Spaces
3. Open issue with Gradio team if confirmed bug

---

## 🔗 SOURCES

- [Gradio not loading css when using Blocks - HF Forums](https://discuss.huggingface.co/t/gradio-not-loading-css-when-using-blocks/56755)
- [Unable to Load External CSS Files in Gradio Blocks · Issue #9613](https://github.com/gradio-app/gradio/issues/9613)
- [[Gradio 5] Some bugs with elements and css · Issue #9671](https://github.com/gradio-app/gradio/issues/9671)
- [Custom CSS And JS - Gradio Docs](https://www.gradio.app/guides/custom-CSS-and-JS)
- [Using Hugging Face Integrations - Gradio Docs](https://www.gradio.app/guides/using-hugging-face-integrations)
- [fix various iFrame related UI issues · PR #11749](https://github.com/gradio-app/gradio/pull/11749)

---

## 💡 NEXT STEPS

**For AI Agent Handoff:**

1. Read this investigation report
2. Implement Solution 1 (gr.HTML CSS injection) first
3. Test locally with `python spaces/app.py`
4. If works locally, redeploy to HF Spaces
5. If still fails, try Solution 2 (Gradio downgrade)
6. Report back with results + screenshots

**Required Context:**
- `spaces/app.py` (current implementation)
- This investigation report
- Access to HF Spaces deployment script (`spaces/deploy_cli.sh`)
