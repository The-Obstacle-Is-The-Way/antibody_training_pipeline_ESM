# Branch Protection Setup - Visual Walkthrough

**Repository:** antibody_training_pipeline_ESM
**Date:** 2025-11-07
**Target Branch:** `leroy-jenkins/full-send` (default branch)

---

## 🎯 Goal

Set up branch protection rules on `leroy-jenkins/full-send` (our default branch) to ensure:
- All code goes through pull requests
- CI checks pass before merging
- Code reviews are required
- No accidental direct pushes

---

## 📋 Step-by-Step Instructions

### Step 1: Open GitHub Repository Settings

1. Go to: https://github.com/the-obstacle-is-the-way/antibody_training_pipeline_ESM
2. Click the **"Settings"** tab (top right, next to Insights)
3. You should see a left sidebar with many options

**Expected view:**
- Left sidebar: General, Access, Secrets and variables, Actions, etc.

---

### Step 2: Navigate to Branches

1. In the left sidebar, scroll down to **"Code and automation"** section
2. Click **"Branches"**
3. You'll see a section titled **"Branch protection rules"**

**Expected view:**
- Header: "Branch protection rules"
- Button: "Add branch protection rule" or "Add rule"
- Possibly empty if no rules exist yet

---

### Step 3: Add Branch Protection Rule

1. Click **"Add branch protection rule"** (green button)
2. You'll see a form with many options

**Expected view:**
- Top field: "Branch name pattern"
- Many checkboxes below with protection options

---

### Step 4: Configure Branch Name Pattern

**Field:** Branch name pattern

**Enter:** `leroy-jenkins/full-send`

**What this does:**
- Applies these rules ONLY to the `leroy-jenkins/full-send` branch (our default branch)
- You could use wildcards like `release/*` for multiple branches
- For now, we just want to protect our default production branch

**Expected state:**
- Text field shows: `leroy-jenkins/full-send`

---

### Step 5: Require Pull Request Before Merging

**Checkbox:** ✅ **Require a pull request before merging**

**Click this checkbox** - A sub-menu will expand with more options:

**Sub-options to configure:**

1. **Required number of approvals before merging**
   - **Set to:** `1`
   - **What this does:** At least 1 person must review and approve the PR before it can merge

2. ✅ **Dismiss stale pull request approvals when new commits are pushed**
   - **Check this**
   - **What this does:** If someone approves, then you push more code, the approval is invalidated (they need to re-review)

3. ✅ **Require review from Code Owners** (OPTIONAL - skip for now)
   - **Leave unchecked** (you don't have CODEOWNERS file yet)
   - **What this does:** If you had a CODEOWNERS file, specific people would be required to review specific files

4. ❌ **Require approval of the most recent reviewable push** (OPTIONAL)
   - **Leave unchecked** (default is fine)

5. ❌ **Require conversation resolution before merging** (RECOMMENDED)
   - **Check this if you want**
   - **What this does:** All PR comments must be marked as "resolved" before merging
   - **Recommendation:** ✅ Check it (good practice)

**Expected state after this step:**
- "Require a pull request before merging" is checked
- Required approvals: 1
- Dismiss stale approvals: checked
- Conversation resolution: checked (optional but recommended)

---

### Step 6: Require Status Checks to Pass Before Merging

**Checkbox:** ✅ **Require status checks to pass before merging**

**Click this checkbox** - A sub-menu will expand:

**Sub-options:**

1. ✅ **Require branches to be up to date before merging**
   - **Check this**
   - **What this does:** PR branch must be rebased/merged with latest `main` before merging
   - **Why:** Prevents merging stale code that might conflict with recent changes

2. **Status checks that are required** (IMPORTANT - but can't set yet!)
   - This is where you'd select which CI jobs must pass
   - **PROBLEM:** The list will be EMPTY until CI runs at least once!
   - **Solution:** We'll come back and add these after CI runs

**Expected state after this step:**
- "Require status checks to pass before merging" is checked
- "Require branches to be up to date" is checked
- Status checks list is empty (we'll add these later)

**NOTE:** You'll need to:
1. Save this rule first
2. Let CI run once on a PR
3. Come back and edit the rule to add specific status checks

---

### Step 7: Require Conversation Resolution (Already set in Step 5)

**Checkbox:** ✅ **Require conversation resolution before merging**

**If not already checked, check it now**

**What this does:**
- All PR review comments must be marked "Resolved" before merging
- Prevents merging code with unaddressed feedback

**Expected state:**
- Checkbox is checked

---

### Step 8: Require Signed Commits (OPTIONAL - Advanced)

**Checkbox:** ❌ **Require signed commits**

**Recommendation:** **Leave unchecked** for now

**What this does:**
- Requires all commits to be GPG-signed
- Adds cryptographic verification of commit author
- **Why skip:** Requires GPG setup, which is extra work

**Expected state:**
- Leave unchecked (unless you already have GPG set up)

---

### Step 9: Require Linear History (OPTIONAL)

**Checkbox:** ❌ **Require linear history**

**Recommendation:** **Leave unchecked** for now

**What this does:**
- Prevents merge commits (forces rebase or squash)
- Keeps git history linear
- **Why skip:** Can be annoying if you're not used to rebasing

**Expected state:**
- Leave unchecked (you can enable later if you want)

---

### Step 10: Include Administrators

**Checkbox:** ✅ **Include administrators**

**IMPORTANT:** **Check this**

**What this does:**
- These rules apply to EVERYONE, even repo admins (you!)
- If unchecked, you could bypass all protections (dangerous!)
- **Best practice:** Check it (forces you to follow your own rules)

**Expected state:**
- Checkbox is checked

---

### Step 11: Restrict Who Can Push to Matching Branches (OPTIONAL)

**Checkbox:** ❌ **Restrict who can push to matching branches**

**Recommendation:** **Leave unchecked** for now

**What this does:**
- Only specific people/teams can push to `main`
- Not needed if you're using "Require PR" (which already blocks direct pushes)

**Expected state:**
- Leave unchecked

---

### Step 12: Allow Force Pushes (DANGEROUS!)

**Checkbox:** ❌ **Allow force pushes**

**IMPORTANT:** **Leave UNCHECKED**

**What this does:**
- Allows `git push --force` to `main` (VERY dangerous!)
- Can rewrite history and break everything
- **Never enable this on `main`**

**Expected state:**
- Leave unchecked (default)

---

### Step 13: Allow Deletions (DANGEROUS!)

**Checkbox:** ❌ **Allow deletions**

**IMPORTANT:** **Leave UNCHECKED**

**What this does:**
- Allows deleting the `main` branch
- **Never enable this**

**Expected state:**
- Leave unchecked (default)

---

### Step 14: Review Your Configuration

**Before clicking "Create", verify:**

✅ Branch name pattern: `leroy-jenkins/full-send`
✅ Require pull request: YES (1 approval)
✅ Dismiss stale approvals: YES
✅ Require conversation resolution: YES (optional but recommended)
✅ Require status checks: YES
✅ Require up-to-date branches: YES
✅ Status checks list: EMPTY (will add after CI runs)
✅ Include administrators: YES
❌ Allow force pushes: NO
❌ Allow deletions: NO

---

### Step 15: Create the Rule

**Click:** **"Create"** button at the bottom

**What happens:**
- Rule is saved and active immediately
- You'll see the rule listed on the Branches page
- Direct pushes to `main` are now blocked

**Expected result:**
- You're redirected back to the Branches page
- You see: "Branch protection rule for leroy-jenkins/full-send" listed
- Status: Active (green checkmark)

---

## ✅ Verification - Test That It Works

### Test 1: Try to Push Directly to leroy-jenkins/full-send (Should Fail)

```bash
# Switch to default branch
git checkout leroy-jenkins/full-send

# Try to push
git push origin leroy-jenkins/full-send
```

**Expected error:**
```
remote: error: GH006: Protected branch update failed for refs/heads/leroy-jenkins/full-send.
remote: error: Changes must be made through a pull request.
```

**Result:** ✅ Direct push blocked (protection working!)

### Test 2: Create a Test PR (Should Work)

```bash
# Create test branch
git checkout -b test/branch-protection-verification

# Make a trivial change
echo "# Branch Protection Test" >> docs/test-protection.md

# Commit and push
git add docs/test-protection.md
git commit -m "test: Verify branch protection works"
git push origin test/branch-protection-verification
```

**Then:**
1. Go to GitHub
2. Create PR to `leroy-jenkins/full-send`
3. You should see:
   - ❌ CI checks pending/running
   - ❌ "Merging is blocked" (waiting for checks + approval)
   - ✅ "Review required" (1 approval needed)

**Result:** ✅ PR requires checks + approval (protection working!)

---

## 🔧 Step 16: Add Status Checks (After CI Runs)

**IMPORTANT:** You can't set required status checks until CI has run at least once.

**After your first PR with CI:**

1. Go back to: **Settings → Branches**
2. Click **"Edit"** on the `leroy-jenkins/full-send` branch protection rule
3. Scroll to **"Require status checks to pass before merging"**
4. In the search box under "Status checks that are required", you should now see:
   - `Code Quality (ruff, mypy, bandit)`
   - `Unit Tests (Python 3.12)`
   - `Integration Tests (Python 3.12)`
   - `CI Pipeline Success`
   - `Test Development Container`
   - `Test Production Container`

5. **Select these checks** by clicking each one
6. Click **"Save changes"** at the bottom

**Expected state:**
- 6 status checks are now required
- PRs can't merge unless all 6 pass

---

## 📊 What You've Accomplished

After setting up branch protection:

✅ **Security:**
- No one can push directly to `leroy-jenkins/full-send` (including you!)
- All changes go through pull requests
- Code must be reviewed

✅ **Quality:**
- CI must pass before merging
- Tests must pass
- Code must be formatted and linted

✅ **Process:**
- Conversations must be resolved
- Branch must be up-to-date
- Stale approvals are dismissed

✅ **Safety:**
- No force pushes allowed
- No branch deletion allowed
- Rules apply to everyone (including admins)

---

## 🚨 If You Need to Emergency-Fix Production

**Scenario:** Production is broken, you need to push a hotfix NOW.

**Option 1: Temporarily Disable Protection (NOT RECOMMENDED)**
1. Go to Settings → Branches
2. Click "Edit" on `leroy-jenkins/full-send` rule
3. Uncheck "Include administrators"
4. Save
5. Push your fix
6. **IMMEDIATELY re-enable** "Include administrators"

**Option 2: Use Emergency PR (RECOMMENDED)**
1. Create hotfix branch
2. Open PR to `leroy-jenkins/full-send`
3. Mark as "emergency"
4. Request expedited review
5. Merge when CI passes

**Best practice:** Always use Option 2 (emergency PR). Breaking protection should be a LAST resort.

---

## 📝 Summary Checklist

Before you start:
- [ ] You're logged into GitHub
- [ ] You have admin access to the repository
- [ ] You understand what branch protection does

During setup:
- [ ] Navigate to Settings → Branches
- [ ] Add branch protection rule for `leroy-jenkins/full-send`
- [ ] Require pull requests (1 approval)
- [ ] Require status checks (but list is empty until CI runs)
- [ ] Include administrators
- [ ] Block force pushes and deletions
- [ ] Create the rule

After setup:
- [ ] Test that direct push fails
- [ ] Create test PR and verify it's blocked until checks pass
- [ ] After first CI run, add required status checks to the rule
- [ ] Celebrate 🎉

---

## 🔗 GitHub Documentation

- [About protected branches](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [Managing a branch protection rule](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/managing-a-branch-protection-rule)

---

**Document Version:** 1.0
**Last Updated:** 2025-11-07
**Status:** Ready to follow
**Estimated Time:** 10 minutes
