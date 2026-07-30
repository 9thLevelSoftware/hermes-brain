# Code Review Guidelines

This file is the canonical code-review guidance for pull requests in this repository. Reviewers and authors should follow it for every PR.

## Mandate

Every pull request gets **two review passes**, in order:

1. A normal correctness review.
2. A mandatory **Ponytail review** focused on simplicity and size.

The Ponytail pass is not optional. Every PR review must end with a Ponytail section, even when the only correct output is:

> Ponytail: Lean already. Ship.

The goal is to keep the codebase correct, secure, maintainable, and as small as possible. Prefer the laziest solution that actually works: fewer files, fewer dependencies, fewer abstractions, fewer branches, fewer concepts.

## General Review Order

### 1. Understand the PR intent

- Read the title, description, linked issue, and changed files.
- Identify what behavior is supposed to change.
- Do not suggest simplification before understanding the real requirement.

### 2. Review correctness first

Look for bugs, broken edge cases, security issues, data-loss risks, race conditions, missing validation, bad error handling, broken tests, and regressions. Do not let the Ponytail pass remove necessary safety, validation, accessibility, observability, tests, or behavior the user explicitly requested.

### 3. Then run a dedicated Ponytail pass

Search the diff for unnecessary complexity. Apply these preferences, in order:

- Prefer deletion over addition.
- Prefer the standard library over hand-rolled code.
- Prefer platform or native framework features over dependencies.
- Prefer existing project patterns over new abstractions.
- Prefer one direct implementation over factories, registries, service layers, interfaces, adapters, or configuration that has only one use.
- Challenge speculative future-proofing.
- Flag code that exists "just in case."
- Flag abstractions with only one implementation.
- Flag wrappers around simple APIs.
- Flag dependencies used for trivial behavior.
- Flag duplicated helpers that the language, framework, or repo already provides.
- Flag generated boilerplate or broad scaffolding that this PR does not require.
- Flag tests that mostly exercise mocks, framework behavior, or implementation details rather than useful behavior.
- Flag documentation or comments that explain obvious code or defend unnecessary complexity.

## Ponytail Tags

Use these tags when reporting Ponytail findings:

- **delete** — dead code, unused flexibility, speculative feature, unnecessary branch, unused config, or scaffolding.
- **stdlib** — hand-rolled logic that the standard library already provides.
- **native** — dependency or custom code doing what the platform or framework already does.
- **yagni** — abstraction, config, or extension point with no current need.
- **shrink** — same behavior expressible with materially less code.
- **reuse** — new helper duplicates an existing project helper or pattern.
- **test-shrink** — test can be simpler while preserving meaningful coverage.

## Ponytail Finding Format

Each Ponytail finding must be concise and actionable:

```
<file>:L<line>: <tag> <what to cut>. <what replaces it>.
```

Examples:

- `src/cache.ts:L42:` **stdlib** — custom LRU cache. Replace with `Map` plus a size cap, or reuse the cache helper in `src/lib/cache.ts`.
- `app/services/UserService.ts:L18:` **yagni** — `IUserService` has one implementation and one caller. Delete the interface and inject `UserService` directly.
- `src/validators/email.ts:L7:` **native** — regex-based email parser. Use the platform/email validation already used in `FormInput`.
- `tests/user.test.ts:L88:` **test-shrink** — five mocked repository tests cover the same branch. Keep one behavior test through the public API.
- `src/config.ts:L31:` **delete** — `FEATURE_X_STRATEGY` has one value and no callers override it. Inline the value.

If there are no Ponytail findings, state exactly:

> Ponytail: Lean already. Ship.

Do not invent findings. If the code is already simple, say so.

## Important Boundaries

The Ponytail pass must not remove:

- Required input validation.
- Security checks.
- Error handling that prevents data loss or silent failure.
- Accessibility basics.
- Tests that protect non-trivial behavior.
- Logging or metrics that are operationally necessary.
- Behavior explicitly required by the PR or linked issue.

Other limits:

- Do not prefer clever one-liners over readable code when the readable version prevents mistakes.
- Do not block a PR only because the code could be shorter. Block only for correctness, security, data-loss, or maintainability risks.

## Review Output Format

Every review must use this structure.

### Verdict

One of:

- **Approve**
- **Request changes**
- **Comment only**

Followed by one short sentence explaining why.

### Correctness / Safety Findings

List only real correctness, safety, security, regression, or test issues.

Format:

```
<severity>: <file>:L<line>: <issue>. <required fix>.
```

Severities:

- **critical** — bug, security, or data-loss risk; must fix before merge.
- **important** — likely defect or maintainability hazard; should fix before merge.
- **minor** — small issue, typo, naming, or clarity problem.

If there are none, state:

> No correctness or safety findings.

### Ponytail Review

Always include this section, even if the only content is `Ponytail: Lean already. Ship.`

List findings in the `<file>:L<line>: <tag> ...` format described above.

End the section with a net line estimate:

```
Ponytail net: -<estimated removable lines> lines.
```

If nothing is removable:

```
Ponytail net: 0 lines.
```

### Suggested Minimal Patch

If there are actionable findings, describe the smallest safe patch set. Rules:

- Prefer the fewest files changed.
- Prefer deleting code.
- Do not introduce new dependencies unless absolutely necessary.
- Do not propose a broad refactor when a local fix solves the issue.
- Keep this section short.

If no patch is needed, state:

> No patch needed.

### Final Merge Guidance

State clearly whether the PR can merge. Examples:

- "Can merge after the critical finding is fixed."
- "Can merge; Ponytail suggestions are optional cleanup."
- "Do not merge until tests cover the changed behavior."
- "Can merge as-is."

## Behavioral Rules

- Be direct.
- Be specific.
- Do not write long essays.
- Do not praise boilerplate.
- Do not ask the author to "consider" vague changes.
- Every finding must identify exactly what should change.
- If a simplification is optional, mark it as optional.
- If a simplification is required because the complexity creates real risk, explain the risk in one sentence.
- Never treat a tool, test, or CI self-report as proof if the diff itself contradicts it.
- Prefer the smallest root-cause fix over patches scattered across callers.

## Mandatory Per-PR Checklist

Before posting a review, confirm:

- [ ] Did I review correctness and security first?
- [ ] Did I run a separate Ponytail pass?
- [ ] Did I look for code to delete?
- [ ] Did I look for stdlib or native replacements?
- [ ] Did I look for one-implementation interfaces, factories, or adapters?
- [ ] Did I look for speculative config or extensibility?
- [ ] Did I avoid removing required validation, security, or tests?
- [ ] Did I include either Ponytail findings or `Ponytail: Lean already. Ship.`?