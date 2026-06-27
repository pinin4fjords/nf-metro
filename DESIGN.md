---
version: "2.0"
name: Seqera
description: "Seqera design system v2 — three-layer token architecture (Primitives → Decision → Component). Fully neutral dark canvas, single green accent (action-bg). Degular display / Inter body, sharp shapes (0px cards, 4px buttons). Green at most 2–3× per screen on primary CTAs only."

# ── Token Architecture ──────────────────────────────────────────────────────────
# Three layers — NEVER skip layers:
#
#   1. Primitives   Raw hex values grouped by hue scale.
#                   e.g. Green/500 = #31C9AC, Brand/dark = #201637, Grey/11 = #242424
#                   Figma: Seqera/Primitives
#
#   2. Decision     Semantic tokens that describe WHAT a color IS (its role),
#                   never which component uses it.
#                   e.g. surface-page, text-default, action-bg, border-focus
#                   Figma: Seqera/Decision Colors
#
#   3. Component    Composition of Decision tokens into component definitions.
#                   e.g. button-primary, card, input, eyebrow-badge
#                   Defined in this file under `components:`
#
# Naming principle: Decision tokens are named by semantic ROLE, not component variant.
#   ✅ action-bg-emphasis   (describes: "dark emphasis fill for any action element")
#   ❌ button-secondary-bg  (describes: "one component" — breaks reuse)
#
# Figma collections: Seqera/Primitives · Seqera/Decision Colors · Seqera/Brand Tokens
# UI Playground file key: MBj38MxoDPSXJ4tzgicfy4

colors:
  # ── Surface — Seqera/Decision Colors > Surface ─────────────────────────────
  # Dark mode surface depth (deepest → highest elevation):
  #   surface-inset  →  surface-page  →  surface-subtle  →  (surface-brand for brand sections)
  #
  surface-page: "#FFFFFF" # Light: #FFFFFF / Dark: #242424 (Grey/1100 — neutral base canvas)
  surface-subtle: "#F7F7F7" # Light: #F7F7F7 (Grey/100) / Dark: #181818 (Grey/1200 — fully neutral, no purple tint)
  surface-inset: "#EAEBEB" # Light: #EAEBEB (Grey/200) / Dark: #242424 (Grey/1100 — code blocks, wells, inputs)
  surface-brand: "#E2F7F3" # Light: #E2F7F3 (Green/100) / Dark: #065647 (Green/1000 — expressive brand sections: footer, dark hero)
  surface-overlay: "#EAEBEB" # Light: #EAEBEB (Grey/200) / Dark: #242424 (Grey/1100 — modal scrim backdrop, hover wash)
  surface-invert: "#201637" # Light: #201637 (Dark-brand) / Dark: #FFFFFF — inverted high-contrast block

  # Deprecated v1 aliases — DO NOT use in new code. Use the tokens above.
  # surface-dark: "#201637"            → use surface-invert (light)
  # surface-elevated-dark: "#2D273C"   → removed; dark mode is now fully neutral grey — use surface-subtle dark (#181818)
  # surface-page-dark: "#160F26"       → removed; use surface-inset dark (#242424)

  # ── Text — Seqera/Decision Colors > Text ────────────────────────────────────
  text-default: "#201637" # Light: #201637 (Dark-brand) / Dark: #FFFFFF — body copy, headlines, UI labels
  text-muted: "#BABCBD" # Light: #BABCBD (Grey/600) / Dark: #CFD0D1 (Grey/400) — metadata, timestamps, helper text
  text-inverse: "#FFFFFF" # Light: #FFFFFF / Dark: #201637 (Dark-brand) — text on dark surfaces
  text-emphasized: "#065647" # Light: #065647 (Green/1000) / Dark: #31C9AC (Green/500) — featured text, callouts, eyebrow badges
  text-link: "#065647" # Light: #065647 (Green/1000) / Dark: #31C9AC (Green/500) — hyperlinks ONLY
  # text-green removed — use text-emphasized (same values: #065647 light / #31C9AC dark)

  # ── Border — Seqera/Decision Colors > Border ────────────────────────────────
  border-default: "#DDDEDE" # Light: #DDDEDE (Grey/300) / Dark: #919393 (Grey/800)
  border-muted: "#EAEBEB" # Light: #EAEBEB (Grey/200) / Dark: #545555 (Grey/1000)
  border-focus: "#065647" # Light: #065647 (Green/1000) / Dark: #31C9AC (Green/500) — focus ring: always 2px solid, 2px offset
  # border-dark removed — was alias for border-default dark; use border-default

  # ── Action / CTA — Seqera/Decision Colors > Action ──────────────────────────
  # Pairing rule: action-text (#201637) ONLY pairs with action-bg (green).
  # When using action-bg-emphasis (dark fill) always use text-inverse (white) — not action-text.
  #
  action-bg: "#31C9AC" # Primary CTA fill — Green/500. Light & Dark (same).
  action-bg-hover: "#0CAE8E" # Hover of action-bg. Light: #0CAE8E (Green/700) / Dark: #56D3BA (Green/400)
  # action-bg-pressed removed — not in Figma variables
  action-bg-emphasis: "#201637" # Emphasis fill: secondary/dark action button. Light: #201637 / Dark: #201637 (same both modes)
  # action-bg-emphasis-hover removed — not in Figma variables
  # action-text: "#201637" — gap: not yet a Figma variable. Use text-default (#201637) as fallback until created.
  action-border: "#31C9AC" # Outline/border for action — Green/500. Light & Dark.
  action-accent: "#0A967B" # Active states, progress, running indicator. Light: #0A967B (Green/800) / Dark: #56D3BA (Green/400)

  # link-default removed — use text-link (same value)

  # ── Disabled — Seqera/Decision Colors > Disabled ────────────────────────────
  disabled-bg: "#F7F7F7" # Light: #F7F7F7 (Grey/100) / Dark: #7B7B7B (Grey/900)
  disabled-text: "#BABCBD" # Light: #BABCBD (Grey/600) / Dark: #BABCBD (Grey/600 — same both modes)

  # neutral-subtle / neutral-emphasis removed — not in Figma variables

  # ── Semantic — Error  (Seqera/Primitives > Red → Decision Colors > Error) ───
  error-surface: "#FBE7E9" # Light: Red/100  / Dark: #63181F (Red/1000)
  error-border: "#EE9AA2" # Light: Red/300  / Dark: #C7303E (Red/700)
  error-default: "#DC3545" # Light: #DC3545 (Red/600) / Dark: #E7727D (Red/400)
  error-text: "#AC2936" # Light: Red/800  / Dark: #F5C2C7 (Red/200)

  # ── Semantic — Warning  (Seqera/Primitives > Yellow → Decision Colors > Warning)
  warning-surface: "#FFF4E0" # Light: Yellow/100   / Dark: #734A00 (Yellow/1000)
  warning-border: "#FFD280" # Light: Yellow/300   / Dark: #E79500 (Yellow/700)
  warning-default: "#FFA500" # Light: #FFA500 (Yellow/600) / Dark: #FFC04D (Yellow/400)
  warning-text: "#A96D00" # Light: Yellow/900   / Dark: #FFE4B3 (Yellow/200)

  # ── Semantic — Success  (Seqera/Primitives > Functional-Green → Decision Colors > Success)
  success-surface: "#E5F5EC" # Light: Functional-Green/100  / Dark: #124E2C (Functional-Green/1000)
  success-border: "#94D7B0" # Light: Functional-Green/300  / Dark: #249D58 (Functional-Green/700)
  success-default: "#28AE61" # Light: #28AE61 (Func-Green/600) / Dark: #69C690 (Func-Green/400)
  success-text: "#1F884C" # Light: Functional-Green/800  / Dark: #BFE7D0 (Functional-Green/200)

  # Info / interaction blue (functional blue scale) — pending decision
  # info-surface: "#E8EBFC"            # step 1
  # info-default: "#4256E7"            # step 6
  # info-text: "#3443B4"               # step 8

typography:
  display:
    fontFamily: "'Degular', 'Bricolage Grotesque', Inter, 'Helvetica Neue', sans-serif"
    fontSize: 68px
    fontWeight: 600
    lineHeight: 1.0
    letterSpacing: -0.025em
  hero:
    fontFamily: "'Degular', 'Bricolage Grotesque', Inter, 'Helvetica Neue', sans-serif"
    fontSize: 56px
    fontWeight: 600
    lineHeight: 1.04
    letterSpacing: -0.025em
  h1:
    fontFamily: "'Degular', 'Bricolage Grotesque', Inter, 'Helvetica Neue', sans-serif"
    fontSize: 40px
    fontWeight: 600
    lineHeight: 1.1
    letterSpacing: -0.025em
  h2:
    fontFamily: "'Degular', 'Bricolage Grotesque', Inter, 'Helvetica Neue', sans-serif"
    fontSize: 32px
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: -0.025em
  h3:
    fontFamily: "'Degular', 'Bricolage Grotesque', Inter, 'Helvetica Neue', sans-serif"
    fontSize: 28px
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: -0.025em
  h4:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 24px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: 0
  h5:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 18px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0
  h6:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 16px
    fontWeight: 600
    lineHeight: 1.4
    letterSpacing: 0
  intro:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 20px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  body:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  small:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  blockquote:
    fontFamily: "'Inter', 'Helvetica Neue', sans-serif"
    fontSize: 20px
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: 0
  mono:
    fontFamily: "'JetBrains Mono', 'Roboto Mono', Monaco, monospace"
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: 0

rounded:
  none: 0px
  xs: 4px
  md: 8px # reserved — intentionally unused, guards against 8px SaaS default creep
  screenshot: 16px
  badge: 20px
  full: 9999px

spacing:
  xxs: 4px
  xs: 8px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
  section-mobile: 48px
  section: 80px
  container: 1280px
  body-text-max: 680px

components:
  button-primary:
    backgroundColor: "{colors.action-bg}"
    textColor: "{colors.action-text}"
    typography: "{typography.small}"
    fontWeight: "500"
    rounded: "{rounded.xs}"
    padding: "6px 18px"
    border: "1px solid {colors.action-bg-hover}"
  button-primary-hover:
    backgroundColor: "{colors.action-bg-hover}"
    border: "1px solid {colors.action-bg-pressed}"
  button-secondary:
    backgroundColor: "{colors.action-bg-emphasis}"
    textColor: "{colors.text-inverse}"
    typography: "{typography.small}"
    fontWeight: "500"
    rounded: "{rounded.xs}"
    padding: "6px 18px"
    border: "1px solid {colors.border-default}"
  button-secondary-hover:
    backgroundColor: "{colors.action-bg-emphasis-hover}"
  button-small:
    padding: "4px 12px"
    rounded: "{rounded.xs}"
    typography: "{typography.small}"

  card:
    backgroundColor: "{colors.surface-page}"
    textColor: "{colors.text-default}"
    rounded: "{rounded.none}"
    border: "1px solid {colors.border-default}"
    padding: "{spacing.lg}"
  card-dark:
    backgroundColor: "{colors.surface-brand}"
    textColor: "{colors.text-inverse}"
    rounded: "{rounded.none}"
    border: "1px solid {colors.border-default}"
    padding: "{spacing.lg}"
  input:
    backgroundColor: "{colors.surface-page}"
    textColor: "{colors.text-default}"
    typography: "{typography.body}"
    rounded: "{rounded.xs}"
    border: "1px solid {colors.border-default}"
    padding: "8px 16px"
  input-focus:
    border: "1px solid {colors.border-focus}"
    outline: "2px solid {colors.border-focus}"
    outlineOffset: 2px
  input-dark:
    backgroundColor: "{colors.surface-subtle}"
    textColor: "{colors.text-inverse}"
    border: "1px solid {colors.border-default}"
  eyebrow-badge:
    backgroundColor: "{colors.surface-brand}"
    textColor: "{colors.text-emphasized}"
    typography: "{typography.small}"
    fontWeight: "500"
    textTransform: uppercase
    letterSpacing: 1px
    rounded: "{rounded.badge}"
    padding: "3px 8px"
  topbar:
    backgroundColor: "{colors.surface-page}"
    textColor: "{colors.text-default}"
    height: 64px
    border: "1px solid {colors.border-default}"
  topbar-dark:
    backgroundColor: "{colors.surface-brand}"
    textColor: "{colors.text-inverse}"
    height: 64px
    border: "1px solid {colors.border-default}"
  sidebar:
    backgroundColor: "{colors.surface-page}"
    textColor: "{colors.text-default}"
    width: 220px
    border: "1px solid {colors.border-default}"
  sidebar-dark:
    backgroundColor: "{colors.surface-brand}"
    textColor: "{colors.text-inverse}"
    width: 220px
    border: "1px solid {colors.border-default}"
  modal:
    backgroundColor: "{colors.surface-page}"
    textColor: "{colors.text-default}"
    rounded: "{rounded.none}"
    border: "1px solid {colors.border-default}"
    maxWidth: 640px
    padding: "{spacing.lg}"
    shadow: "0 25px 50px -12px rgba(0,0,0,0.25)"
  modal-wide:
    maxWidth: 960px
  product-screenshot:
    rounded: "{rounded.screenshot}"
    border: "1px solid {colors.border-default}"
    shadow: "0 25px 50px -12px rgba(0,0,0,0.25)"
  code-block:
    backgroundColor: "{colors.surface-inset}"
    textColor: "{colors.text-default}"
    typography: "{typography.mono}"
    rounded: "{rounded.xs}"
    border: "1px solid {colors.border-default}"
    padding: "{spacing.lg}"
  code-block-dark:
    backgroundColor: "{colors.surface-inset}"
    textColor: "{colors.text-inverse}"
    typography: "{typography.mono}"
    rounded: "{rounded.xs}"
    border: "1px solid {colors.border-default}"
    padding: "{spacing.lg}"
---

## Overview

Seqera is the company behind Nextflow and the Seqera Platform — infrastructure for scientific computing at scale. The design system reflects this: precision over decoration, structure over expressiveness, real product data over abstract illustration. The audience is computational biologists, engineers, and platform teams who distrust marketing aesthetics. The design earns trust by looking like it was built by people who understand the problem, not by people who want to impress.

The base canvas in light mode is **pure white** (`{colors.surface-page}` — #FFFFFF), with all text running in **Brand Dark** (`{colors.text-default}` — #201637). **Seqera Green** (`{colors.action-bg}` — #31C9AC) is the single brand voltage: it appears on the primary CTA, active/running pipeline state indicator, success confirmation, logo mark dot, and progress bar fill — at most 2–3 times per screen. It is never a background on large areas and never used as text on light surfaces (contrast ratio 3.83:1 fails WCAG AA).

In dark mode, the entire surface stack is **fully neutral grey** — no purple tint. The page floor is `{colors.surface-page}` (#242424, Grey/1100). Cards and panels elevate to `{colors.surface-subtle}` (#181818, Grey/1200 — a deeper neutral). Brand and hero sections use `{colors.surface-brand}` (#065647, Green/1000) — deep green, not purple. The teal (`{colors.action-bg}`) carries unchanged across both modes — it is the constant.

Type is set in **Degular** for all display sizes (`{typography.display}` through `{typography.h3}`) and **Inter** for all body and UI copy (`{typography.h4}` through `{typography.small}`). Degular runs at weight 600 with `letter-spacing: -0.025em` — the negative tracking is not optional; it is the single typographic detail that separates Seqera's display type from a generic bold heading.

The shape language is **structural**. Cards, panels, containers, and layout elements hold at `{rounded.none}` (0px). Only interactive elements — buttons and inputs — earn `{rounded.xs}` (4px). Product screenshots get `{rounded.screenshot}` (16px) plus a prominent drop-shadow.

**Key characteristics:**

- Single green accent (`{colors.action-bg}`) appears at most 2–3 times per screen; color is structural, not decorative
- Dark mode is fully neutral grey — `{colors.surface-page}` #242424 (floor) → `{colors.surface-subtle}` #181818 (cards/panels). Brand sections use deep green (`{colors.surface-brand}` #065647), not purple
- Aggressive typographic scale — hero pairs with small directly, no intermediate buffer
- 0px card radius — structural, not bubbly; rounding reserved for interactive elements only
- Asymmetric layouts — visual at 55–60%, text at 40–45%; centered heroes are explicitly prohibited
- Data and product screenshots are the visual hero, never decorative illustration

---

## Token Architecture

### Three Layers

The Seqera token system is organized in three strict layers. Each layer references only the layer immediately below it — no skipping.

```
Seqera/Primitives
  └── Green/500 = #31C9AC
  └── Brand/dark = #201637
  └── Grey/11 = #242424

Seqera/Decision Colors
  └── action-bg → Green/500
  └── surface-brand (dark) → Brand/dark
  └── surface-page (dark) → Grey/11

Components (code / DESIGN.md)
  └── button-primary.backgroundColor → {colors.action-bg}
  └── card-dark.backgroundColor → {colors.surface-brand}
```

**Layer 1 — Primitives:** Raw hex values organized by hue scale (Green/100–1100, Brand/100–1100, Grey/1–12, Red, Yellow, Functional-Green). Audience: design system maintainers only. Not used directly in components or code.

**Layer 2 — Decision Colors:** Semantic tokens that answer "what is this color for?" Names describe the role, not the component. Audience: UI designers and developers building components. This is the layer you use in code.

**Layer 3 — Component Tokens:** Compose Decision tokens into complete component definitions. These define the surface, text, border, radius, and spacing for each component. Defined in the `components:` section of this file.

### Naming Principle

Decision token names describe what the color **IS** (its semantic role), never which component uses it.

```
✅ action-bg-emphasis   — "dark fill used for emphasis in any action context"
✅ surface-inset        — "a surface that sits below the page level"
✅ text-emphasized      — "emphasized text color, more prominent than muted"

❌ button-secondary-bg  — tied to one component, breaks reuse across others
❌ nav-active-bg        — too specific, prevents use on other active patterns
❌ dark-card-surface    — describes visual appearance, not semantic role
```

### Brand Tokens vs Decision Colors

These are **separate collections** serving different audiences.

| Collection                 | Figma                               | Audience                                          | Purpose                                              |
| -------------------------- | ----------------------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| **Seqera/Brand Tokens**    | Brand Colors + Neutrals             | Brand designers, marketers, presentation creators | Simplified palette for brand assets, slides, print   |
| **Seqera/Decision Colors** | Surface, Text, Border, Action, etc. | UI designers, developers, component authors       | All UI decisions — the only layer referenced in code |

Brand Tokens are NOT used in UI components. Decision Colors are NOT used in brand/marketing assets outside the digital product.

---

## Colors

### Brand & Accent

- **Seqera Green** (`{colors.action-bg}` — #31C9AC): The single brand accent. Primary CTA background, active/running pipeline state, logo mark dot, progress bar fill. At most 2–3 times per screen. Never on large backgrounds. Never as text on light surfaces (WCAG AA fail).
- **Action Hover** (`{colors.action-bg-hover}` — #0CAE8E): Hover state of the primary green button. Same value as `{colors.action-accent}` in light mode — they share the Green/600 primitive.
- **Action Pressed** (`{colors.action-bg-pressed}` — #065647): Pressed/active state of the primary button.
- **Action Accent** (`{colors.action-accent}` — #0CAE8E light / #6FD5BB dark): For active/running state indicators, progress fills, and inline highlights that need green but aren't CTAs. In dark mode uses Green/400 (#6FD5BB) for sufficient contrast.
- **Surface Brand** (`{colors.surface-brand}` — #E2F7F3 light / #201637 dark): Eyebrow badge background, active nav item background, footer background in dark mode, dark hero sections. In light mode it is the soft green tint; in dark mode it is the expressive brand-purple surface.

The only permitted **dark green text** on light backgrounds is `{colors.text-link}` / `{colors.text-green}` (#065647) — same primitive, two semantics: `{colors.text-link}` for hyperlinks only; `{colors.text-green}` for non-interactive green on `{colors.surface-brand}` or white (badges, eyebrows, active nav). Never `{colors.action-bg}` as text on white.

### Surface

The surface stack models physical depth. In dark mode, surfaces get lighter as they elevate above the page floor — this is opposite to how light is often used, but matches the physical metaphor of elevation.

| Token                      | Light   | Dark    | Role                                                                       |
| -------------------------- | ------- | ------- | -------------------------------------------------------------------------- |
| `{colors.surface-inset}`   | #F3F2F5 | #160F26 | Below the page floor — code blocks, text input backgrounds, recessed wells |
| `{colors.surface-page}`    | #FFFFFF | #242424 | The base canvas — all marketing and product pages                          |
| `{colors.surface-subtle}`  | #F3F2F5 | #2D273C | Elevated above page — cards, panels, table rows, dropdowns                 |
| `{colors.surface-brand}`   | #E2F7F3 | #201637 | Expressive brand sections — footer, dark hero, sidebar in product          |
| `{colors.surface-overlay}` | #E5E3EA | #242424 | Modal scrim backdrop; hover wash on interactive items                      |
| `{colors.surface-invert}`  | #201637 | #FFFFFF | Inverted high-contrast surfaces; text-on-dark design blocks                |

**`{colors.surface-page}` is the page floor — use it once per view, as the outermost background only.** Do not apply it to cards, panels, or nested containers. Inner elements that need a white fill should use `{colors.surface-subtle}` or define an explicit component token.

**Dark mode key insight:** The base canvas (`{colors.surface-page}` dark = #242424, Grey/11) is a **neutral dark grey** — no purple. The brand identity enters through elevated surfaces (`{colors.surface-subtle}` #2D273C) and expressive sections (`{colors.surface-brand}` #201637). This separates Seqera's dark mode from generic SaaS dark themes without making the full page feel purple/lilac.

### Text

| Token                      | Light   | Dark    | Use                                                                          |
| -------------------------- | ------- | ------- | ---------------------------------------------------------------------------- |
| `{colors.text-default}`    | #201637 | #FFFFFF | All body copy, paragraphs, headlines, UI labels                              |
| `{colors.text-muted}`      | #736F7D | #B9B7BE | Metadata, timestamps, placeholder text, secondary UI chrome                  |
| `{colors.text-inverse}`    | #FFFFFF | #201637 | Text on dark surfaces (dark buttons, dark cards, inverted blocks)            |
| `{colors.text-emphasized}` | #065647 | #6FD5BB | Emphasized callouts, featured stats, prominent non-link green text           |
| `{colors.text-link}`       | #065647 | #31C9AC | Hyperlinks only — always paired with `href`                                  |
| `{colors.text-green}`      | #065647 | #31C9AC | Green non-link UI — badges, eyebrows, active nav on `{colors.surface-brand}` |

**Body text is always `{colors.text-default}`.** `{colors.text-muted}` is for interface chrome only — timestamps, placeholders, helper text. Any editorial copy (feature descriptions, blog paragraphs) uses `{colors.text-default}`.

**`{colors.text-emphasized}` vs `{colors.text-green}`:** Both resolve to the same primitive in light mode (#065647) but serve different purposes. `text-green` is for UI elements with brand context (eyebrows on `surface-brand`). `text-emphasized` is for emphasized data, callout numbers, or featured paragraph text that needs green weight without being a link or badge.

### Border & Focus

| Token                     | Light   | Dark    | Use                                               |
| ------------------------- | ------- | ------- | ------------------------------------------------- |
| `{colors.border-default}` | #DDDEDE | #5C5767 | All card borders, input borders, dividers, topbar |
| `{colors.border-muted}`   | #EAEBEB | #453F51 | Subtle dividers, de-emphasized separators         |
| `{colors.border-focus}`   | #065647 | #31C9AC | Focus ring — 2px solid, 2px offset. Always green. |

`{colors.border-focus}` is the focus ring color. Always `2px solid {colors.border-focus}`, `outline-offset: 2px`. Never blue.

### Action / CTA

**Pairing rule:** `{colors.action-text}` (#201637, Brand Dark) **only pairs with `{colors.action-bg}`** (green). The dark brand text on green has sufficient contrast. When the background is `{colors.action-bg-emphasis}` (dark fill), the text must be `{colors.text-inverse}` (white) — NOT `{colors.action-text}`.

| Token                               | Light   | Dark    | Use                                                     |
| ----------------------------------- | ------- | ------- | ------------------------------------------------------- |
| `{colors.action-bg}`                | #31C9AC | #31C9AC | Primary CTA background — Green/500                      |
| `{colors.action-bg-hover}`          | #0CAE8E | #0CAE8E | Hover state for primary CTA                             |
| `{colors.action-bg-pressed}`        | #065647 | #065647 | Pressed state for primary CTA                           |
| `{colors.action-bg-emphasis}`       | #201637 | #2D273C | Secondary/emphasis action button background — dark fill |
| `{colors.action-bg-emphasis-hover}` | #160F26 | #201637 | Hover state for emphasis button                         |
| `{colors.action-text}`              | #201637 | #201637 | Text on `action-bg` (green) only                        |
| `{colors.action-border}`            | #31C9AC | #31C9AC | Outline/border for action elements                      |
| `{colors.action-accent}`            | #0CAE8E | #6FD5BB | Progress bars, active states, running indicators        |

```
button-primary:  bg=action-bg (#31C9AC) + text=action-text (#201637)     ✅
button-secondary: bg=action-bg-emphasis (#201637) + text=text-inverse (#FFFFFF)  ✅
button-secondary: bg=action-bg-emphasis (#201637) + text=action-text (#201637)   ❌ invisible
```

### Neutral

For UI elements that carry no brand identity — unbranded chips, placeholder graphics, neutral dividers.

| Token                       | Light   | Dark    | Use                                                 |
| --------------------------- | ------- | ------- | --------------------------------------------------- |
| `{colors.neutral-subtle}`   | #F3F2F5 | #545555 | Neutral surface: unbranded badges, chip backgrounds |
| `{colors.neutral-emphasis}` | #A8AAAB | #C4C6C7 | Neutral icons, decorative lines, placeholder fills  |

Unlike `{colors.surface-subtle}`, neutral tokens carry **no brand hue**. Use them when the element should be invisible in terms of brand identity (e.g., a grey "coming soon" chip, a placeholder illustration fill).

### Semantic States

Not brand colors — use exclusively for system feedback states.

**Error** (`Seqera/Decision Colors > Error` ← `Seqera/Primitives > Red`):

| Token                    | Light             | Dark               |
| ------------------------ | ----------------- | ------------------ |
| `{colors.error-surface}` | #FBE7E9 (Red/100) | #63181F (Red/1000) |
| `{colors.error-border}`  | #EE9AA2 (Red/300) | #C7303E (Red/700)  |
| `{colors.error-default}` | #DC3545 (Red/600) | #DC3545            |
| `{colors.error-text}`    | #AC2936 (Red/800) | #F5C2C7 (Red/200)  |

**Warning** (`Seqera/Decision Colors > Warning` ← `Seqera/Primitives > Yellow`):

| Token                      | Light   | Dark    |
| -------------------------- | ------- | ------- |
| `{colors.warning-surface}` | #FFF4E0 | #734A00 |
| `{colors.warning-border}`  | #FFD280 | #E79500 |
| `{colors.warning-default}` | #FFA500 | #FFA500 |
| `{colors.warning-text}`    | #A96D00 | #FFE4B3 |

**Success** (`Seqera/Decision Colors > Success` ← `Seqera/Primitives > Functional-Green`):

| Token                      | Light   | Dark    |
| -------------------------- | ------- | ------- |
| `{colors.success-surface}` | #E5F5EC | #124E2C |
| `{colors.success-border}`  | #94D7B0 | #249D58 |
| `{colors.success-default}` | #28AE61 | #28AE61 |
| `{colors.success-text}`    | #1F884C | #BFE7D0 |

### CSS Token → Variable Mapping

DESIGN.md token names map **directly** to CSS custom properties and Tailwind utility classes. No lookup table needed — the pattern is always `{colors.x-y}` → `--color-x-y` → `bg-x-y` / `text-x-y` / `border-x-y`.

Scale tokens (`bg-nextflow-500`, `text-brand-700`) remain valid for fine-grained palette work.

| Token                               | CSS variable                       | Tailwind class                | Hex (light) |
| ----------------------------------- | ---------------------------------- | ----------------------------- | ----------- |
| `{colors.action-bg}`                | `--color-action-bg`                | `bg-action-bg`                | `#31C9AC`   |
| `{colors.action-bg-hover}`          | `--color-action-bg-hover`          | `bg-action-bg-hover`          | `#0CAE8E`   |
| `{colors.action-bg-pressed}`        | `--color-action-bg-pressed`        | `bg-action-bg-pressed`        | `#065647`   |
| `{colors.action-bg-emphasis}`       | `--color-action-bg-emphasis`       | `bg-action-bg-emphasis`       | `#201637`   |
| `{colors.action-bg-emphasis-hover}` | `--color-action-bg-emphasis-hover` | `bg-action-bg-emphasis-hover` | `#160F26`   |
| `{colors.action-text}`              | `--color-action-text`              | `text-action-text`            | `#201637`   |
| `{colors.action-accent}`            | `--color-action-accent`            | `bg-action-accent`            | `#0CAE8E`   |
| `{colors.surface-page}`             | `--color-surface-page`             | `bg-surface-page`             | `#FFFFFF`   |
| `{colors.surface-subtle}`           | `--color-surface-subtle`           | `bg-surface-subtle`           | `#F3F2F5`   |
| `{colors.surface-inset}`            | `--color-surface-inset`            | `bg-surface-inset`            | `#F3F2F5`   |
| `{colors.surface-brand}`            | `--color-surface-brand`            | `bg-surface-brand`            | `#E2F7F3`   |
| `{colors.surface-overlay}`          | `--color-surface-overlay`          | `bg-surface-overlay`          | `#E5E3EA`   |
| `{colors.surface-invert}`           | `--color-surface-invert`           | `bg-surface-invert`           | `#201637`   |
| `{colors.text-default}`             | `--color-text-default`             | `text-text-default`           | `#201637`   |
| `{colors.text-muted}`               | `--color-text-muted`               | `text-text-muted`             | `#736F7D`   |
| `{colors.text-inverse}`             | `--color-text-inverse`             | `text-text-inverse`           | `#FFFFFF`   |
| `{colors.text-emphasized}`          | `--color-text-emphasized`          | `text-text-emphasized`        | `#065647`   |
| `{colors.text-link}`                | `--color-text-link`                | `text-text-link`              | `#065647`   |
| `{colors.text-green}`               | `--color-text-green`               | `text-text-green`             | `#065647`   |
| `{colors.border-default}`           | `--color-border-default`           | `border-border-default`       | `#DDDEDE`   |
| `{colors.border-muted}`             | `--color-border-muted`             | `border-border-muted`         | `#EAEBEB`   |
| `{colors.border-focus}`             | `--color-border-focus`             | `border-border-focus`         | `#065647`   |
| `{colors.neutral-subtle}`           | `--color-neutral-subtle`           | `bg-neutral-subtle`           | `#F3F2F5`   |
| `{colors.neutral-emphasis}`         | `--color-neutral-emphasis`         | `bg-neutral-emphasis`         | `#A8AAAB`   |
| `{colors.disabled-bg}`              | `--color-disabled-bg`              | `bg-disabled-bg`              | `#F3F2F5`   |
| `{colors.disabled-text}`            | `--color-disabled-text`            | `text-disabled-text`          | `#DDDEDE`   |
| `{colors.error-surface}`            | `--color-error-surface`            | `bg-error-surface`            | `#FBE7E9`   |
| `{colors.error-default}`            | `--color-error-default`            | `text-error-default`          | `#DC3545`   |
| `{colors.error-text}`               | `--color-error-text`               | `text-error-text`             | `#AC2936`   |
| `{colors.warning-default}`          | `--color-warning-default`          | `text-warning-default`        | `#FFA500`   |
| `{colors.success-default}`          | `--color-success-default`          | `text-success-default`        | `#28AE61`   |

Figma variables live in the **Seqera/Decision Colors** collection in the UI Playground file (`MBj38MxoDPSXJ4tzgicfy4`). Figma uses Title Case group prefixes: `Action/bg`, `Surface/page`, `Text/default`, `Border/default`.

---

## Brand Tokens

Brand Tokens are a simplified palette for brand designers, marketers, and presentation creators. They are **not used in UI code**. They live in the `Seqera/Brand Tokens` Figma collection.

### Brand Colors (Figma: Brand Tokens > Brand Colors)

The Seqera brand identity palette. Used in marketing materials, slide decks, sales collateral, and brand illustrations.

| Token              | Hex     | Primitive  | Use                                            |
| ------------------ | ------- | ---------- | ---------------------------------------------- |
| Green (brand hero) | #31C9AC | Green/500  | The primary brand color — logo, brand graphics |
| Dark green         | #065647 | Green/1000 | Dark green text in brand materials             |
| Dark brand         | #201637 | Brand/dark | Primary dark brand background                  |
| Deep brand         | #160F26 | Brand/1100 | Deeper dark brand for contrast                 |

Dark brand (#201637) belongs here in Brand Colors, not in Neutrals. It carries the brand hue and is used for the primary brand expression (dark hero sections, footer, brand blocks in marketing). It is the expressive surface of Seqera — not a neutral.

### Neutrals (Figma: Brand Tokens > Neutrals)

| Token        | Hex     | Primitive | Use                                                                          |
| ------------ | ------- | --------- | ---------------------------------------------------------------------------- |
| Light grey   | #F3F2F5 | Brand/200 | Off-white surfaces in brand materials                                        |
| Mid grey     | #736F7D | Brand/700 | Muted text, secondary information                                            |
| Dark neutral | #242424 | Grey/11   | Neutral dark for dark-mode brand backgrounds where no purple tint is desired |

Grey/11 (#242424) is a neutral dark with no brand hue. It is the correct choice when you want dark without purple — for example, a dark section that should not read as "Seqera brand purple". In brand materials, use it for backgrounds that need to recede without asserting the brand identity.

---

## Typography

### The Principle

Seqera's typographic personality comes from **scale jumps**, not graduation. A headline next to supporting text should create visual tension — that gap is the voice. Do not use intermediate sizes to smooth it out.

### Font Families

- **Display/Headlines** (`{typography.display}` → `{typography.h3}`): `Degular` — weight 600 only. Closest open-source alternative: [Bricolage Grotesque](https://fonts.google.com/specimen/Bricolage+Grotesque). Use this when Degular is unavailable in the environment (e.g. Figma) before falling back to Inter Bold.
- **Body/UI** (`{typography.h4}` → `{typography.small}`): `Inter` — weights 400, 500, 600 only. Never 300 (Light) in product UI.
- **Code/Mono** (`{typography.mono}`): `JetBrains Mono` — always, for all code, CLI strings, and technical rendering. Never fall back to system monospace.

### Scale

| Token                     | Size | Weight | Line Height | Letter Spacing | Use                                   |
| ------------------------- | ---- | ------ | ----------- | -------------- | ------------------------------------- |
| `{typography.display}`    | 68px | 600    | 1.0         | -0.025em       | Once per page. The primary statement. |
| `{typography.hero}`       | 56px | 600    | 1.04        | -0.025em       | Section heroes, slide headlines       |
| `{typography.h1}`         | 40px | 600    | 1.1         | -0.025em       | Page titles                           |
| `{typography.h2}`         | 32px | 600    | 1.15        | -0.025em       | Major section headings                |
| `{typography.h3}`         | 28px | 600    | 1.2         | -0.025em       | Sub-section headings                  |
| `{typography.h4}`         | 24px | 600    | 1.3         | 0              | Card titles, module headings (Inter)  |
| `{typography.h5}`         | 18px | 600    | 1.4         | 0              | Small headings (Inter)                |
| `{typography.h6}`         | 16px | 600    | 1.4         | 0              | Smallest heading level (Inter)        |
| `{typography.intro}`      | 20px | 400    | 1.5         | 0              | Hero subheadlines, section leads      |
| `{typography.body}`       | 16px | 400    | 1.5         | 0              | All body text                         |
| `{typography.small}`      | 14px | 400    | 1.5         | 0              | Supporting text, captions, timestamps |
| `{typography.blockquote}` | 20px | 400    | 1.5         | 0              | Pull quotes                           |
| `{typography.mono}`       | 14px | 400    | 1.6         | 0              | Code, CLI, technical strings          |

Tailwind semantic classes map directly: `typo-display`, `typo-hero`, `typo-h1` → `typo-h6`, `typo-intro`, `typo-body`, `typo-small`. Always use these classes — never compose custom font sizes.

### The 5 Tension Rules

1. **Jump aggressively.** `{typography.hero}` (56px) pairs with `{typography.small}` (14px) for supporting text — not `{typography.body}`. The gap is intentional.
2. **Numbers are typography.** Large metrics (94.2%, 10,000 runs) use `{typography.h1}` or larger in `{colors.text-default}`. Emphasis comes from scale, not color — never use teal for metric values.
3. **Eyebrow text** (small label above a headline): `{typography.small}`, uppercase, `letter-spacing: 1px`. Background: `{colors.surface-brand}`, text: `{colors.text-green}`. Never teal text, never gray, never a ghost variant.
4. **Maximum body text column width: 680px** (`{spacing.body-text-max}`). Wider lines break legibility and make text read as an unstructured block.
5. **Negative letter-spacing on all display text.** Headlines at `{typography.h3}` and above use `letter-spacing: -0.025em`. This is the single typographic detail that separates precision from genericness.

---

## Layout & Composition

### The Seqera Layout Logic

**Content is the visual.** A well-designed table or a product screenshot is worth more than any decorative illustration. Do not add visual elements to "fill" — add them to inform.

**One dominant element per section.** Every scroll block has something noticeably larger than everything else. That element is the argument of that section. Everything else supports it.

**Deliberate asymmetry.** In heroes and feature sections, the visual takes 55–60% of the width, the text 40–45%. Perfect centering is the most generic layout that exists.

### Spacing

Base unit: **4px**. All spacing is a multiple of 4. Use Tailwind's numeric scale (`p-6` = 24px, `py-20` = 80px) — never arbitrary values.

| Context                            | Token                      | Value                      |
| ---------------------------------- | -------------------------- | -------------------------- |
| Component internal padding         | `{spacing.lg}`             | 24px                       |
| Card padding                       | `{spacing.lg}`             | 24px                       |
| Card padding (compact)             | `{spacing.md}`             | 16px                       |
| Gap between components             | `{spacing.lg}`             | 24px                       |
| Section vertical padding (desktop) | `{spacing.section}`        | 80px                       |
| Section vertical padding (mobile)  | `{spacing.section-mobile}` | 48px                       |
| Container horizontal padding       | —                          | 24px mobile / 48px desktop |
| Max content width                  | `{spacing.container}`      | 1280px                     |
| Max body text column width         | `{spacing.body-text-max}`  | 680px                      |

### Approved Layout Patterns

**Marketing hero (1440px)**

- Left column (45%): `{component.eyebrow-badge}` + headline in `{typography.hero}` + subheadline in `{typography.intro}` + `{component.button-primary}`
- Right column (55%): product screenshot (`{component.product-screenshot}`), data-driven visualization, or code block (`{component.code-block}`)
- Light mode background: `{colors.surface-page}` (#FFFFFF). `{colors.surface-invert}` is used in the **footer only** in light-mode web pages.
- Dark mode background: `{colors.surface-page}` dark (#242424) — the full page surface.
- Dark-mode texture (marketing only): SVG block pattern (`opacity: 0.3–0.4`) + SVG noise layer (`mix-blend-mode: overlay, opacity: 0.15`) stacked over `{colors.surface-page}` dark
- Dark-mode animated grid: fine `1px` grid lines that draw in on load (`drawLines` keyframe, stroke `rgba(255,255,255,0.06)`) — dark mode only

**Feature section**

- Text left, product right — or inverted, alternating between features
- Not three equal columns with icon + title + text
- Vertical column borders: `border-x border-black/15` on sections using full container width
- Corner bracket decoration: L-shaped corner marks in `{colors.border-default}` at corners of featured content areas

**Product card**

- Border: `1px solid {colors.border-default}` (light) / `1px solid {colors.border-dark}` (dark)
- `{rounded.none}` — no rounding on cards or layout containers
- Padding: `{spacing.lg}` (24px)

**Product screenshot treatment**

- `{rounded.screenshot}` (16px), `border: 1px solid {colors.border-default}`, `box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25)`
- Background gradient behind screenshot: `from-gray-200/80 to-gray-100/80` (light) or matching dark surface tone in dark mode

---

## Elevation & Depth

Seqera uses elevation levels communicated through surface color change and borders — not through stacked shadow tiers.

| Level                                  | Treatment                                                                                   | Use                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **Inset**                              | `{colors.surface-inset}` + border                                                           | Recessed surfaces: code blocks, text inputs, data wells — sit below the page floor |
| **Flat**                               | No shadow. Border only: `1px solid {colors.border-default}`                                 | Cards, topbar, sidebar, panels, footer — 90% of surfaces                           |
| **Lifted**                             | `box-shadow: 0 4px 16px rgba(0,0,0,0.12)` + border                                          | Modals, dropdowns, tooltips — elements floating above the page surface             |
| **Hero**                               | `box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25)` + `{rounded.screenshot}` + border          | Product screenshots resting on a page                                              |
| **Marketing texture** (dark mode only) | SVG block pattern `opacity: 0.3–0.4` + noise layer `mix-blend-mode: overlay, opacity: 0.15` | Hero sections in dark-mode marketing pages. Never in product UI.                   |

**Shadow philosophy.** The prominent `box-shadow` on `{component.product-screenshot}` is used on product imagery — not on cards, not on buttons, not on nav elements. In product UI, elevation comes from surface color change and borders, not shadows.

---

## Shapes

The `rounded:` scale defines every corner radius in the system. Shape is one of the most recognizable Seqera identifiers — most surfaces are deliberately square.

| Token                  | Tailwind class       | Value  | Use                                                                                |
| ---------------------- | -------------------- | ------ | ---------------------------------------------------------------------------------- |
| `{rounded.none}`       | `rounded-none`       | 0px    | Cards, panels, containers, modals, navigation, footer — all layout surfaces        |
| `{rounded.xs}`         | `rounded-xs`         | 4px    | Buttons, form inputs, code blocks — interactive and inline elements                |
| `{rounded.md}`         | `rounded-md`         | 8px    | Reserved — not currently assigned to any active pattern                            |
| `{rounded.screenshot}` | `rounded-screenshot` | 16px   | Product screenshots only                                                           |
| `{rounded.badge}`      | `rounded-badge`      | 20px   | Eyebrow badge pill — the one non-interactive element with deliberate full rounding |
| `{rounded.full}`       | `rounded-full`       | 9999px | True pill — available in the scale but not used for buttons                        |

**The rule is structural.** If it is a layout surface or container, radius is `{rounded.none}`. If it is an interactive element (button, input), it earns `{rounded.xs}`. Screenshots are the single exception.

**Never use `{rounded.md}` (8px) for buttons.** The common SaaS default is 6–8px — Seqera uses 4px intentionally. The difference reads as precision.

---

## Components

### Buttons

**`{component.button-primary}`** — Seqera Green fill (`{colors.action-bg}`), Brand Dark text (`{colors.action-text}`, `{typography.small}` 14px / Medium 500), 4px radius, `6px 18px` padding, `1px solid {colors.action-bg-hover}` border. The primary CTA across all pages. One per section — never two primaries in the same viewport.

- Hover: background → `{colors.action-bg-hover}`, border → `{colors.action-bg-pressed}`.
- Small variant (`{component.button-small}`): same surface, `4px 12px` padding.

**`{component.button-secondary}`** — Emphasis dark fill (`{colors.action-bg-emphasis}` #201637 light / #2D273C dark), white text (`{colors.text-inverse}`, `{typography.small}` 14px / Medium 500), 4px radius, `6px 18px` padding, `1px solid {colors.border-default}` border. Used for secondary actions, ghost CTAs, and cancel/dismiss patterns.

- Hover: background → `{colors.action-bg-emphasis-hover}`.
- Text is always `{colors.text-inverse}` (white) — NOT `{colors.action-text}`. Dark fill requires light text.

**Never:**

- Pill-shaped buttons
- Two primary buttons in the same section
- White text on the primary (green) button — Brand Dark is required

### Cards & Panels

**`{component.card}`** — Pure white (`{colors.surface-page}`) surface with `1px solid {colors.border-default}`, `{rounded.none}` (0px), and `{spacing.lg}` (24px) padding. The border IS the boundary — no `box-shadow`.

**`{component.card-dark}`** — Brand dark surface (`{colors.surface-brand}` dark = #201637) with `1px solid {colors.border-dark}`. Used inside dark-mode pages and as the sidebar/panel surface in the product.

Cards never have `border-radius`. This is the primary visual distinction between Seqera and generic SaaS.

### Form Inputs

**`{component.input}`** — `{colors.surface-page}` background, `1px solid {colors.border-default}`, `{rounded.xs}` (4px), `8px 16px` padding, `{typography.body}`. Label above in `{typography.small}` `{colors.text-muted}`.

**`{component.input-focus}`** — Border and outline switch to `2px solid {colors.border-focus}` with `2px` offset.

**`{component.input-dark}`** — `{colors.surface-subtle}` dark value (#2D273C) background, `{colors.border-dark}` border. Same shape.

### Eyebrow Badge

**`{component.eyebrow-badge}`** — `{colors.surface-brand}` background, `{colors.text-green}` text, `{typography.small}` uppercase with `1px` letter-spacing, `{rounded.badge}` (20px), `3px 8px` padding.

Never: teal text, gray background, ghost/outline variant, inline within body text.

### Navigation

**`{component.topbar}`** — `{colors.surface-page}`, `1px solid {colors.border-default}` bottom border, 64px height. Light mode.

**`{component.topbar-dark}`** — `{colors.surface-brand}` dark value (#201637), `1px solid {colors.border-dark}`, 64px height.

**`{component.sidebar}`** — `{colors.surface-page}`, `1px solid {colors.border-default}` right border, 220px width. Active nav item: `{colors.surface-brand}` bg with `{colors.text-green}` text.

**`{component.sidebar-dark}`** — `{colors.surface-brand}` dark value (#201637), `1px solid {colors.border-dark}`, 220px width. Active nav item: `{colors.surface-subtle}` dark (#2D273C) background.

### Modal

**`{component.modal}`** — `{colors.surface-page}`, `1px solid {colors.border-default}`, `{rounded.none}` (0px), 640px max-width, `{spacing.lg}` padding. Lifted elevation: `box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25)`. Scrim: `rgba(0,0,0,0.5)`.

**`{component.modal-wide}`** — Same surface, 960px max-width.

No `border-radius` on modals. The structural language applies even to overlaid elements.

### Product Screenshot

**`{component.product-screenshot}`** — `{rounded.screenshot}` (16px), `1px solid {colors.border-default}`, `box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25)`.

Background gradient: `from-gray-200/80 to-gray-100/80` in light mode.

### Code Block

**`{component.code-block}`** — `{colors.surface-inset}` background (light: #F3F2F5), `1px solid {colors.border-default}`, `{rounded.xs}` (4px), `{spacing.lg}` padding, `{typography.mono}`.

**`{component.code-block-dark}`** — `{colors.surface-inset}` dark value (#160F26), `1px solid {colors.border-dark}`. Code blocks are the deepest surface in dark mode — recessed below the page floor.

---

## Do's and Don'ts

### Do

- Use `{colors.action-bg}` (#31C9AC) only for the primary CTA, active/running states, and brand confirmation moments — 2–3 times per screen maximum.
- Set all headlines at `{typography.h3}` and above with `letter-spacing: -0.025em`. This is non-negotiable.
- Use `{colors.text-link}` for hyperlinks; `{colors.text-green}` for green non-link UI on light surfaces — never `{colors.action-bg}` as text on white.
- Apply `{rounded.none}` (0px) to all cards, panels, containers, and modals.
- Use `{component.eyebrow-badge}` with `{colors.surface-brand}` bg + `{colors.text-green}` text — always, no variants.
- Use `{colors.text-inverse}` (white) as the text color on `{colors.action-bg-emphasis}` (dark button fill) — NOT `{colors.action-text}`.
- In dark mode, use `{colors.surface-page}` (#242424, Grey/11) as the base canvas. Let brand identity enter through elevated `{colors.surface-subtle}` and `{colors.surface-brand}` sections.
- Use `{colors.border-focus}` for all focus rings (2px solid, 2px offset).
- Name new Decision tokens by role, not component — "bg-emphasis" not "secondary-button-bg".

### Don't

- Don't introduce a second brand accent — every primary action signal is `{colors.action-bg}`.
- Don't use `{colors.action-bg}` as large metric or stat color — size provides emphasis, not color.
- Don't add `border-radius` to cards, panels, layout containers, or modals.
- Don't center body text — only headlines, and only when ≤ 3 words.
- Don't use two primary buttons in the same section.
- Don't use `rgba(color, 0.X)` to invent lighter tones — choose an existing palette step instead.
- Don't use Inter Light (weight 300) in product UI.
- Don't pair `{colors.action-text}` with `{colors.action-bg-emphasis}` — dark text on dark background. `action-text` only pairs with the green CTA.
- Don't use the old `surface-dark` / `surface-elevated-dark` / `surface-page-dark` tokens in new code — they are deprecated aliases.
- Don't skip token layers — components reference Decision tokens, not Primitives.

---

## What Seqera is NOT

Prohibitions are more useful than prescriptions. Every rule exists because the opposite is what AI tools default to.

### Aesthetics

- ❌ No gradient text (pink→purple, teal→blue, any gradient applied to letterforms)
- ❌ No organic blob shapes or amorphous forms in illustrations
- ❌ No isometric 3D style, no flat-design characters
- ❌ No "neon glow" — glows are subtle, always in teal only, never arbitrary colors
- ❌ No purple, indigo, or violet as a primary or accent color
- ❌ No `border-radius` on cards, panels, layouts, or containers — `{rounded.none}` (0px). Rounding is only for interactive elements (buttons, inputs).
- ❌ No `box-shadow` as the primary card boundary — cards use borders (`{colors.border-default}`), not shadows
- ❌ No three-column emoji-icon feature rows
- ❌ No `{colors.action-bg}` (#31C9AC) on large stat or metric numbers — numbers get emphasis from scale, not color. Use `{colors.text-default}`.

### Layout

- ❌ No perfectly centered hero with two equal-weight CTAs below
- ❌ No centered body text — only headlines, and only when ≤ 3 words
- ❌ No decorative elements that carry no information

### Components

- ❌ Never two primary buttons in the same section
- ❌ Never more than two font weights in a single component
- ❌ No Inter Light (`fontWeight: 300`) in product UI
- ❌ No pill-shaped buttons — Seqera uses `{rounded.xs}` (4px), not 8px, not 24px
- ❌ No eyebrow labels in teal — eyebrows use `{colors.surface-brand}` bg with `{colors.text-green}` text
- ❌ No `{colors.action-bg}` (#31C9AC) as text on white or light backgrounds — contrast fails WCAG AA
- ❌ No `{colors.action-text}` (#201637) on `{colors.action-bg-emphasis}` (dark button) — dark on dark, unreadable
- ❌ No invented opacity variants — do not write `rgba(color, 0.5)` to invent a lighter tone. Choose an existing palette color. **Exception:** CSS `box-shadow`, SVG `stroke`, and dark-mode texture layers may use `rgba` — these are CSS rendering effects, not fill colors.
- ❌ No deprecated v1 tokens (`surface-dark`, `surface-elevated-dark`, `surface-page-dark`) in new code

---

## Visual DNA

### From Linear — spatial precision and scale contrast

**What we apply:**

- Dramatic typographic scale — very large headline paired with very small supporting text, no intermediate sizes to soften the jump
- Generous section padding (≥ `{spacing.section}` / 80px vertical). Whitespace is respect, not emptiness
- The product or data IS the visual hero — never a decorative illustration in place of the real thing
- Layouts deliberately break the grid: one element at 60% width, the other at 40%

### From Vercel — monochrome discipline and dark-mode craft

**What we apply:**

- Color punctuates, it does not fill — most of the page is neutral; teal appears 2–3 times per screen maximum
- Dark mode is designed from scratch, not an inverted light mode
- Code is a design element — `{typography.mono}` has its own space, padding, and background
- If an element does not serve the content, it is not there

### From Supabase — developer-first legibility

Supabase faces the same challenge as Seqera: a technical product with a green accent.

**What we apply:**

- The green accent appears exactly where the user needs to act or receive confirmation
- Tables and data are designed, not afterthoughts
- Product screenshots are the visual heroes, not abstract illustrations
- Sections state what the product does using real data, not marketing speak

### The intersection — what all three share

- Typography does the heavy lifting, not decoration
- Color is structural, never decorative
- Layouts have one clearly dominant element per section
- Real product (screenshot, code, data) is worth more than any illustration

---

## Dark Mode

Dark mode is not inverting light mode. It is a different surface hierarchy built from the same primitives, with the base canvas using a neutral dark and the brand identity entering through elevation.

### Surface Hierarchy

| Role                              | Light            | Dark                 | Token                      |
| --------------------------------- | ---------------- | -------------------- | -------------------------- |
| Page background                   | #FFFFFF          | #242424 (Grey/11)    | `{colors.surface-page}`    |
| Card / panel surface              | #F3F2F5 + border | #2D273C (Brand/1000) | `{colors.surface-subtle}`  |
| Code block / input well           | #F3F2F5          | #160F26 (Brand/1100) | `{colors.surface-inset}`   |
| Brand section (footer, dark hero) | #E2F7F3          | #201637 (Brand/dark) | `{colors.surface-brand}`   |
| Modal scrim / overlay wash        | #E5E3EA          | #242424 (Grey/11)    | `{colors.surface-overlay}` |
| Inverted block                    | #201637          | #FFFFFF              | `{colors.surface-invert}`  |

**Why Grey/11 (#242424) as the dark page floor:**

The previous dark page background was Brand/dark (#201637, H271° S43% L~10%), which at higher lightness reads as lilac — distinctly purple and saturated at the page level. Grey/11 (#242424) is a neutral dark with no hue, which removes the purple from the base canvas. The brand identity enters through elevated surfaces (`{colors.surface-subtle}` = Brand/1000, #2D273C) and expressive sections (`{colors.surface-brand}` = Brand/dark, #201637) — on cards, footers, hero sections, and nav — where the purple hue reads as intentional and distinctive. The page floor recedes; the surfaces assert.

### Text & Borders

| Role          | Light                             | Dark                                   |
| ------------- | --------------------------------- | -------------------------------------- |
| Default text  | `{colors.text-default}` #201637   | `{colors.text-default}` #FFFFFF        |
| Muted text    | `{colors.text-muted}` #736F7D     | `{colors.text-muted}` #B9B7BE          |
| Borders       | `{colors.border-default}` #DDDEDE | `{colors.border-default}` #5C5767      |
| Links         | `{colors.text-link}` #065647      | `{colors.text-link}` #31C9AC           |
| Active nav bg | `{colors.surface-brand}` #E2F7F3  | `{colors.surface-subtle}` dark #2D273C |

**`{colors.action-bg}` (#31C9AC) does not change between modes.** It is the constant — the brand signal that works on both surfaces without adjustment.

### Building Dark Mode Correctly

```
✅ Dark mode surface stack:
   surface-inset  #160F26  →  deepest (code blocks)
   surface-page   #242424  →  base canvas
   surface-subtle #2D273C  →  cards, panels (elevated)
   surface-brand  #201637  →  brand sections (footer, hero)

❌ Don't treat dark mode as inverted light mode
❌ Don't use pure black (#000000) as the page floor
❌ Don't use surface-brand (#201637) for the full page background — it reads as lilac at scale
```

---

## Illustrations

Seqera produces its own illustrations. The full system is pending definition. These directional rules apply now:

**What Seqera illustrations are:**

- Geometric, not organic — no blobs, no amorphous shapes
- Encoded — they represent something real about the product: pipeline flows, data structures, coverage graphs, DAGs
- Restricted palette: nextflow scale (100–1000), brand scale (100–1100), and gray scale only
- Depth through layered opacity — maximum 3 stacked layers. This is the one permitted use of opacity on geometry; it does not apply to UI fill colors.
- No color gradients — use opacity-layered solids instead

**What they are not:**

- No illustrations that could belong to any tech company
- No generic metaphors: no network of nodes, no rocket ships, no lightbulbs
- No colors outside the brand palette

**When no illustration is available:** use a real product screenshot or a well-formatted code block (`{component.product-screenshot}`, `{component.code-block}`). Always better than a generic illustration.

---

## Iconography

Seqera has its own custom icon set. Until it is available for import:

- Use [Lucide](https://lucide.dev) as the fallback outline icon set — not filled, not duotone. Do not mix icon libraries.
- Sizes: 16px, 20px, 24px — only these three
- Color: always inherit from surrounding text context (`currentColor`), or `{colors.action-accent}` for action/active icons
- No emoji as icons in product UI — ever

---

## Responsive Behavior

### Breakpoints

| Name        | Width  | Key Changes                                                                         |
| ----------- | ------ | ----------------------------------------------------------------------------------- |
| `xs`        | 340px  | Single column; min layout. Eyebrow badges hide if they push line.                   |
| `sm`        | 640px  | Mobile landscape. 2-column grids collapse to 1. Hero switches to single column.     |
| `md`        | 768px  | Tablet. Feature sections stack. Primary nav condenses.                              |
| `lg`        | 1024px | Desktop entry. Full 2-column heroes. Feature grid at 2-up.                          |
| `header-lg` | 1091px | Header layout switches to full navigation row (custom breakpoint).                  |
| `xl`        | 1280px | Full layout. Content locks at `{spacing.container}` (1280px). Feature grid at 3-up. |
| `xxl`       | 1536px | Extra wide — margins absorb the rest, content width stays at 1280px.                |

### Typography Scaling

| Token                  | Desktop | Tablet (md) | Mobile (sm) |
| ---------------------- | ------- | ----------- | ----------- |
| `{typography.display}` | 68px    | 48px        | 36px        |
| `{typography.hero}`    | 56px    | 40px        | 32px        |
| `{typography.h1}`      | 40px    | 32px        | 28px        |
| `{typography.h2}`      | 32px    | 28px        | 24px        |
| `{typography.intro}`   | 20px    | 18px        | 16px        |
| `{typography.body}`    | 16px    | 16px        | 16px        |

Body text (`{typography.body}`) never scales below 16px.

### Collapsing Strategy

- **Topbar navigation:** Full link row at `lg` and above → hamburger menu below `md`. Logo stays left; primary CTA stays visible at all breakpoints.
- **Hero layout:** 2-column asymmetric (45/55) at `lg` → single column, text above visual, below `md`. Right-column visual stacks below text — never hidden.
- **Feature sections:** 2-column alternating at `lg` → single column stacked at `md`. Alternating direction collapses to consistent text-above-image.
- **Card grids:** 3-up at `xl` → 2-up at `lg` → 1-up at `sm`.
- **Section padding:** `{spacing.section}` (80px) at desktop → `{spacing.section-mobile}` (48px) at mobile.
- **Container horizontal padding:** 48px at `lg` → 24px at `sm`.
- **Sidebar (product UI):** Full 220px sidebar at `lg` → collapses to drawer overlay below `md`.

### Touch Targets

- Minimum 44 × 44px for all interactive elements per WCAG 2.2 (success criterion 2.5.8).
- `{component.button-primary}` with default padding reaches ~32px height — add `min-height: 44px` on mobile breakpoints.

---

## Checklist

Before delivering any asset, screen, or component:

- [ ] Does teal `{colors.action-bg}` appear at most 2–3 times on screen?
- [ ] Are all headlines at `{typography.h3}` and above using `letter-spacing: -0.025em`?
- [ ] Is there zero gradient text anywhere?
- [ ] Is there zero purple, indigo, or violet?
- [ ] Does the layout have one clearly visually dominant element per section?
- [ ] Are headlines left-aligned (centered only if ≤ 3 words)?
- [ ] Are numbers / metrics in `{colors.text-default}` — not teal?
- [ ] Are all body paragraphs and editorial copy in `{colors.text-default}` — not muted gray?
- [ ] Do eyebrow labels use `{colors.surface-brand}` bg / `{colors.text-green}` text — not teal text?
- [ ] Do buttons use `{rounded.xs}` (4px) — not 8px, not pill?
- [ ] Do cards, panels, containers, and modals use `{rounded.none}` (0px)?
- [ ] Are card borders `{colors.border-default}` — no shadow boundaries?
- [ ] Are all color values from the token palette — no invented `rgba(color, 0.X)` fills?
- [ ] Is `{colors.border-focus}` the focus ring color (2px solid, 2px offset)?
- [ ] Does dark mode use `{colors.surface-page}` dark (#242424) as the page floor?
- [ ] Does dark mode use `{colors.surface-subtle}` dark (#2D273C) for elevated cards?
- [ ] Does dark mode use `{colors.surface-brand}` dark (#201637) for brand sections — not the full page?
- [ ] Does the secondary button use `{colors.text-inverse}` (white) — not `{colors.action-text}` (dark)?
- [ ] Are all components referencing Decision tokens — not Primitives directly?
- [ ] Are no deprecated v1 tokens used (`surface-dark`, `surface-elevated-dark`, `surface-page-dark`)?
- [ ] Are headlines using Degular or Bricolage Grotesque — not Inter for display sizes?
- [ ] Is the visual hero real product or real data — not a decorative illustration?
- [ ] Does the CTA button text describe what happens, not what to click? ("See how it runs" not "Learn more")
- [ ] Could this design belong only to Seqera, or could it belong to any SaaS?

The last question is the definitive test.

---

## Known Gaps

- **v1 → v2 migration:** Three deprecated surface tokens (`surface-dark` #201637, `surface-elevated-dark` #2D273C, `surface-page-dark` #160F26) are still referenced in existing codebase. Replace with: `surface-invert` or `surface-brand` dark value, `surface-subtle` dark value, `surface-inset` dark value respectively. CSS variables for the deprecated tokens should remain as aliases until migration is complete.
- **surface-subtle light-mode value change:** v2 changes `surface-subtle` from `#F7F7F7` (neutral grey) to `#F3F2F5` (Brand/200 — faint brand tint). This is a visual breaking change for any component using `bg-surface-subtle` in light mode. Audit before shipping.
- **action-bg-emphasis CSS variable:** New token — not yet in the CSS variable system or Tailwind config. Needs `--color-action-bg-emphasis` and `--color-action-bg-emphasis-hover` added, plus Tailwind utility classes.
- **neutral-subtle / neutral-emphasis CSS variables:** New tokens — not yet wired into CSS.
- **text-emphasized CSS variable:** New token — not yet wired into CSS.
- **surface-inset / surface-invert CSS variables:** New tokens — not yet wired into CSS.
- **Figma Decision Colors sync:** Several new tokens (surface-inset, surface-invert, action-bg-emphasis, action-accent, text-emphasized, neutral-subtle, neutral-emphasis) need to be added to the `Seqera/Decision Colors` collection in Figma file `MBj38MxoDPSXJ4tzgicfy4`.
- **border-default hex discrepancy:** `{colors.border-default}` is defined as Grey/300 (#DDDEDE), but the CSS variable `--color-border` in the live codebase has historically been #CFD0D1 (Grey/400). Verify in the repo before shipping.
- **text-muted Figma sync:** `Text/muted` variable in `Seqera/Decision Colors` should be confirmed as #736F7D (Brand/700) light / #B9B7BE dark.
- **Info / interaction blue:** The functional blue scale (#4256E7 at step 6) exists in Figma as a 10-step palette in `Seqera/Primitives`. `{colors.info-*}` tokens are commented out pending a scope decision on where blue belongs in the system.
- **Button height spec:** current button padding (`6px 18px` at 14px font) produces ~34px height, below the 44px touch target minimum. A `min-height: 44px` rule for mobile is needed.
- **Active nav item dark mode:** v2 defines `{colors.surface-subtle}` dark (#2D273C) as the active nav bg in dark mode (was `surface-elevated-dark`). Update sidebar and nav components accordingly.
- **Loading and skeleton states:** not documented. Skeleton screens for cards and tables are pending.
- **Illustration system:** directional rules are defined but the full system (component library, approved motifs) is pending.
- **Warning / success / error toast patterns:** tokens are defined but the complete toast/banner component is not documented.
- **Form validation full pattern:** error tokens defined, but the complete input + helper-text + icon combination on validation failure is not documented as a component.
- **`typo-h7` / label / caption utility:** exists in the codebase, should be renamed `typo-label` or `typo-caption` (pending code update).
