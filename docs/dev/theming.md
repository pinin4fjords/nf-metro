---
title: "Theming"
description: How nf-metro builds one SVG that adapts to a light or dark viewer, and how to use the same technique outside nf-metro.
sidebar:
  order: 9
---

An nf-metro SVG is a single file. It has no light version and no dark version,
and nf-metro does not re-render it when the viewer's theme changes. Instead,
every themeable color is baked into the file as a pair, and a CSS mechanism
built into SVG picks the right half of the pair at display time. This page
explains that mechanism in general terms, then shows exactly where nf-metro
uses it, so the technique is reusable in any SVG you generate yourself.

## The three ingredients

### 1. `light-dark()` - a CSS value that carries two colors

`light-dark()` is a CSS function that takes two values and resolves to one of
them:

```css
fill: light-dark(#ffffff, #1a1a1a);
```

In a light context this is `#ffffff`. In a dark context it's `#1a1a1a`. Both
values live in the same file; nothing is regenerated to switch between them.
It works on any CSS color property - `fill`, `stroke`, `stop-color` - so it
applies directly to SVG presentation attributes and `<style>` rules.

### 2. `color-scheme` - what tells `light-dark()` which half to pick

`light-dark()` does nothing on its own. It resolves against the nearest
`color-scheme` in effect, so something in the document has to declare one:

```css
:root {
  color-scheme: light dark;
}
```

`light dark` means "resolve to whichever the viewer prefers" - normally the
OS or browser dark-mode setting. You can also pin it to one side explicitly
(`color-scheme: dark`), which forces every `light-dark()` in that document to
resolve to its dark value regardless of viewer preference. That's a useful
escape hatch when you're baking a single concrete image (a PNG export, say)
and want a deterministic result instead of "whatever this renderer's own
environment happens to report."

### 3. `var(..., light-dark(...))` - an optional third layer for overrides

Wrapping the pair in a custom property gives a host page a way to override
the color entirely, while keeping the light/dark pair as the fallback:

```css
fill: var(--my-bg, light-dark(#ffffff, #1a1a1a));
```

If nothing sets `--my-bg`, the light/dark pair still works. If a host page
sets `--my-bg: pink` on a wrapping element, that wins and the light/dark
logic never runs. This third layer is optional; the first two are the whole
mechanism.

## How nf-metro applies it

nf-metro defines every theme as a light/dark pair up front - `nfcore.py` and
`seqera.py` in `src/nf_metro/themes/` each export a light and a dark `Theme`.
When it builds the SVG, `_inject_chrome_css` (`src/nf_metro/render/svg.py`)
recovers that pair with `mode_pair(theme)` and writes CSS like:

```css
.nf-metro-map-bg {
  fill: var(--nfm-bg, light-dark(#ffffff, #1a1a1a));
}
```

for the background, section fill, label color, and the other chrome colors
described in [Embedding](/nf-metro/embedding/). Route and line colors are
excluded on purpose: they carry meaning (which line is which), so they stay
as fixed presentation attributes rather than mode-adaptive values.

Then, on the `<svg>` root element itself, nf-metro sets the `color-scheme`
that activates all of this:

```html
<svg style="color-scheme: light dark"></svg>
```

That single style declaration is what makes every `light-dark()` inside the
file resolve to the viewer's preference. Without it, `light-dark()` values
are present in the CSS but inert - the spec requires a `color-scheme` to
resolve against, so nothing switches.

## Two things that break it, and why

**Referencing the SVG with `<img>` isolates it.** An `<img src="map.svg">`
loads the SVG into its own internal document, which does not inherit
`color-scheme` from the host page. The SVG only sees the browser or OS-level
preference, never a manual light/dark toggle the host page implements with
its own `color-scheme` or a `data-theme` attribute. That's why nf-metro's own
docs site inlines its maps (`<Metro>`, `ZoomableSVG.astro`) instead of
pointing an `<img>` at the file: inlining puts the SVG in the same document
as the page, so it picks up whatever `color-scheme` the page has set.
`<object>` and `<embed>` behave like `<img>` here too; only inline `<svg>`
(or, as nf-metro does, setting `color-scheme` directly on the SVG's own root)
is reliable regardless of embedding method.

**Not every consumer understands `light-dark()` or `var()`.** Both are
ordinary CSS, but some SVG consumers parse a restricted subset of it and
abort outright on values they don't recognize - `cairosvg` is one. If you
know the target is a consumer like that, resolve the color choice yourself
and bake one concrete value per property instead of relying on the two
ingredients above at export time. nf-metro's `--no-chrome-css` flag does
exactly this: it substitutes the resolved light or dark hex value in place of
every `var(..., light-dark(...))` expression, so the file stays valid for a
rasterizer that can parse neither construct. `--mode light`/`--mode dark` is
the companion piece: it pins `color-scheme` on the root to one side (the
escape hatch from ingredient 2), so even a rasterizer that _does_ understand
`light-dark()` produces a deterministic result instead of following whatever
`color-scheme` its own environment happens to report.

nf-metro also ships one fallback for a case the two ingredients cover badly:
a transparent-background theme has no `--nfm-bg` pair doing any work, so text
drawn straight onto the canvas (titles, section labels) can still turn
illegible against whatever background the host page happens to have. For
that case only, nf-metro additionally injects a `@media (prefers-color-scheme:
dark)` block as a second, independent adaptation path. It's a coarser
mechanism than `light-dark()` - a media query is all-or-nothing per rule, not
a per-value pair - kept around only because it's the one lever left once
there's no baked background color to attach a `light-dark()` pair to.
`--no-dark-mode-css` turns it off, for a host that already manages its own
theming and doesn't want the SVG's media query fighting it.

## Using this in your own SVG, outside nf-metro

The whole mechanism is three lines, none of them nf-metro-specific:

```html
<svg viewBox="0 0 200 100" style="color-scheme: light dark">
  <style>
    rect {
      fill: light-dark(#ffffff, #1a1a1a);
    }
    text {
      fill: light-dark(#111111, #eeeeee);
    }
  </style>
  <rect width="200" height="100" />
  <text x="10" y="50">Adapts to the viewer's theme</text>
</svg>
```

Inline that in an HTML page - not `<img src="...">` - and it follows the
page's `color-scheme`, which in turn follows the OS preference unless the
page overrides it, for example with a manual dark-mode toggle that sets
`document.documentElement.style.colorScheme = "dark"`. Wrap any value you
also want a host page to override directly in `var(--your-prop,
light-dark(...))`, the same way nf-metro's `--nfm-*` properties work.

If the SVG needs to work standalone too - opened directly as a file, with no
host page to inherit from - keep `color-scheme: light dark` on the SVG's own
root element rather than relying on a page's `:root` to supply it.
