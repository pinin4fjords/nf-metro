// @ts-check
import {
  defineEcConfig,
  definePlugin,
} from "@astrojs/starlight/expressive-code";
import { pluginColorChips } from "expressive-code-color-chips";
// Custom TextMate grammars, imported as JSON so they resolve at bundle/load
// time. A runtime fs read keyed on import.meta.url breaks when this config is
// bundled for the <Code> component (the URL points at the emitted chunk, not
// the source), which silently drops every option from that render path.
//
// `metro` highlights nf-metro's dialect in ```metro / ```mmd fences and <Code>
// blocks (%%metro directives, graph/subgraph keywords, edges, labels, hex
// colors); real Mermaid lives in ```mermaid fences, rendered as diagrams by the
// astro-mermaid integration. `lark` covers the Lark grammar in the parser docs,
// which Shiki has no bundled language for.
import metroGrammar from "./src/grammars/metro.tmLanguage.json" with { type: "json" };
import larkGrammar from "./src/grammars/lark.tmLanguage.json" with { type: "json" };

// pluginColorChips's `languages` option REPLACES its built-in defaults rather
// than extending them, so the CSS dialects are repeated here to keep chips on
// CSS blocks; "metro"/"mmd" are the only additions. Keep the CSS entries in
// sync if the plugin ever adds a dialect (its default: css, scss, sass, less,
// stylus).
const COLOR_CHIP_LANGS = [
  "css",
  "scss",
  "sass",
  "less",
  "stylus",
  "metro",
  "mmd",
];

// The official plugin hardcodes `vertical-align: text-bottom`, which sits the
// chip too low against the line text. This plugin, loaded after
// pluginColorChips(), nudges all chips up to the text midline.
function pluginChipVerticalAlign() {
  return definePlugin({
    name: "ChipVerticalAlign",
    baseStyles() {
      return `.ec-css-color-chip::before { vertical-align: middle; margin-bottom: 2px; }`;
    },
  });
}

// Expressive Code config lives here (rather than inline in astro.config.mjs)
// because the <Code> component requires these options to be loadable on their
// own, and a plugin instance like pluginColorChips() is not JSON-serializable.

export default defineEcConfig({
  shiki: { langs: [metroGrammar, larkGrammar] },
  // Render a color swatch next to hex/rgb values - handy for the `#hex` line
  // colors in %%metro directives. Square-ish chips (15%) rather than the
  // plugin's default circle (50%).
  plugins: [
    pluginColorChips({ languages: COLOR_CHIP_LANGS }),
    pluginChipVerticalAlign(),
  ],
  styleOverrides: {
    colorChips: {
      borderRadius: "15%",
      size: "0.9em", // a little smaller than the default 1.2em
    },
  },
});
