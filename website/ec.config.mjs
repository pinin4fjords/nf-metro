// @ts-check
import { readFileSync } from "node:fs";
import {
  defineEcConfig,
  definePlugin,
  ExpressiveCodeAnnotation,
} from "@astrojs/starlight/expressive-code";
import { h } from "@expressive-code/core/hast";
import { pluginColorChips } from "expressive-code-color-chips";

// expressive-code-color-chips only annotates CSS-family languages, but our
// colourful content is the `#hex` values in ```metro / ```mmd line directives.
// This companion plugin runs the same idea for those languages and emits the
// official plugin's chip class ("ec-css-color-chip") + CSS var, so its baseStyles
// and the borderRadius override below style these chips too — no CSS duplication.
// (Upstream feature request: an option to choose which languages get chips.)
// The only coupling is that class name; if it ever changes, metro chips lose
// their styling until this string is updated to match.
const CHIP_CLASS = "ec-css-color-chip";
const CHIP_VAR = "--ec-css-color-chip";
const METRO_LANGS = new Set(["metro", "mmd"]);
// metro colours are always hex (#rgb / #rrggbb [/aa]); no rgb()/hsl() in the dialect.
const HEX_COLOR = /#[0-9a-fA-F]{3,8}\b/g;

class MetroColorChipAnnotation extends ExpressiveCodeAnnotation {
  constructor({ color, inlineRange }) {
    super({ inlineRange });
    this.color = color;
  }
  render({ nodesToTransform }) {
    return nodesToTransform.map((node) =>
      h(`span.${CHIP_CLASS}`, { style: `${CHIP_VAR}: ${this.color}` }, node),
    );
  }
}

function pluginMetroColorChips() {
  return definePlugin({
    name: "MetroColorChips",
    hooks: {
      postprocessAnalyzedCode({ codeBlock }) {
        if (!METRO_LANGS.has(codeBlock.language)) return;
        for (const line of codeBlock.getLines()) {
          for (const match of line.text.matchAll(HEX_COLOR)) {
            const columnStart = match.index;
            line.addAnnotation(
              new MetroColorChipAnnotation({
                color: match[0],
                inlineRange: {
                  columnStart,
                  columnEnd: columnStart + match[0].length,
                },
              }),
            );
          }
        }
      },
    },
    // The official plugin hardcodes `vertical-align: text-bottom`, which sits the
    // chip too low against the line text. Nudge all chips (CSS + metro) up to the
    // text midline. Loaded after pluginColorChips() so this wins.
    baseStyles() {
      return `.${CHIP_CLASS}::before { vertical-align: middle; margin-bottom: 2px; }`;
    },
  });
}

// Expressive Code config lives here (rather than inline in astro.config.mjs)
// because the <Code> component requires these options to be loadable on their
// own, and a plugin instance like pluginColorChips() is not JSON-serializable.

// Custom TextMate grammar so ```metro / ```mmd blocks highlight nf-metro's
// dialect (%%metro directives, graph/subgraph keywords, edges, node labels,
// hex colors). Real Mermaid lives in ```mermaid fences, rendered as diagrams
// by the astro-mermaid integration in astro.config.mjs.
const metroGrammar = JSON.parse(
  readFileSync(
    new URL("./src/grammars/metro.tmLanguage.json", import.meta.url),
    "utf8",
  ),
);

// Shiki has no bundled Lark grammar, so ```lark fences in the parser docs would
// render unhighlighted. This custom TextMate grammar covers Lark's rule/terminal
// definitions, priorities, regex/string literals, and %directives.
const larkGrammar = JSON.parse(
  readFileSync(
    new URL("./src/grammars/lark.tmLanguage.json", import.meta.url),
    "utf8",
  ),
);

export default defineEcConfig({
  shiki: { langs: [metroGrammar, larkGrammar] },
  // Render a color swatch next to hex/rgb values — handy for the `#hex` line
  // colors in %%metro directives. Square-ish chips (15%) rather than the
  // plugin's default circle (50%).
  plugins: [pluginColorChips(), pluginMetroColorChips()],
  styleOverrides: {
    colorChips: {
      borderRadius: "15%",
      size: "0.9em", // a little smaller than the default 1.2em
    },
  },
});
