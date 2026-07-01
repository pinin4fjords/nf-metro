/**
 * Build-time Open Graph preview images: a branded 1200x630 card pairing a
 * pipeline/gallery map (rendered with baked colours via `renderMetroFile`,
 * since satori/resvg can't resolve the live `light-dark()`/`var()` chrome
 * `<Metro>` relies on) with the entry's title and description.
 *
 * satori lays out the card (title, description, embedded map image) as an
 * SVG using vector glyph paths - no system font or CSS text-rendering
 * dependency - which sharp then rasterizes to PNG, matching the SVG→PNG step
 * the CLI's own `--no-chrome-css` raster workflow already relies on.
 */

import satori from "satori";
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { renderMetroFile, REPO_ROOT } from "./render-metro.mjs";

const WIDTH = 1200;
const HEIGHT = 630;
const PANEL_WIDTH = 500;
const MAP_MAX_WIDTH = WIDTH - PANEL_WIDTH - 80;
const MAP_MAX_HEIGHT = HEIGHT - 80;

const BRAND_DARK = "#201637";
const MAP_STAGE = "#161022";
const ACCENT = "#56d3ba";
const INK = "#ffffff";
const INK_MUTE = "#c9c2da";

const FONT_DIR = join(
  REPO_ROOT,
  "website/node_modules/@fontsource/inter/files",
);
const fonts = [
  {
    name: "Inter",
    data: readFileSync(join(FONT_DIR, "inter-latin-400-normal.woff")),
    weight: 400,
    style: "normal",
  },
  {
    name: "Inter",
    data: readFileSync(join(FONT_DIR, "inter-latin-700-normal.woff")),
    weight: 700,
    style: "normal",
  },
];

/** Truncate to `max` chars on a word boundary, appending an ellipsis. */
function truncate(text, max) {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  return `${cut.slice(0, lastSpace > 0 ? lastSpace : max)}…`;
}

/** Smaller font for longer titles so it still fits the panel without clipping. */
function titleFontSize(title) {
  if (title.length > 26) return 34;
  if (title.length > 16) return 42;
  return 52;
}

/**
 * Render the target `.mmd` with baked colours and rasterize it to a PNG
 * buffer sized to fit within the card's map stage.
 * @param {string} mmdRelPath  Repo-relative path, e.g. "examples/rnaseq_auto.mmd".
 */
async function rasterizeMap(mmdRelPath) {
  const svg = renderMetroFile(join(REPO_ROOT, mmdRelPath), {
    chromeCss: false,
  });
  const { data, info } = await sharp(Buffer.from(svg), { density: 200 })
    .resize({
      width: MAP_MAX_WIDTH,
      height: MAP_MAX_HEIGHT,
      fit: "inside",
      withoutEnlargement: true,
    })
    .png()
    .toBuffer({ resolveWithObject: true });
  return {
    dataUri: `data:image/png;base64,${data.toString("base64")}`,
    width: info.width,
    height: info.height,
  };
}

/**
 * Render an OG preview card: a title/description panel beside the pipeline's
 * metro map.
 * @param {{ kicker: string, title: string, subtitle: string, mmdPath: string }} opts
 * @returns {Promise<Buffer>} PNG bytes.
 */
export async function renderOgImage({ kicker, title, subtitle, mmdPath }) {
  const map = await rasterizeMap(mmdPath);

  const tree = {
    type: "div",
    props: {
      style: {
        width: WIDTH,
        height: HEIGHT,
        display: "flex",
        fontFamily: "Inter",
        background: BRAND_DARK,
      },
      children: [
        {
          type: "div",
          props: {
            style: {
              width: PANEL_WIDTH,
              height: HEIGHT,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              padding: "56px 48px",
            },
            children: [
              {
                type: "div",
                props: {
                  style: { display: "flex", flexDirection: "column" },
                  children: [
                    {
                      type: "div",
                      props: {
                        style: {
                          fontSize: 20,
                          fontWeight: 700,
                          color: ACCENT,
                          textTransform: "uppercase",
                          letterSpacing: 3,
                        },
                        children: kicker,
                      },
                    },
                    {
                      type: "div",
                      props: {
                        style: {
                          fontSize: titleFontSize(title),
                          fontWeight: 700,
                          color: INK,
                          marginTop: 18,
                          lineHeight: 1.15,
                        },
                        children: title,
                      },
                    },
                    {
                      type: "div",
                      props: {
                        style: {
                          fontSize: 22,
                          fontWeight: 400,
                          color: INK_MUTE,
                          marginTop: 18,
                          lineHeight: 1.4,
                        },
                        children: truncate(subtitle, 150),
                      },
                    },
                  ],
                },
              },
              {
                type: "div",
                props: {
                  style: {
                    display: "flex",
                    alignItems: "baseline",
                    fontSize: 26,
                    fontWeight: 700,
                  },
                  children: [
                    { type: "span", props: { style: { color: ACCENT }, children: "nf-" } },
                    { type: "span", props: { style: { color: INK }, children: "metro" } },
                  ],
                },
              },
            ],
          },
        },
        {
          type: "div",
          props: {
            style: {
              width: WIDTH - PANEL_WIDTH,
              height: HEIGHT,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: MAP_STAGE,
            },
            children: {
              type: "img",
              props: { src: map.dataUri, width: map.width, height: map.height },
            },
          },
        },
      ],
    },
  };

  const svg = await satori(tree, { width: WIDTH, height: HEIGHT, fonts });
  return sharp(Buffer.from(svg)).png().toBuffer();
}

/**
 * Wrap a PNG buffer as an immutable `Response` for a static `.png.ts` route.
 * Every OG image lives at a URL scoped to one build (versioned deploys and PR
 * previews each get their own `base`), so its bytes never change post-deploy.
 * @param {Buffer} png
 */
export function pngResponse(png) {
  return new Response(png, {
    headers: {
      "Content-Type": "image/png",
      "Cache-Control": "public, max-age=31536000, immutable",
    },
  });
}
