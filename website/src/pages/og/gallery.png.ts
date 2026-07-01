import type { APIRoute } from "astro";
import { renderOgImage, pngResponse } from "../../lib/og-image.mjs";

export const GET: APIRoute = async () => {
  const png = await renderOgImage({
    kicker: "nf-metro",
    title: "Gallery",
    subtitle:
      "Layout patterns covering fan-outs, folds, diamonds, and realistic bioinformatics workflows.",
    mmdPath: "examples/rnaseq_sections.mmd",
  });
  return pngResponse(png);
};
