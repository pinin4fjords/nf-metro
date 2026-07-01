import type { APIRoute } from "astro";
import { renderOgImage, pngResponse } from "../../lib/og-image.mjs";

export const GET: APIRoute = async () => {
  const png = await renderOgImage({
    kicker: "nf-metro",
    title: "nf-core pipelines",
    subtitle:
      "Real-world nf-core pipelines rendered as metro-style SVG diagrams.",
    mmdPath: "examples/rnaseq_auto.mmd",
  });
  return pngResponse(png);
};
