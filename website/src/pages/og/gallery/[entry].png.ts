import type { APIRoute, GetStaticPaths } from "astro";
import { getCollection, type CollectionEntry } from "astro:content";
import { renderOgImage, pngResponse } from "../../../lib/og-image.mjs";

export const getStaticPaths = (async () => {
  const entries = await getCollection("gallery");
  return entries.map((entry) => ({
    params: { entry: entry.id },
    props: { entry },
  }));
}) satisfies GetStaticPaths;

export const GET: APIRoute<{ entry: CollectionEntry<"gallery"> }> = async ({
  props,
}) => {
  const { title, description, category, src } = props.entry.data;
  const png = await renderOgImage({
    kicker: category,
    title,
    subtitle: description,
    mmdPath: src,
  });
  return pngResponse(png);
};
