import { base, PAGES_ORIGIN } from "../site";

/**
 * `head` frontmatter entries pointing `og:image`/`twitter:image` at a
 * build-time OG PNG. Passed to `StarlightPage`/docs frontmatter to override
 * the site-wide default set in astro.config.mjs for a specific page.
 * @param relPath  Path under the site base, e.g. "og/pipelines/rnaseq_auto.png".
 */
export function ogImageHead(relPath: string) {
  const url = `${PAGES_ORIGIN}${base}${relPath}`;
  return [
    { tag: "meta" as const, attrs: { property: "og:image", content: url } },
    { tag: "meta" as const, attrs: { name: "twitter:image", content: url } },
  ];
}
