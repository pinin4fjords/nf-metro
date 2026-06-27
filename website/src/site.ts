/** Site base path, normalised to a trailing slash (respects `base` in astro.config). */
export const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");

/**
 * GitHub Pages project root, constant across versioned deploys. `base` carries
 * the version segment (e.g. /nf-metro/latest/); SITE_BASE is the root the
 * version switcher strips and re-prefixes to build cross-version URLs.
 */
export const SITE_BASE = "/nf-metro/";

/** GitHub repository, as `owner/name` and full URL. */
export const REPO = "pinin4fjords/nf-metro";
export const GITHUB_URL = `https://github.com/${REPO}`;
