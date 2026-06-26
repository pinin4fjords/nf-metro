/** Site base path, normalised to a trailing slash (respects `base` in astro.config). */
export const base = import.meta.env.BASE_URL.replace(/\/?$/, "/");

/** GitHub repository, as `owner/name` and full URL. */
export const REPO = "pinin4fjords/nf-metro";
export const GITHUB_URL = `https://github.com/${REPO}`;
