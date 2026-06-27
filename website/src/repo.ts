/**
 * Single source of truth for the GitHub owner/repo and the Pages origin.
 *
 * On a repository transfer, update the values here and everything else follows:
 * the canonical site URL, edit links, the header GitHub icon, and the version
 * switcher's release links all derive from these.
 */
export const REPO = "pinin4fjords/nf-metro";
export const GITHUB_URL = `https://github.com/${REPO}`;
export const PAGES_ORIGIN = "https://pinin4fjords.github.io";
