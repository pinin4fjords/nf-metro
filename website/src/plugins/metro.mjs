/**
 * Astro integration + satteri mdast plugin that renders %%metro fenced code
 * blocks as inline SVG at build time.
 *
 * Any ```mermaid fenced block whose content contains at least one %%metro
 * directive is intercepted before astro-mermaid sees it, rendered via the
 * nf-metro CLI (Python), and replaced with a raw <svg> HTML node. Blocks
 * without %%metro directives pass through untouched to astro-mermaid.
 *
 * Rendered SVGs are cached by SHA-256 of the block content under
 * website/.metro-cache/ so re-builds and hot-reload only shell out on the
 * first encounter of each unique block. The cache is excluded from git.
 *
 * Requires nf-metro to be on PATH (activate the nf-core micromamba env first).
 * If nf-metro is absent and the cache is cold, the build fails with a clear
 * error message.
 */

import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { satteri, isSatteriProcessor } from '@astrojs/markdown-satteri';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = join(__dirname, '../../.metro-cache');

/** XML prolog that drawsvg emits; strip it before inlining into HTML. */
const XML_PROLOG_RE = /^<\?xml.*?>\s*/;

/**
 * Render a %%metro block, using the content-hash cache when available.
 * Returns the SVG string (prolog stripped).
 */
function renderMetro(content) {
  const hash = createHash('sha256').update(content).digest('hex').slice(0, 16);
  const cacheFile = join(CACHE_DIR, `${hash}.svg`);

  if (existsSync(cacheFile)) {
    return readFileSync(cacheFile, 'utf-8');
  }

  mkdirSync(CACHE_DIR, { recursive: true });

  const tmpInput = join(tmpdir(), `metro-${hash}.mmd`);
  const tmpOutput = join(tmpdir(), `metro-${hash}.svg`);

  writeFileSync(tmpInput, content, 'utf-8');

  try {
    execFileSync('nf-metro', ['render', tmpInput, '-o', tmpOutput], {
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  } catch (err) {
    const stderr = err.stderr?.toString().trim() || '';
    throw new Error(
      `nf-metro render failed for %%metro block:\n${stderr}\n\n` +
        `Make sure 'nf-metro' is on PATH (activate the nf-core micromamba env).`
    );
  }

  const svg = readFileSync(tmpOutput, 'utf-8').replace(XML_PROLOG_RE, '');
  writeFileSync(cacheFile, svg, 'utf-8');
  return svg;
}

/**
 * Satteri mdast plugin (Astro 7+): intercepts mermaid code nodes that contain
 * %%metro directives and replaces them with inline SVG HTML nodes.
 *
 * Must be registered BEFORE astro-mermaid's satteri plugin so that %%metro
 * blocks are claimed first and regular mermaid blocks pass through intact.
 */
export const satteriMetroPlugin = {
  name: 'nf-metro',
  code(node) {
    if (node.lang !== 'mermaid' || !node.value.includes('%%metro')) return;
    const svg = renderMetro(node.value);
    return { type: 'html', value: svg };
  },
};

/**
 * Legacy remark plugin fallback for Astro < 7 (unified processor).
 * Walks the MDAST and replaces matching code nodes in-place.
 */
function remarkMetroPlugin() {
  return function transformer(tree) {
    /** @type {{ node: any; index: number; parent: any }[]} */
    const targets = [];
    function walk(node) {
      if (!node.children) return;
      node.children.forEach((child, i) => {
        if (
          child.type === 'code' &&
          child.lang === 'mermaid' &&
          child.value.includes('%%metro')
        ) {
          targets.push({ node: child, index: i, parent: node });
        }
        walk(child);
      });
    }
    walk(tree);
    for (const { index, parent, node } of targets) {
      const svg = renderMetro(node.value);
      parent.children[index] = { type: 'html', value: svg };
    }
  };
}

/**
 * Astro integration.  Add this to `integrations` in astro.config.mjs,
 * BEFORE the astro-mermaid integration so the %%metro satteri plugin is
 * registered first in the mdastPlugins array.
 *
 * @returns {import('astro').AstroIntegration}
 */
export function metroPlugin() {
  return {
    name: 'nf-metro',
    hooks: {
      'astro:config:setup'({ config, updateConfig, logger }) {
        const existingProcessor = config.markdown?.processor;

        if (isSatteriProcessor(existingProcessor)) {
          const existingOptions = existingProcessor.options ?? {};
          updateConfig({
            markdown: {
              processor: satteri({
                ...existingOptions,
                mdastPlugins: [
                  ...(existingOptions.mdastPlugins ?? []),
                  satteriMetroPlugin,
                ],
              }),
            },
          });
          logger.info('nf-metro: registered satteri mdast plugin');
          return;
        }

        // Fallback for Astro < 7 (unified processor or no processor).
        updateConfig({
          markdown: {
            remarkPlugins: [
              ...(config.markdown?.remarkPlugins ?? []),
              remarkMetroPlugin,
            ],
          },
        });
        logger.info('nf-metro: registered remark plugin (legacy path)');
      },
    },
  };
}
