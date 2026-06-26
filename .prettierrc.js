// Prettier config. Plugins are resolved with require.resolve so the path is
// absolute: prettier v3 resolves bare plugin names from the working directory
// (the repo root, which has no node_modules), but the prek hook installs
// prettier-plugin-astro into its own isolated node env. require.resolve runs
// in that env and hands prettier a concrete path. See:
// https://github.com/prettier/prettier/issues/15388
module.exports = {
  plugins: [require.resolve("prettier-plugin-astro")],
  overrides: [
    {
      files: "*.astro",
      options: {
        parser: "astro",
      },
    },
  ],
};
