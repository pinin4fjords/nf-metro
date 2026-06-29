import { defineCollection } from "astro:content";
import { docsLoader } from "@astrojs/starlight/loaders";
import { docsSchema } from "@astrojs/starlight/schema";
import { file } from "astro/loaders";
import { z } from "astro:content";

const gallerySchema = z.object({
  order: z.number(),
  title: z.string(),
  src: z.string(),
  description: z.string(),
  category: z.string(),
});

const pipelinesSchema = z.object({
  title: z.string(),
  src: z.string(),
  repo_url: z.string().url(),
  description: z.string(),
});

export const collections = {
  docs: defineCollection({ loader: docsLoader(), schema: docsSchema() }),
  gallery: defineCollection({
    loader: file("./src/content/gallery.json"),
    schema: gallerySchema,
  }),
  pipelines: defineCollection({
    loader: file("./src/content/pipelines.json"),
    schema: pipelinesSchema,
  }),
};
