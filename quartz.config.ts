import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "RicePunchb's Garden",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "ko-KR",
    baseUrl: "ricepunchb.github.io/Idsh-obsidian-garden",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "IBM Plex Sans KR",
        body: {
          name: "Noto Sans KR",
          weights: [400, 700],
          includeItalic: true,
          },
        code: "JetBrains Mono",
      },
      colors: {
        lightMode: {
          light: "#FFFBF5",
          lightgray: "#F9F1E8",
          gray: "#8C7A6B",
          darkgray: "#5F4B3F",
          dark: "#2C1F18",
          secondary: "#D97706",
          tertiary: "#B45309",
          highlight: "rgba(217, 119, 6, 0.12)",
          textHighlight: "#D97706",
        },
        darkMode: {
          light: "#2B1F1A",
          lightgray: "#3A2C25",
          gray: "#8C7666",
          darkgray: "#D9C2B0",
          dark: "#F5E6D9",
          secondary: "#FFB84D",
          tertiary: "#FF9F1C",
          highlight: "rgba(255, 177, 77, 0.18)",
          textHighlight: "#FFB84D",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({
        renderEngine: "mathjax"
        }),    // katex
    ],
    filters: [Plugin.RemoveDrafts(), Plugin.ExplicitPublish()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      // Comment out CustomOgImages to speed up build time
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
