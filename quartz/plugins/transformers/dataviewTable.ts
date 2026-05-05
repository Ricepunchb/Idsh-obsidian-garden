import fs from "fs"
import path from "path"
import matter from "gray-matter"
import { QuartzTransformerPlugin } from "../types"

interface Options {
  contentDir: string
}

const defaultOptions: Options = {
  contentDir: "content",
}

function escapeMd(text: unknown): string {
  return String(text ?? "").replace(/\|/g, "\\|").trim()
}

function parseDataviewBlock(block: string) {
  const normalized = block.replace(/\r\n/g, "\n").trim()

  const fromMatch = normalized.match(/FROM\s+"([^"]+)"/i)
  const sortMatch = normalized.match(/SORT\s+([^\s]+)\s+(ASC|DESC)/i)
  const tableMatch = normalized.match(/TABLE\s+([\s\S]*?)\s+FROM\s+"[^"]+"/i)

  if (!fromMatch || !tableMatch) return null

  const rawColumns = tableMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)

  const columns = rawColumns.map((expr) => {
    const m = expr.match(/^(.+?)\s+AS\s+"(.+?)"$/i)
    if (m) {
      return { field: m[1].trim(), label: m[2].trim() }
    }
    return { field: expr.trim(), label: expr.trim() }
  })

  return {
    from: fromMatch[1].trim(),
    sortField: sortMatch?.[1]?.trim(),
    sortOrder: (sortMatch?.[2]?.toUpperCase() ?? "ASC") as "ASC" | "DESC",
    columns,
  }
}

function walkMarkdownFiles(dir: string): string[] {
  if (!fs.existsSync(dir)) return []
  const entries = fs.readdirSync(dir, { withFileTypes: true })
  const files: string[] = []

  for (const entry of entries) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      files.push(...walkMarkdownFiles(full))
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      files.push(full)
    }
  }

  return files
}

function buildMarkdownTable(rows: Record<string, unknown>[], columns: { field: string; label: string }[]) {
  const header = `| ${columns.map((c) => escapeMd(c.label)).join(" | ")} |`
  const divider = `| ${columns.map(() => "---").join(" | ")} |`
  const body = rows.map((row) => {
    return `| ${columns.map((c) => escapeMd(row[c.field])).join(" | ")} |`
  })

  return [header, divider, ...body].join("\n")
}

export const DataviewTable: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }

  return {
    name: "DataviewTable",
    textTransform(_ctx, src) {
      const text = src.toString()

      return text.replace(/```dataview\s*([\s\S]*?)```/g, (_match, block) => {
        const parsed = parseDataviewBlock(block)
        if (!parsed || parsed.columns.length === 0) {
          return _match
        }

        const targetDir = path.join(process.cwd(), opts.contentDir, parsed.from)
        const files = walkMarkdownFiles(targetDir)

        let rows = files.map((file) => {
          const raw = fs.readFileSync(file, "utf8")
          const { data } = matter(raw)
          return data as Record<string, unknown>
        })

        if (parsed.sortField) {
          rows = rows.sort((a, b) => {
            const av = String(a[parsed.sortField!] ?? "")
            const bv = String(b[parsed.sortField!] ?? "")
            const cmp = av.localeCompare(bv, "ko")
            return parsed.sortOrder === "DESC" ? -cmp : cmp
          })
        }

        return buildMarkdownTable(rows, parsed.columns)
      })
    },
  }
}

export default DataviewTable