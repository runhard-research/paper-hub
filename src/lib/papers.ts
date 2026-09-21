import { parse } from 'yaml';

export type Classification = {
  primary_category?: string;
  categories?: string[];
  tasks?: string[];
  technologies?: string[];
  applications?: string[];
  tags?: string[];
};

export type Paper = {
  slug: string;
  folderYear: string;
  title: string;
  authors: string[];
  abstract: string;
  summary: string;
  publishedAt?: string;
  arxivId?: string;
  arxivUrl?: string;
  pdfUrl?: string;
  classification: Classification;
  body: string;
};

const modules = import.meta.glob('../../papers/**/*.md', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>;

function section(body: string, heading: string): string {
  const match = body.match(new RegExp(`^##\\s+${heading}\\s*\\r?\\n`, 'mi'));
  if (!match || match.index === undefined) return '';
  const afterHeading = body.slice(match.index + match[0].length);
  const nextHeading = afterHeading.search(/^##\s/m);
  return (nextHeading >= 0 ? afterHeading.slice(0, nextHeading) : afterHeading).trim();
}

function field(body: string, label: string): string | undefined {
  const match = body.match(new RegExp(`^-\\s+\\*\\*${label}\\*\\*:\\s*(.+)$`, 'mi'));
  return match?.[1].trim() || undefined;
}

function frontMatter(raw: string): [Record<string, unknown>, string] {
  const match = raw.match(/^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n/);
  if (!match) return [{}, raw];
  const data = parse(match[1]);
  return [data && typeof data === 'object' ? data as Record<string, unknown> : {}, raw.slice(match[0].length)];
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function parsePaper(path: string, raw: string): Paper {
  const [meta, body] = frontMatter(raw);
  const classification = (meta.classification && typeof meta.classification === 'object' ? meta.classification : {}) as Classification;
  const title = typeof meta.title === 'string' ? meta.title : body.match(/^#\s+(.+)$/m)?.[1]?.trim() ?? path.split('/').at(-1)!.replace(/\.md$/, '');
  const arxivUrl = typeof meta.arxiv_url === 'string' ? meta.arxiv_url : field(body, 'arXiv');
  const arxivId = typeof meta.arxiv_id === 'string' ? meta.arxiv_id : arxivUrl?.match(/arxiv\.org\/abs\/([\w.]+)/)?.[1];
  const inlineTags = field(body, 'Tags')?.match(/#[\w-]+/g)?.map((tag) => tag.slice(1)) ?? [];
  const relative = path.replace(/^.*\/papers\//, '').replace(/\.md$/, '');
  const folderYear = relative.split('/')[0] ?? 'unknown';
  return {
    slug: relative,
    folderYear,
    title,
    authors: strings(meta.authors),
    abstract: typeof meta.abstract === 'string' ? meta.abstract : section(body, 'Notes'),
    summary: typeof meta.summary === 'string' ? meta.summary : section(body, 'One-liner'),
    publishedAt: typeof meta.published_at === 'string' ? meta.published_at : field(body, 'Added'),
    arxivId,
    arxivUrl,
    pdfUrl: typeof meta.pdf_url === 'string' ? meta.pdf_url : field(body, 'PDF'),
    classification: { ...classification, categories: strings(classification.categories), tasks: strings(classification.tasks), technologies: strings(classification.technologies), applications: strings(classification.applications), tags: strings(classification.tags).length ? strings(classification.tags) : inlineTags },
    body,
  };
}

export const papers = Object.entries(modules).map(([path, raw]) => parsePaper(path, raw));
export const recentPapers = [...papers].sort((a, b) => (b.publishedAt ?? '').localeCompare(a.publishedAt ?? ''));
export const years = [...new Set(papers.map((paper) => paper.folderYear))].sort().reverse();
export const paperBySlug = (slug: string) => papers.find((paper) => paper.slug === slug);
