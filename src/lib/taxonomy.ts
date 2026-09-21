import { parse } from 'yaml';

export type TaxonomyItem = { id: string; name: string; description?: string };

const modules = import.meta.glob('../../taxonomy/*.yaml', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>;

export function taxonomyItems(key: string): TaxonomyItem[] {
  const source = Object.entries(modules).find(([path]) => path.endsWith(`/${key}.yaml`))?.[1];
  if (!source) return [];
  const data = parse(source) as Record<string, unknown>;
  const items = data[key];
  return Array.isArray(items)
    ? items.filter((item): item is TaxonomyItem => Boolean(item && typeof item === 'object' && 'id' in item && 'name' in item))
    : [];
}

export const categories = taxonomyItems('categories');
export const categoryNameById = new Map(categories.map((item) => [item.id, item.name]));
export const categoryName = (id: string) => categoryNameById.get(id) ?? id;
