type R2ImageObject = {
  body: ReadableStream;
  size: number;
  httpEtag: string;
  httpMetadata?: { contentType?: string };
  writeHttpMetadata(headers: Headers): void;
};

type ImageContext = {
  request: Request;
  params: { arxivId?: string };
  env: {
    ARXIV_IMAGES: {
      get(key: string): Promise<R2ImageObject | null>;
    };
  };
};

function imageKey(arxivId: string): string | null {
  // Match only canonical arXiv IDs so a visitor cannot use this route to
  // request an arbitrary object from the bound bucket.
  const normalized = arxivId.replace(/v\d+$/, '');
  if (!/^\d{4}\.\d{4,5}$/.test(normalized)) return null;
  return `papers/${normalized}.png`;
}

async function serveImage(context: ImageContext, includeBody: boolean): Promise<Response> {
  const key = context.params.arxivId ? imageKey(context.params.arxivId) : null;
  if (!key) return new Response('Invalid arXiv ID', { status: 400 });

  const object = await context.env.ARXIV_IMAGES.get(key);
  if (!object) return new Response('Image not found', { status: 404 });

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set('Content-Type', object.httpMetadata?.contentType ?? 'image/png');
  headers.set('Content-Length', String(object.size));
  headers.set('ETag', object.httpEtag);
  headers.set('Cache-Control', 'private, max-age=86400');
  return new Response(includeBody ? object.body : null, { headers });
}

export const onRequestGet = (context: ImageContext) => serveImage(context, true);
export const onRequestHead = (context: ImageContext) => serveImage(context, false);
