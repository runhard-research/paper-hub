type R2AudioObject = {
  body: ReadableStream;
  size: number;
  range?: { offset: number; length: number };
  httpEtag: string;
  httpMetadata?: { contentType?: string };
  writeHttpMetadata(headers: Headers): void;
};

type AudioContext = {
  request: Request;
  params: { arxivId?: string };
  env: {
    PODCAST_EPISODES: {
      get(key: string, options?: { range?: Headers }): Promise<R2AudioObject | null>;
    };
  };
};

function audioKey(arxivId: string): string | null {
  // Only accept canonical arXiv IDs. This also prevents a request from using
  // path traversal to retrieve an arbitrary R2 object.
  const normalized = arxivId.replace(/v\d+$/, '');
  if (!/^\d{4}\.\d{4,5}$/.test(normalized)) return null;
  return `episodes/arXiv_${normalized}/audio/arXiv_${normalized}.mp3`;
}

async function serveAudio(context: AudioContext, includeBody: boolean): Promise<Response> {
  const key = context.params.arxivId ? audioKey(context.params.arxivId) : null;
  if (!key) return new Response('Invalid arXiv ID', { status: 400 });

  const rangeHeader = context.request.headers.get('Range');
  const object = await context.env.PODCAST_EPISODES.get(key, rangeHeader ? { range: context.request.headers } : undefined);
  if (!object) return new Response('Audio not found', { status: 404 });

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set('Content-Type', object.httpMetadata?.contentType ?? 'audio/mpeg');
  headers.set('Accept-Ranges', 'bytes');
  headers.set('ETag', object.httpEtag);
  headers.set('Cache-Control', 'private, max-age=3600');

  if (object.range) {
    const { offset, length } = object.range;
    headers.set('Content-Length', String(length));
    headers.set('Content-Range', `bytes ${offset}-${offset + length - 1}/${object.size}`);
    return new Response(includeBody ? object.body : null, { status: 206, headers });
  }

  headers.set('Content-Length', String(object.size));
  return new Response(includeBody ? object.body : null, { headers });
}

export const onRequestGet = (context: AudioContext) => serveAudio(context, true);
export const onRequestHead = (context: AudioContext) => serveAudio(context, false);
