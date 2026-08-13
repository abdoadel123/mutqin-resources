/**
 * مُجمِّع بيانات مُتقِن — Cloudflare Worker + R2.
 * يستقبل دفعات الأحداث وملفات صوت المساهمة من التطبيق ويخزّنها في R2
 * منظّمةً بالجهاز والتاريخ — تسحبها لاحقًا للمعالجة والتدريب.
 *
 * النشر (مرة واحدة):
 *  1. أنشئ حساب Cloudflare (مجاني) → R2 → أنشئ bucket باسم mutqin-data.
 *  2. Workers → Create Worker → الصق هذا الملف.
 *  3. Settings → Bindings → R2 bucket: الاسم DATA ← mutqin-data.
 *  4. Settings → Variables → Secret باسم KEY (مفتاح عشوائي طويل من عندك).
 *  5. انسخ رابط الـWorker وضعه مع المفتاح في أسرار GitHub للريبو:
 *     SYNC_URL = https://<worker>.workers.dev   و SYNC_TOKEN = نفس KEY.
 * والتدريب كله من هنا: ‏Colab يسحب الداتا (‏/data/*) ويرفع الموديل الجديد
 * (‏POST ‏/put) بمفتاح التطبيق نفسه — والتطبيقات تنزّل الموديل من ‏/model/*.
 */
export default {
  async fetch(req, env) {
    const url = new URL(req.url)

    // تنزيل الموديل: عام (لا مفتاح) — ملفات ثابتة بكاش طويل وCORS مفتوح
    if (req.method === 'GET' && url.pathname.startsWith('/model/')) {
      const key = 'model/' + url.pathname.slice(7).replace(/[^\w.-]/g, '')
      const obj = await env.DATA.get(key)
      if (!obj) return new Response('not found', { status: 404 })
      return new Response(obj.body, { headers: {
        'content-type': key.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        'content-length': String(obj.size),
        'cache-control': 'public, max-age=86400, immutable',
        'access-control-allow-origin': '*',
      } })
    }

    // بيانات التدريب لـColab — بالمفتاح (فيها صوت المستخدمين: خاصة تمامًا)
    const keyed = req.headers.get('x-mutqin-key') === env.KEY
    if (req.method === 'GET' && url.pathname === '/data/list') {
      if (!keyed) return new Response('nope', { status: 403 })
      const out = []
      let cursor
      do {
        const l = await env.DATA.list({ prefix: url.searchParams.get('prefix') ?? '', cursor, limit: 1000 })
        out.push(...l.objects.map((o) => ({ key: o.key, size: o.size })))
        cursor = l.truncated ? l.cursor : undefined
      } while (cursor)
      return Response.json(out)
    }
    if (req.method === 'GET' && url.pathname === '/data/get') {
      if (!keyed) return new Response('nope', { status: 403 })
      const obj = await env.DATA.get(url.searchParams.get('key') ?? '')
      return obj ? new Response(obj.body) : new Response('not found', { status: 404 })
    }

    if (req.method !== 'POST') return new Response('mutqin collector', { status: 200 })
    if (!keyed) return new Response('nope', { status: 403 })
    const day = new Date().toISOString().slice(0, 10)

    // رفع ملفات التدريب والموديل الجديد (من Colab) — على بادئتين مسموحتين فقط
    if (url.pathname === '/put') {
      const key = String(url.searchParams.get('key') ?? '')
      if (!/^(train|model)\/[\w.-]+$/.test(key)) return new Response('bad key', { status: 400 })
      await env.DATA.put(key, req.body)
      return Response.json({ ok: true, key })
    }

    if (url.pathname === '/events') {
      const body = await req.json()
      const device = String(body.device ?? 'unknown').replace(/[^\w-]/g, '')
      const key = `events/${device}/${day}/${Date.now()}.json`
      await env.DATA.put(key, JSON.stringify(body))
      return Response.json({ ok: true, key })
    }

    if (url.pathname === '/audio') {
      const device = String(url.searchParams.get('device') ?? 'unknown').replace(/[^\w-]/g, '')
      const name = String(url.searchParams.get('name') ?? 'file').replace(/[^\w.-]/g, '')
      const key = `audio/${device}/${day}/${name}`
      await env.DATA.put(key, req.body)
      return Response.json({ ok: true, key })
    }

    return new Response('not found', { status: 404 })
  },
}
