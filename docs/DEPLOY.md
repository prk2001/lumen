# Deploying the marketing site

`apps/web/` is a single-file static site. It deploys to any major host
without a build step — `index.html` is the entry, all assets are
relative.

Currently live at **<https://prk2001.github.io/lumen/>** via GitHub
Pages, auto-redeployed on every push to `main` that touches
`apps/web/**` (see `.github/workflows/pages.yml`).

## Quick switch table

| Host | One-liner | Config file |
| --- | --- | --- |
| **GitHub Pages** | already wired (Actions auto-deploy) | `.github/workflows/pages.yml` |
| **Netlify** | drag-drop `apps/web/` or connect repo | `apps/web/netlify.toml` |
| **Cloudflare Pages** | `wrangler pages deploy apps/web` | `apps/web/wrangler.toml` |
| **Vercel** | `vercel --cwd apps/web --prod` | `apps/web/vercel.json` |
| **AWS S3 + CloudFront** | `aws s3 sync apps/web/ s3://bucket/` | n/a — use the CSP/HSTS headers from `_headers` |
| **Local preview** | `python3 -m http.server 8088` from `apps/web/` | n/a |

## What's in `apps/web/`

```text
apps/web/
├── index.html           # 580 lines, ~30 KB; one-file landing page
├── 404.html             # themed not-found
├── favicon.svg          # gold-dot brand mark on dark plate
├── og.png               # 1200x630 social card
├── robots.txt
├── sitemap.xml
├── _headers             # Netlify / Cloudflare Pages security headers
├── _redirects           # Netlify / Cloudflare 404 fallback
├── netlify.toml
├── vercel.json
└── wrangler.toml        # Cloudflare Pages
```

## Custom domain (custom URL)

The site currently lives at the GitHub-Pages default URL
`https://prk2001.github.io/lumen/`. To switch to a custom domain:

1. **Create the DNS record** at your registrar (apex or subdomain):
   - For `lumen.primorispartners.com`, add a `CNAME` pointing at
     `prk2001.github.io`.
   - For an apex (`primorispartners.com`), use four A records pointing
     at GitHub's Pages IPs (185.199.108.153–111.153). See
     [GitHub docs](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site).
2. **Drop a `CNAME` file** into `apps/web/` containing just the host:
   ```
   lumen.primorispartners.com
   ```
3. **Update `sitemap.xml`, `robots.txt`, the `<link rel="canonical">`,
   and the OG `og:url` meta tag** in `index.html` to the new URL
   (search for `lumen.primorispartners.com` — already pointed at it as
   the canonical).
4. Push. GitHub Pages picks up the CNAME on next deploy.

## Security headers

The `_headers` / `vercel.json` / `netlify.toml` configs ship the same
header set:

- **HSTS** preload (2-year max-age, includeSubDomains, preload)
- **X-Frame-Options** DENY
- **X-Content-Type-Options** nosniff
- **Referrer-Policy** strict-origin-when-cross-origin
- **Permissions-Policy** interest-cohort=()
- **Content-Security-Policy** default-src 'self'; img-src 'self' data:;
  style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self';
  connect-src 'self'

GitHub Pages does not expose custom headers — rely on the defaults
(GitHub serves Pages with HSTS already enforced via
`https_enforced=true`).

## Smoke-testing a deploy

```bash
for p in / og.png favicon.svg robots.txt sitemap.xml 404.html nope; do
  curl -sIL -o /dev/null -w "%{http_code}  $p\n" "https://YOUR-HOST/$p"
done
```

Expected: `200` for everything except `nope` which should `404`.

## Roadmap for the site itself

The current page is a one-file static landing. Phase 6 of the project
plan turns it into a Vite + React + WASM-core build that lets the
visitor actually run Lumen pipelines in the browser. The current page
content (and its dark-theme palette) carry forward into that build.
