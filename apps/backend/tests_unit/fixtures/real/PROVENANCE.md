# Real fetched pages (regression fixtures)

Each page was fetched on 2026-11-06 (IST) with this repo's own `event_scraper.crawl()` (real `fetch_page`, SSRF guard on),
starting from the start URL below, and stored gzip-compressed and byte-identical to what the scraper saw.
`manifest.json` has per page: final URL, fetch time (UTC), size, sha256 of the raw HTML.

| key | start URL | why it is here |
|-----|-----------|----------------|
| hlth | https://hlth.com/ | session titles + site name emitted as exhibitors; date/venue strings read as country |
| fabtech | https://www.fabtechexpo.com/ | ld+json owning associations (AWS, FMA, SME, PMA, IFCA) emitted as exhibitors, status ok |
| packexpo | https://www.packexpolasvegas.com/ | UI icon chosen as floor map over the "View Floor Plans" mapyourshow link |
| inbound | https://www.inbound.com/ (redirects to https://unbound.hubspot.com/) | sponsor logo ("Map My Customers logo") chosen as floor map |
| dax | https://www.daxshow.com/exhibit_list_and_show_special/169/ | control: 30 exhibitors with booths + PNG floor map |
| automate | https://www.automate.org/ | control: company cards with country (15 real exhibitors), home page ld+json is the site itself |

Pages are third-party content, kept only as test input.

## Added after held-out round 1 (these three sites exposed junk, so they are now development fixtures, not held-out)

| key | start URL | why it is here |
|-----|-----------|----------------|
| sema | https://www.semashow.com/ | venue hall tiles ("West Hall", "Central Hall") on /exhibitor emitted as exhibitors |
| bauma | https://www.bauma.de/en/ | promo/teaser cards and page metadata emitted as exhibitors via a "Hall plans" text match |
| imts | https://www.imts.com/ | only page 3 kept: "Show Planner" (login tool) and a per-booth `floorplan_link.cfm?...booth=` link scored as a floor map |
