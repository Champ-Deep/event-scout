# Held-out results (Event Scout scraper correction, t_d8124789)

Probe: `heldout/probe.py` = `scrape_event_site(url, use_llm=False)` over the real network, no DB. Pages saved in `heldout/pages/` (untracked, ignored).
Junk = extracted rows that are not exhibitors, counted by reading every name. Zero rows is NOT proof of recall.

## Final held-out, part 1 (round 3, rules frozen at da8b669)

| URL | pages fetched | rows (with booth/hall) | junk rows | floor map | map verdict | class / note |
|---|---|---|---|---|---|---|
| https://www.sial-paris.com/ | 0 | - | - | - | - | NOT FETCHED: unreachable: ConnectError: https://www.sial-paris.com/ (unreachable/blocked, no judgment possible) |
| https://www.ispo.com/ | 4 (visible text/'exhibitor' count per page: 8375ch/3exh, 3680ch/5exh, 5650ch/3exh, 3536ch/3exh) | 0 (0) | 0 | https://ispo2026.mapyourshow.com/8_0/exhview/index.cfm | genuine floor/hall plan (mapyourshow viewer, page 3 button '2026 ISPO Floorplan') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.anuga.com/ | 6 (visible text/'exhibitor' count per page: 7510ch/13exh, 13263ch/19exh, 12910ch/19exh, 12785ch/19exh, 12833ch/19exh, 12713ch/19exh) | 100 (100) | 0 | https://www.anuga.com/anuga-exhibitors/hall-plan/ | genuine floor/hall plan (nav link 'Hall plan' page) | extracted a precise directory. all 100 names read: real companies, each with 'Hall x | stand' |
| https://www.drupa.com/ | 2 (visible text/'exhibitor' count per page: 9942ch/0exh, 5077ch/0exh) | 0 (0) | 0 | https://www.drupa.com/en/trade-fair-information/preparation/hall-plan/ | genuine floor/hall plan (the fetched page itself is the hallenplan page; matched via its en alternate) | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.shotshow.org/ | 4 (visible text/'exhibitor' count per page: 5938ch/22exh, 6455ch/33exh, 10372ch/32exh, 25ch/0exh) | 0 (0) | 0 | https://shotshow.org/floorplan | genuine floor/hall plan (menu link 'Floorplan' -> goexposoftware viewer page) | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.ifsecglobal.com/ | 0 | - | - | - | - | NOT FETCHED: blocked: HTTP 403 (unreachable/blocked, no judgment possible) |
| https://www.interpack.com/ | 4 (visible text/'exhibitor' count per page: 9316ch/14exh, 4869ch/18exh, 4187ch/26exh, 3136ch/9exh) | 0 (0) | 0 | https://www.interpack.com/en/Visit/Preparations/Floor_Plan_Site_Map | genuine floor/hall plan (nav 'Floor Plan & Site Map') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.eimamilano.com/ | 0 | - | - | - | - | NOT FETCHED: unreachable: cannot resolve www.eimamilano.com (unreachable/blocked, no judgment possible) |
| https://www.cosmoprof.com/ | 4 (visible text/'exhibitor' count per page: 11946ch/0exh, 8307ch/0exh, 10912ch/0exh, 10256ch/0exh) | 0 (0) | 0 | none | none found | 0 rows (miss vs no-static-directory NOT established; recall unproven). static pages, no exhibitor wording at all in 4 fetched pages (visible text has 0 'exhibitor') |
| https://www.intersolar.de/ | 4 (visible text/'exhibitor' count per page: 7197ch/21exh, 12483ch/20exh, 6162ch/35exh, 8021ch/17exh) | 0 (0) | 0 | https://www.intersolar.de/exhibition-halls | genuine floor/hall plan (nav 'Exhibition Site Plan') | 0 rows (miss vs no-static-directory NOT established; recall unproven). /exhibitorlist is nav + intro text; list is loaded dynamically |
| https://www.chinaplas.com/ | 0 | - | - | - | - | NOT FETCHED: unreachable: ConnectError: https://www.chinaplas.com/ (unreachable/blocked, no judgment possible) |
| https://www.siggraph.org/ | 0 | - | - | - | - | NOT FETCHED: no_content: HTTP 307 (unreachable/blocked, no judgment possible) |

## Final held-out, part 2 (round 3b, rules unchanged)

| URL | pages fetched | rows (with booth/hall) | junk rows | floor map | map verdict | class / note |
|---|---|---|---|---|---|---|
| https://www.fachpack.de/ | 4 (visible text/'exhibitor' count per page: 7888ch/16exh, 5436ch/19exh, 3876ch/12exh, 4973ch/12exh) | 0 (0) | 0 | https://edge.sitecorecloud.io/nurnbergmes0e18-nmmultisite6e34-prod50da-6065/media/Project/NuernbergMesse/FACHP | genuine floor/hall plan (PDF button 'Download FACHPACK 2027 Floor Plan') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.productronica.com/ | 3 (visible text/'exhibitor' count per page: 4604ch/0exh, 6054ch/0exh, 3716ch/0exh) | 0 (0) | 0 | https://productronica.com/en/trade-fair/hall-plan/ | genuine floor/hall plan (hall-plan page) | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.electronica.de/ | 4 (visible text/'exhibitor' count per page: 4582ch/6exh, 3355ch/28exh, 7372ch/16exh, 3927ch/6exh) | 0 (0) | 0 | https://electronica.de/en/trade-fair/hall-plan/ | genuine floor/hall plan (nav 'Hall plan') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://sps.mesago.com/ | 4 (visible text/'exhibitor' count per page: 9404ch/10exh, 3188ch/12exh, 20511ch/40exh, 9139ch/17exh) | 0 (0) | 0 | https://sps.mesago.com/nuernberg/en/expo/planning-preparation/venue-site.html | genuine floor/hall plan (nav 'Hall & site plan') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.analytica.de/ | 2 (visible text/'exhibitor' count per page: 5379ch/0exh, 4734ch/0exh) | 0 (0) | 0 | https://analytica.de/en/munich/hall-plan/ | genuine floor/hall plan (hall-plan page) | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.k-online.com/ | 4 (visible text/'exhibitor' count per page: 5866ch/15exh, 5009ch/18exh, 4274ch/25exh, 4505ch/14exh) | 0 (0) | 0 | https://www.k-online.com/cgi-bin/md_k/lib/pub/object/downloadfile.cgi/KMD25_19_Gela%CC%88ndeplan_DIN_A4_202508 | genuine floor/hall plan (PDF 'Download site map', Gelaendeplan) | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.euroshop-tradefair.com/ | 4 (visible text/'exhibitor' count per page: 13313ch/18exh, 8657ch/23exh, 54ch/0exh, 4089ch/14exh) | 0 (0) | 0 | https://www.euroshop-tradefair.com/cgi-bin/md_euroshop/lib/pub/object/downloadfile.cgi/ESP26_19_Hallenplan_202 | genuine floor/hall plan (PDF 'Download hall plan') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.globalpetexpo.org/ | 3 (visible text/'exhibitor' count per page: 8335ch/6exh, 1581ch/8exh, 5848ch/16exh) | 0 (0) | 0 | https://globalpetexpo27.mapyourshow.com/8_0/floorplan/index.cfm?hallID=campus&level=0 | genuine floor/hall plan (mapyourshow '2027 Show Floorplan') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.glasstec-online.com/ | 4 (visible text/'exhibitor' count per page: 12845ch/19exh, 3505ch/15exh, 5246ch/15exh, 3835ch/12exh) | 0 (0) | 0 | https://www.glasstec-online.com/en/Visitors/Preparation/Hall-and-site-plans | genuine floor/hall plan (nav 'Hall & site plans') | 0 rows (miss vs no-static-directory NOT established; recall unproven).  |
| https://www.expopharm.de/ | 2 (visible text/'exhibitor' count per page: 14631ch/0exh, 10897ch/0exh) | 0 (0) | 0 | none | none found | 0 rows (miss vs no-static-directory NOT established; recall unproven). 2 pages fetched, nothing exhibitor-like |

## Earlier rounds (rules were then changed; kept as an honest record, not part of the final claim)

* Round 1 (rules aa858c5): semashow.com 4 junk rows (venue halls), bauma.de 8 junk rows (teasers/page metadata), imts.com map = 'Show Planner' login tool then a single-booth link. Fixed in 5423842.
* Round 2 (rules 5423842): agritechnica.com 4 junk rows (teaser cards), nationalrestaurantshow.com 3 junk rows (show-floor theaters with booth numbers). Fixed in da8b669. ifa-berlin.com 140 real rows (booth text 'Hall 4.1 H4.1-112, H4.1-112' duplicated: data-quality note, not junk).
* Round 3 fetched only 7 of 12 sites (sial-paris, ifsecglobal 403, eimamilano DNS, chinaplas, siggraph unreachable), so round 3b added 10 fresh sites with unchanged rules.
