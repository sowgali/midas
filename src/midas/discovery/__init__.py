"""V1.9.2 BFS source-discovery.

Architectural shift: pre-V1.9.2 every IR source had to be hand-curated in
``ir_sources.yaml``. That works for the top-of-pyramid hyperscalers but
falls over the moment we want to follow chains downstream — every new
counterparty (Vertiv, Constellation, Crusoe, Fluidstack, ...) would
require a human to find the feed URL and add a YAML row before the
next ingest pass could touch them.

This module makes that loop **automatic**. For each entity discovered
during ingest:

1. Derive a small set of candidate domains from ``canonical_name``.
2. For each domain, probe a fixed list of common feed URL patterns
   (``/feed``, ``/news/rss``, ``/blog/feed.xml``, ``investors.{domain}``,
   etc.).
3. Validate via cheap HTTP GET + content-type / body sniff.
4. Persist any valid feed to the :class:`DiscoveredSource` table so the
   next ingest pass picks it up.

Combined with the :class:`midas.entity_resolution.EntityResolver`
open-world creation path, this gives us a true BFS expansion: each
ingest run can both surface new entities *and* the sources that
describe them, so the system grows automatically until the chain
terminates at entities with no public web presence.
"""

from .sources import (
    SourceCandidate,
    derive_domain_candidates,
    discover_for_entity,
    feed_url_candidates,
    is_feed_response,
    probe_feed,
)

__all__ = [
    "SourceCandidate",
    "derive_domain_candidates",
    "discover_for_entity",
    "feed_url_candidates",
    "is_feed_response",
    "probe_feed",
]
