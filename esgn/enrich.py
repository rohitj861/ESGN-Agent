"""Enrichment: ESG pillar tagging, topic + regulation extraction, market
attribution and a relevance score.

Keyword driven on purpose -- transparent, cheap, and auditable. Swap
`score_article` for a model call later without touching the pipeline.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# ESG pillars
# ---------------------------------------------------------------------------
PILLAR_KEYWORDS = {
    "E": [
        "climate", "emission", "carbon", "net zero", "net-zero", "decarboni",
        "greenhouse gas", "ghg", "scope 1", "scope 2", "scope 3", "renewable",
        "solar", "wind power", "offshore wind", "biodiversity", "deforestation",
        "circular economy", "waste", "recycling", "water stress", "pollution",
        "air quality", "energy transition", "clean energy", "fossil fuel",
        "coal", "methane", "nature-based", "sustainable aviation fuel",
        "electric vehicle", "battery", "hydrogen", "carbon capture",
        "climate risk", "physical risk", "transition risk", "adaptation",
        "environmental impact", "green bond", "carbon credit", "carbon market",
        "offset", "science-based target", "sbti",
    ],
    "S": [
        "human rights", "labour", "labor rights", "forced labour",
        "forced labor", "child labour", "child labor", "modern slavery",
        "supply chain due diligence", "diversity", "inclusion", "dei",
        "gender pay", "pay gap", "workforce", "health and safety",
        "worker safety", "community impact", "indigenous", "just transition",
        "living wage", "employee wellbeing", "social impact",
        "affordable housing", "data privacy", "product safety",
        "responsible sourcing", "conflict minerals",
    ],
    "G": [
        "governance", "board", "executive pay", "executive compensation",
        "say on pay", "shareholder", "proxy", "proxy voting", "activist",
        "disclosure", "reporting standard", "assurance", "audit",
        "anti-corruption", "bribery", "whistleblower", "greenwashing",
        "fiduciary", "stewardship", "materiality", "double materiality",
        "compliance", "enforcement", "fine", "lawsuit", "litigation",
        "sanction", "transparency", "esg rating", "ratings agency",
    ],
}

# ---------------------------------------------------------------------------
# Regulations / frameworks -- the high-value entities for ESG desks
# ---------------------------------------------------------------------------
REGULATION_PATTERNS = {
    "CSRD": r"\bcsrd\b|corporate sustainability reporting directive",
    "CSDDD": r"\bcsddd\b|\bcs3d\b|corporate sustainability due diligence",
    "SFDR": r"\bsfdr\b|sustainable finance disclosure regulation",
    "EU_Taxonomy": r"eu taxonomy|taxonomy regulation",
    "ESRS": r"\besrs\b|european sustainability reporting standards",
    "CBAM": r"\bcbam\b|carbon border adjustment",
    "EUDR": r"\beudr\b|deforestation regulation",
    "EU_ETS": r"\beu ets\b|emissions trading system",
    "Green_Claims_Directive": r"green claims directive|empowering consumers directive",
    # "Omnibus" alone also means a US spending bill, so require an EU
    # sustainability term somewhere in the text (either side of the word).
    "Omnibus": (
        r"omnibus (?:package|proposal|simplification|directive|regulation)"
        r"|(?:simplification|sustainability) omnibus"
        r"|\bomnibus\b(?=[\s\S]*(?:csrd|csddd|sfdr|esrs|taxonomy|sustainab))"
        r"|(?:csrd|csddd|sfdr|esrs|taxonomy|sustainab)[\s\S]*?\bomnibus\b"
    ),
    "SEC_Climate_Rule": r"sec climate(?:[- ]related)? (?:disclosure )?rule|climate disclosure rule",
    "California_SB253": r"\bsb[- ]?253\b|climate corporate data accountability",
    "California_SB261": r"\bsb[- ]?261\b",
    "IRA": r"inflation reduction act\b",
    "ISSB": r"\bissb\b|ifrs s1|ifrs s2|international sustainability standards board",
    "TCFD": r"\btcfd\b|task force on climate",
    "TNFD": r"\btnfd\b|taskforce on nature",
    "GRI": r"\bgri standards?\b|global reporting initiative",
    "Paris_Agreement": r"paris agreement",
    "UK_SDR": r"\bsdr\b|sustainability disclosure requirements",
}

# ---------------------------------------------------------------------------
# Topical buckets for downstream filtering / routing
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    "regulation_policy": [
        "regulation", "directive", "rule", "legislation", "parliament",
        "council", "policy", "mandate", "guidance", "consultation",
        "framework", "law", "bill", "act",
    ],
    "reporting_disclosure": [
        "disclosure", "reporting", "report", "assurance", "audit",
        "materiality", "taxonomy", "kpi", "metrics",
    ],
    "sustainable_finance": [
        "green bond", "sustainability-linked", "esg fund", "impact investing",
        "transition finance", "blended finance", "asset manager", "divest",
        "sustainable finance", "climate finance", "aum",
    ],
    "carbon_markets": [
        "carbon credit", "carbon market", "offset", "carbon price",
        "emissions trading", "article 6", "removal credit",
    ],
    "energy_transition": [
        "renewable", "solar", "wind", "hydrogen", "grid", "battery storage",
        "nuclear", "power purchase agreement", "ppa", "electrification",
    ],
    "corporate_action": [
        "announced", "launches", "acquires", "partnership", "invests",
        "commits", "pledge", "target", "appoints",
    ],
    "litigation_enforcement": [
        "lawsuit", "sues", "court", "ruling", "fine", "penalty",
        "enforcement", "settlement", "investigation", "charges",
    ],
    "greenwashing": [
        "greenwash", "misleading claim", "false advertising",
        "unsubstantiated claim",
    ],
    "supply_chain": [
        "supply chain", "supplier", "due diligence", "traceability",
        "sourcing", "procurement",
    ],
}

# ---------------------------------------------------------------------------
# Market attribution signals (used when the source is GLOBAL)
# ---------------------------------------------------------------------------
# Market signals split into two groups.
#
# Unambiguous words are matched case-insensitively. Bare acronyms are matched
# CASE-SENSITIVELY against the original headline text, because lowercasing
# first turns "US" (the country) into "us" (the pronoun) and "DOE" (Department
# of Energy) into "doe" (a deer) -- both of which fire constantly on nature and
# climate copy. Word boundaries alone do not save you here.
EU_WORD_SIGNALS = [
    r"european union", r"brussels", r"european commission",
    r"european parliament", r"european council", r"european central bank",
    r"eurozone", r"germany", r"german\b", r"france", r"french\b", r"spain",
    r"spanish\b", r"italy", r"italian\b", r"netherlands", r"dutch\b",
    r"poland", r"polish\b", r"sweden", r"swedish\b", r"denmark", r"danish\b",
    r"belgium", r"ireland", r"irish\b", r"austria", r"portugal", r"finland",
    r"greece", r"czech", r"eu taxonomy",
]
EU_ACRONYM_SIGNALS = [
    r"\bEU\b", r"\bEU's\b", r"\bEBA\b", r"\bESMA\b", r"\bEIOPA\b", r"\bECB\b",
    r"\bCSRD\b", r"\bSFDR\b", r"\bESRS\b", r"\bCBAM\b", r"\bEUDR\b",
    r"\bCSDDD\b", r"\bEU ETS\b",
]

US_WORD_SIGNALS = [
    r"united states", r"american\b", r"washington", r"congress", r"senate",
    r"white house", r"california", r"texas", r"new york", r"biden", r"trump",
    r"federal reserve", r"inflation reduction act", r"state department",
    r"supreme court", r"department of energy",
    r"environmental protection agency", r"securities and exchange commission",
]
US_ACRONYM_SIGNALS = [
    r"\bU\.S\.", r"\bUS\b", r"\bUSA\b", r"\bSEC\b", r"\bEPA\b", r"\bFERC\b",
    r"\bCFTC\b", r"\bFHFA\b", r"\bDOE\b", r"\bIRS\b", r"\bNOAA\b",
    r"\bSB[- ]?253\b", r"\bSB[- ]?261\b",
]

EU_WORD_RE = re.compile("|".join(EU_WORD_SIGNALS), re.IGNORECASE)
US_WORD_RE = re.compile("|".join(US_WORD_SIGNALS), re.IGNORECASE)
EU_ACRONYM_RE = re.compile("|".join(EU_ACRONYM_SIGNALS))   # case-sensitive
US_ACRONYM_RE = re.compile("|".join(US_ACRONYM_SIGNALS))   # case-sensitive

# Core ESG vocabulary used for the base relevance signal.
#
# Matched as word-boundary regexes, not substrings. Bare containment produced
# real false positives in production: "green" hit "green light" and
# "Greenland", and "transition" hit "the post-Orban transition" -- pulling
# Gaza and Hungarian politics into an ESG dataset. Ambiguous words therefore
# carry either a negative lookahead or a required domain qualifier.
_CORE_PATTERNS = [
    r"\besg\b",
    r"sustainab",                      # sustainable / sustainability
    r"\bclimate\b",
    r"\bcarbon\b",
    r"\bemissions?\b",
    r"net[-\s]zero",
    r"\bgreen\b(?!\s+light)",          # not "green light"; \b excludes Greenland
    r"\brenewables?\b",
    r"decarboni",
    r"\benvironmental\b",
    r"\bgovernance\b",
    r"\bbiodiversity\b",
    r"circular economy",
    r"\bfossil fuels?\b",
    # "transition" only counts with an energy/climate qualifier.
    r"(?:energy|just|climate|low[-\s]carbon|green|net[-\s]zero) transition",
    r"transition (?:plan|finance|risk|pathway)",
]

_CORE_RE = re.compile("|".join(_CORE_PATTERNS), re.IGNORECASE)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _raw_text(article):
    """Original-case searchable text -- required for acronym matching."""
    parts = [
        article.get("title") or "",
        article.get("summary") or "",
        " ".join(article.get("categories") or []),
    ]
    return " " + " ".join(parts) + " "


def _haystack(article):
    """Lowercased searchable text for one article."""
    return _raw_text(article).lower()


def detect_pillars(text):
    return [p for p, kws in PILLAR_KEYWORDS.items() if any(k in text for k in kws)]


def detect_regulations(text):
    return sorted(
        name for name, pattern in REGULATION_PATTERNS.items()
        if re.search(pattern, text, re.IGNORECASE)
    )


def detect_topics(text):
    return sorted(
        topic for topic, kws in TOPIC_KEYWORDS.items() if any(k in text for k in kws)
    )


def detect_markets(text, raw_text, source_market):
    """Markets an article is relevant to. Source market always counts.

    `text` is lowercased; `raw_text` preserves case for acronym matching.
    """
    markets = set()
    if source_market in ("EU", "US"):
        markets.add(source_market)
    if EU_WORD_RE.search(text) or EU_ACRONYM_RE.search(raw_text):
        markets.add("EU")
    if US_WORD_RE.search(text) or US_ACRONYM_RE.search(raw_text):
        markets.add("US")
    if not markets:
        markets.add("GLOBAL")
    return sorted(markets)


def score_article(article, text, pillars, regulations, topics):
    """0.0-1.0 relevance heuristic.

    Weighted so that an on-topic regulator item about CSRD outranks a passing
    mention of "green" in a general business story.
    """
    score = 0.0

    # Distinct vocabulary matched, not raw occurrences -- one word repeated
    # five times is weaker evidence than five different ESG terms.
    core_hits = len({m.group(0).lower() for m in _CORE_RE.finditer(text)})
    score += min(core_hits * 0.08, 0.35)

    if _CORE_RE.search(article.get("title") or ""):
        score += 0.20

    score += min(len(pillars) * 0.07, 0.21)
    score += min(len(regulations) * 0.12, 0.24)
    score += min(len(topics) * 0.03, 0.12)

    if article.get("source_type") == "regulator" and regulations:
        score += 0.08

    # Thin items (headline only, no summary) are less useful downstream.
    if not article.get("summary"):
        score -= 0.05

    return round(max(0.0, min(score, 1.0)), 4)


def normalize_title(title):
    """Whitespace/punctuation-insensitive title key for near-duplicate collapse."""
    if not title:
        return ""
    return " ".join(_WORD_RE.findall(title.lower()))


def enrich(article):
    """Return a new dict with enrichment fields added."""
    raw_text = _raw_text(article)
    text = raw_text.lower()
    pillars = detect_pillars(text)
    regulations = detect_regulations(text)
    topics = detect_topics(text)
    markets = detect_markets(text, raw_text, article.get("source_market", "GLOBAL"))

    enriched = dict(article)
    enriched.update(
        {
            "esg_pillars": pillars,
            "regulations": regulations,
            "topics": topics,
            "markets": markets,
            "is_eu": "EU" in markets,
            "is_us": "US" in markets,
            "relevance_score": score_article(
                article, text, pillars, regulations, topics
            ),
            "has_publish_date": bool(article.get("published_at")),
            "title_key": normalize_title(article.get("title")),
            "dedupe_key": article.get("canonical_url")
            or normalize_title(article.get("title")),
        }
    )
    return enriched


def is_esg_relevant(article, threshold=0.15):
    """Gate for the curated layer.

    Regulator items with a detected regulation always pass -- those are the
    primary-source records an ESG desk cannot afford to drop on a score.
    """
    if article.get("source_type") == "regulator" and article.get("regulations"):
        return True
    if article.get("esg_pillars") and article.get("relevance_score", 0) >= threshold:
        return True
    return article.get("relevance_score", 0) >= threshold + 0.15
