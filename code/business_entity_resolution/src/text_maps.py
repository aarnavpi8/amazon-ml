"""Static normalisation vocabularies: legal forms, abbreviations, state/region names.

These are general language / postal conventions (like "Rd" = "Road"), not looked-up
business data. Keys are already lower-cased and accent-free, matching the cleaned text.
Countries without an entry fall back to the default maps, so the pipeline works for any
country label.
"""

# ---------------------------------------------------------------- names

# Token rewrites applied to every name token (spelling variants -> one form).
NAME_TOKEN_MAP = {
    "private": "pvt", "limited": "ltd", "incorporated": "inc", "corporation": "corp",
    "company": "co", "cie": "co", "etablissements": "ets", "etablissement": "ets",
    "intl": "international", "mfg": "manufacturing", "sys": "systems", "bros": "brothers",
    "mgmt": "management", "ctr": "center", "cntr": "center", "centre": "center",
    "svc": "services", "svcs": "services", "service": "services", "grp": "group",
    "hldgs": "holdings", "univ": "university", "inst": "institute", "natl": "national",
    "govt": "government", "dept": "department", "hosp": "hospital",
    "shree": "shri", "sri": "shri", "sree": "shri",
}

# Legal-form tokens (after NAME_TOKEN_MAP). Kept separately, removed from the core name.
LEGAL_TOKENS = {
    "pvt", "ltd", "inc", "corp", "co", "llc", "llp", "lp", "pllc", "pc", "plc",
    "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "ei", "gmbh", "ets",
}

# Tokens that carry no identity (honorifics, articles, dba marker) — removed from the core name.
NAME_NOISE_TOKENS = {"the", "mr", "mrs", "ms", "smt", "dr", "messrs", "dba", ""}

# "doing business as" style separators, replaced by a " dba " marker token.
DBA_PATTERN = (
    r"\b(?:d\s*/\s*b\s*/\s*a|d\.b\.a|dba|doing business as|t\s*/\s*a|trading as"
    r"|a\s*/\s*k\s*/\s*a|aka)\b"
)

# Domain suffixes stripped from website-style names ("srgold.com" -> "srgold").
DOMAIN_PATTERN = r"\.(?:co\.in|com|net|org|in|co|fr|biz|info|us|io)\b"

# ---------------------------------------------------------------- addresses

# Placeholder values injected into addresses.
ADDRESS_PLACEHOLDER_PATTERN = r"<\s*null\s*>|\bnull\b|\bn\s*/\s*a\b|\bnone\b|\bnan\b"

# Token rewrites for every country. Mapping to "" drops the token.
DEFAULT_ADDRESS_ABBREV = {
    "st": "street", "str": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "dr": "drive", "ln": "lane", "ct": "court", "cir": "circle",
    "pl": "place", "pkwy": "parkway", "hwy": "highway", "trl": "trail", "sq": "square",
    "rt": "route", "rte": "route", "mt": "mount", "ft": "fort", "pt": "point",
    "hts": "heights", "jct": "junction", "expy": "expressway", "fwy": "freeway",
    "twp": "township", "ste": "suite", "apt": "apartment", "fl": "floor", "flr": "floor",
    "bldg": "building", "blk": "block", "sec": "sector", "ph": "phase", "opp": "opposite",
    "nr": "near", "dist": "district", "tq": "taluka", "tal": "taluka", "tk": "taluka",
    "mkt": "market", "stn": "station", "rly": "railway", "ext": "extension",
    "extn": "extension", "ind": "industrial", "indl": "industrial", "no": "", "hno": "",
}

# Country-specific overrides layered on top of the defaults.
COUNTRY_ADDRESS_ABBREV = {
    "India": {"dr": "doctor"},
    "US": {
        "n": "north", "s": "south", "e": "east", "w": "west",
        "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    },
    "France": {
        "r": "rue", "av": "avenue", "ave": "avenue", "bd": "boulevard", "bld": "boulevard",
        "bldv": "boulevard", "blvd": "boulevard", "boul": "boulevard", "all": "allee",
        "imp": "impasse", "pl": "place", "rte": "route", "ch": "chemin", "chem": "chemin",
        "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "crs": "cours", "st": "saint",
        "ste": "sainte", "res": "residence", "n": "",
    },
}

_US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "district of columbia": "dc",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id", "illinois": "il",
    "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi",
    "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn",
    "texas": "tx", "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy", "puerto rico": "pr",
}

_INDIA_STATE_CODES = {
    "an": "andaman and nicobar islands", "ap": "andhra pradesh", "ar": "arunachal pradesh",
    "as": "assam", "br": "bihar", "cg": "chhattisgarh", "ct": "chhattisgarh",
    "ch": "chandigarh", "dl": "delhi", "ga": "goa", "gj": "gujarat",
    "hp": "himachal pradesh", "hr": "haryana", "jh": "jharkhand", "jk": "jammu and kashmir",
    "ka": "karnataka", "kl": "kerala", "la": "ladakh", "ld": "lakshadweep",
    "mh": "maharashtra", "ml": "meghalaya", "mn": "manipur", "mp": "madhya pradesh",
    "mz": "mizoram", "nl": "nagaland", "od": "odisha", "or": "odisha", "pb": "punjab",
    "py": "puducherry", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "tg": "telangana", "ts": "telangana", "tr": "tripura", "uk": "uttarakhand",
    "ut": "uttarakhand", "up": "uttar pradesh", "wb": "west bengal",
}
_INDIA_STATE_VARIANTS = {
    "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
    "tamilnadu": "tamil nadu", "chattisgarh": "chhattisgarh", "chhatisgarh": "chhattisgarh",
    "jammu kashmir": "jammu and kashmir",
}

_FRANCE_REGIONS = [
    "auvergne rhone alpes", "bourgogne franche comte", "bretagne", "centre val de loire",
    "corse", "grand est", "hauts de france", "ile de france", "normandie",
    "nouvelle aquitaine", "occitanie", "pays de la loire", "provence alpes cote d azur",
]
# Departement -> region for the departements that appear in the data.
_FRANCE_DEPARTEMENTS = {
    "nord": "hauts de france", "pas de calais": "hauts de france",
    "gironde": "nouvelle aquitaine", "loire atlantique": "pays de la loire",
}

# Whole-segment state/region canonicalisation per country: segment text -> canonical form.
STATE_MAPS = {
    "US": {**_US_STATES, **{code: code for code in _US_STATES.values()}},
    "India": {
        **_INDIA_STATE_CODES,
        **_INDIA_STATE_VARIANTS,
        **{name: name for name in _INDIA_STATE_CODES.values()},
    },
    "France": {**{r: r for r in _FRANCE_REGIONS}, **_FRANCE_DEPARTEMENTS},
}


# States merged into one block during blocking. S2/S3 still label Hyderabad addresses with
# the pre-2014 state (Andhra Pradesh) — 36k of the 37k Indian state mismatches in train.
BLOCK_STATE_ALIASES = {"India": {"telangana": "andhra pradesh"}}

# State names that are also common city names ("1313 Fairmont St, Washington, DC").
# When an address has another state candidate, these lose the tie-break.
AMBIGUOUS_STATE_SEGMENTS = {"US": {"washington"}}


def address_abbrev_for(country):
    """Token abbreviation map for a country (defaults + overrides)."""
    return {**DEFAULT_ADDRESS_ABBREV, **COUNTRY_ADDRESS_ABBREV.get(country, {})}


def state_map_for(country):
    """Segment -> canonical state map for a country (empty for unknown countries)."""
    return STATE_MAPS.get(country, {})


def ambiguous_states_for(country):
    """State segments that should lose ties to other state candidates."""
    return sorted(AMBIGUOUS_STATE_SEGMENTS.get(country, set()))
