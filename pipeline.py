"""Week 3: beauty & wellness review-data cleaning pipeline (simulated Sephora-style data).
Run: python pipeline.py  -> writes cleaned CSV, step log (JSON) and figures.
"""
import hashlib, html, json, re
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
LOG = []  # one record per step: rows before/after and what changed


def log(step, before, after, note=""):
    LOG.append({"step": step, "rows_before": int(before), "rows_after": int(after), "note": note})


# ---------------------------------------------------------------- 1. simulate raw data
def simulate_raw(n_products=300, n_reviews=20000):
    brands = ["The Ordinary", "CeraVe", "Fenty Beauty", "Drunk Elephant", "Glow Recipe",
              "La Roche-Posay", "Neutrogena", "Tatcha", "Kiehl's", "Paula's Choice"]
    cats = ["Moisturizers", "Cleansers", "Serums", "Sunscreen", "Eye Care", "Masks"]
    actives = ["niacinamide", "retinol", "hyaluronic acid", "salicylic acid", "glycerin",
               "ceramide np", "vitamin c", "fragrance", "alcohol denat.", "panthenol", "squalane"]
    prod = pd.DataFrame({
        "product_id": [f"P{i:04d}" for i in range(n_products)],
        "brand_name": RNG.choice(brands, n_products),
        "primary_category": RNG.choice(cats, n_products),
        "price_usd": np.round(np.exp(RNG.normal(3.3, 0.7, n_products)), 2),
    })
    prod["ingredients"] = [", ".join(RNG.choice(actives, RNG.integers(5, 10), replace=False)) for _ in range(n_products)]

    skin_types = ["dry", "oily", "combination", "normal", "sensitive"]
    skin_tones = ["fair", "light", "medium", "tan", "deep", "dark"]
    rev = pd.DataFrame({
        "author_id": RNG.integers(1, 9000, n_reviews).astype(str),
        "product_id": RNG.choice(prod.product_id, n_reviews),
        "rating": RNG.choice([1, 2, 3, 4, 5], n_reviews, p=[.05, .06, .11, .26, .52]).astype(float),
        "skin_type": RNG.choice(skin_types, n_reviews),
        "skin_tone": RNG.choice(skin_tones, n_reviews),
        "submission_time": pd.to_datetime("2019-01-01") + pd.to_timedelta(RNG.integers(0, 1500, n_reviews), unit="D"),
    })
    pos = ["Love it, so hydrating!", "Great texture, absorbs fast", "My skin feels amazing", "Holy grail product"]
    neg = ["Broke me out badly", "Too greasy for my skin", "Irritated my face, burning", "Smells too strong, waste of money"]
    rev["review_text"] = [RNG.choice(pos) if r >= 4 else RNG.choice(neg) if r <= 2 else "It is okay, nothing special"
                          for r in rev.rating]
    rev["is_recommended"] = (rev.rating >= 4).astype(float)
    flip = RNG.random(n_reviews) < 0.07
    rev.loc[flip, "is_recommended"] = 1 - rev.loc[flip, "is_recommended"]
    df = rev.merge(prod, on="product_id")

    # ---- inject realistic quality problems
    n = len(df)
    df.loc[RNG.random(n) < 0.18, "skin_type"] = np.nan                  # optional attribute left blank
    df.loc[RNG.random(n) < 0.22, "skin_tone"] = np.nan
    df.loc[RNG.random(n) < 0.12, "is_recommended"] = np.nan
    df.loc[RNG.random(n) < 0.03, "review_text"] = np.nan
    df["skin_type"] = df.skin_type.map(lambda v: v if pd.isna(v) or RNG.random() > .15 else RNG.choice([v.upper(), v.title(), f" {v} "]))
    df["brand_name"] = df.brand_name.map(lambda v: RNG.choice([v, v.upper(), v.lower(), v + " "]) if RNG.random() < .2 else v)
    df["rating"] = df.rating.astype(object)
    bad = RNG.random(n) < 0.004
    df.loc[bad, "rating"] = RNG.choice([0, 6, 7, -1], bad.sum())          # impossible ratings
    df["price_usd"] = df.price_usd.astype(object)
    s = RNG.random(n) < 0.05
    df.loc[s, "price_usd"] = df.loc[s, "price_usd"].map(lambda v: f"${v}")  # price as text
    o = RNG.random(n) < 0.003
    df.loc[o, "price_usd"] = df.loc[o, "price_usd"].map(lambda v: float(str(v).strip("$")) * 100)  # decimal slip
    df.loc[RNG.random(n) < 0.002, "price_usd"] = -5
    df["submission_time"] = df.submission_time.dt.strftime("%Y-%m-%d")
    m = RNG.random(n) < 0.04
    df.loc[m, "submission_time"] = pd.to_datetime(df.loc[m, "submission_time"]).dt.strftime("%d/%m/%Y")
    df.loc[RNG.random(n) < 0.002, "submission_time"] = "2031-02-30"
    df.loc[RNG.random(n) < 0.002, "submission_time"] = "not available"
    t = RNG.random(n) < 0.05
    df.loc[t, "review_text"] = df.loc[t, "review_text"].map(lambda v: f"<p>{v}</p> &amp; so good \U0001F60D\U0001F60D!!!" if isinstance(v, str) else v)
    df.loc[RNG.random(n) < 0.01, "review_text"] = "Great"
    df.loc[RNG.random(n) < 0.015, "review_text"] = "Muy bueno, me encanta este producto"
    ing = RNG.random(n) < 0.15
    df.loc[ing, "ingredients"] = df.loc[ing, "ingredients"].map(lambda v: v.replace(", ", " ; ").title() + ".")
    df.loc[RNG.random(n) < 0.02, "ingredients"] = np.nan
    dup = df.sample(frac=0.03, random_state=1)                            # exact duplicates
    near = df.sample(frac=0.01, random_state=2).copy()                    # near duplicates (case/space)
    near["review_text"] = near.review_text.map(lambda v: v.upper() + " " if isinstance(v, str) else v)
    df = pd.concat([df, dup, near], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
    return df


# ---------------------------------------------------------------- 2. cleaning steps
def clean(df):
    d = df.copy()
    n0 = len(d)
    # S1 schema and types
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d["price_usd"] = pd.to_numeric(d["price_usd"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
    iso = pd.to_datetime(d["submission_time"], format="%Y-%m-%d", errors="coerce")
    alt = pd.to_datetime(d["submission_time"], format="%d/%m/%Y", errors="coerce")
    d["submission_time"] = iso.fillna(alt)
    log("S1 type conversion", n0, len(d), "ratings/prices to numeric, two date formats parsed, bad dates -> NaT")

    # S2 duplicates (exact, then normalised-text near duplicates)
    key = ["author_id", "product_id", "submission_time", "review_text"]
    d = d.drop_duplicates()
    n1 = len(d)
    d["_norm"] = d.review_text.fillna("").str.lower().str.strip()
    d = d.drop_duplicates(subset=["author_id", "product_id", "submission_time", "_norm"]).drop(columns="_norm")
    log("S2 deduplication", n0, len(d), f"exact: {n0 - n1}, near-duplicate: {n1 - len(d)}")

    # S3 validity rules
    n = len(d)
    bad_rating = ~d.rating.between(1, 5) & d.rating.notna()
    bad_price = d.price_usd <= 0
    bad_date = d.submission_time.isna() | (d.submission_time > pd.Timestamp("2026-06-30"))
    d.loc[bad_rating, "rating"] = np.nan
    d.loc[bad_price, "price_usd"] = np.nan
    d.loc[bad_date, "submission_time"] = pd.NaT
    d = d.dropna(subset=["rating"])
    d["review_year"] = d.submission_time.dt.year
    log("S3 validity rules", n, len(d), f"invalid ratings {int(bad_rating.sum())}, non-positive prices {int(bad_price.sum())}, invalid dates {int(bad_date.sum())}")

    # S4 category standardisation
    n = len(d)
    for c in ["brand_name", "skin_type", "skin_tone", "primary_category"]:
        d[c] = d[c].astype("string").str.strip().str.lower()
    log("S4 category standardisation", n, len(d), "trim + lower-case; variants of the same brand merged")

    # S5 ingredient parsing
    n = len(d)
    def parse(v):
        if not isinstance(v, str):
            return []
        toks = re.split(r"[;,]", v.lower().strip(" ."))
        return sorted({t.strip() for t in toks if t.strip()})
    d["ingredient_list"] = d.ingredients.map(parse)
    d["n_ingredients"] = d.ingredient_list.map(len)
    d.loc[d.n_ingredients == 0, "n_ingredients"] = np.nan
    log("S5 ingredient parsing", n, len(d), "split on , and ;, normalised case/punctuation, set of unique INCI-style tokens")

    # S6 text cleaning
    n = len(d)
    def clean_text(v):
        if not isinstance(v, str):
            return np.nan
        v = html.unescape(re.sub(r"<[^>]+>", " ", v))
        v = re.sub(r"[^\w\s.,!?'-]", " ", v)          # drop emoji/symbols
        v = re.sub(r"([!?.])\1+", r"\1", v)           # repeated punctuation
        return re.sub(r"\s+", " ", v).strip()
    d["review_clean"] = d.review_text.map(clean_text)
    d["review_len_words"] = d.review_clean.str.split().str.len()
    spanish = d.review_clean.fillna("").str.contains(r"\b(?:muy|bueno|este|producto|encanta)\b", case=False)
    d["is_english"] = ~spanish          # production version: use langdetect / fastText lid.176
    d["is_short"] = d.review_len_words < 3
    log("S6 text cleaning", n, len(d), f"flagged non-English {int(spanish.sum())}, flagged very short {int(d.is_short.sum())}")

    # S7 missing values: flag then impute (never impute the target)
    n = len(d)
    for c in ["skin_type", "skin_tone"]:
        d[f"{c}_missing"] = d[c].isna().astype(int)
        d[c] = d[c].fillna("unknown")
    d["is_recommended_missing"] = d.is_recommended.isna().astype(int)
    # price: median by category (robust), then flag
    d["price_missing"] = d.price_usd.isna().astype(int)
    d["price_usd"] = d.price_usd.fillna(d.groupby("primary_category")["price_usd"].transform("median"))
    log("S7 missing values", n, len(d), "'unknown' category + missing flags for attributes; category-median price; target never imputed")

    # S8 outliers (price: IQR on log scale per category; flagged and winsorised, not deleted)
    n = len(d)
    lp = np.log1p(d.price_usd)
    g = lp.groupby(d.primary_category)
    q1, q3 = g.transform(lambda x: x.quantile(.25)), g.transform(lambda x: x.quantile(.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    d["price_outlier"] = ((lp < lo) | (lp > hi)).astype(int)
    d["price_usd_w"] = np.expm1(lp.clip(lo, hi))
    log("S8 outlier handling", n, len(d), f"{int(d.price_outlier.sum())} price outliers flagged (log-IQR per category) and winsorised")

    # S9 target consistency: recommend flag vs rating contradictions
    n = len(d)
    d["rec_conflict"] = (((d.rating >= 4) & (d.is_recommended == 0)) | ((d.rating <= 2) & (d.is_recommended == 1))).astype(int)
    log("S9 consistency checks", n, len(d), f"{int(d.rec_conflict.sum())} rating/recommend contradictions flagged for sensitivity analysis")

    # S10 scaling and encoding features (fit later on train split only)
    d["log_price"] = np.log1p(d.price_usd_w)
    d["author_hash"] = d.author_id.map(lambda a: hashlib.sha256(("salt-" + a).encode()).hexdigest()[:12])  # pseudonymise
    d = d.drop(columns="author_id")
    return d


def profile(df, label):
    cols = ["skin_type", "skin_tone", "is_recommended", "review_text", "price_usd", "ingredients", "submission_time", "rating"]
    return {c: round(float(df[c].isna().mean() * 100), 1) for c in cols if c in df}


if __name__ == "__main__":
    raw = simulate_raw()
    raw.to_csv("raw_simulated.csv", index=False)
    before = profile(raw.assign(rating=pd.to_numeric(raw.rating, errors="coerce")), "raw")
    cleaned = clean(raw)
    after_cols = {c: round(float((cleaned[c].isna() | (cleaned[c] == "unknown")).mean() * 100), 1) for c in ["skin_type", "skin_tone"]}
    cleaned.drop(columns=["ingredient_list"]).to_csv("cleaned_simulated.csv", index=False)
    out = {"raw_rows": len(raw), "clean_rows": len(cleaned), "missing_before_pct": before, "after_unknown_pct": after_cols,
           "log": LOG,
           "dtypes_before": {c: str(t) for c, t in raw.dtypes.items()}}
    json.dump(out, open("pipeline_log.json", "w"), indent=1, default=str)
    print(json.dumps(out["log"], indent=1)); print(before); print(after_cols); print(len(raw), len(cleaned))
