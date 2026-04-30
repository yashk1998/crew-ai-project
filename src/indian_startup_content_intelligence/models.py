"""Pydantic schemas for structured task handoffs.

Every task uses one of these as `output_pydantic`. This forces the LLM into
structured-output mode and validates every handoff so a single bad response
can't corrupt the rest of the pipeline.

Schema-design note (2026-04-30): the previous InstagramBrief had 25+ fields
which routinely caused the LLM to truncate at 2-of-3 briefs and miss required
fields. Slimmed to ~14 fields with denser composite text fields where it makes
sense. 3 briefs now fit comfortably in a single generation.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------- Task 1: collect_indian_startup_content ----------

class RawItem(BaseModel):
    title: str
    brief_description: str
    url: str
    source: str
    published: str = Field(
        default="",
        description="ISO 8601 timestamp from the feed, empty if absent.",
    )


class RawItemBatch(BaseModel):
    items: list[RawItem] = Field(min_length=3, max_length=30)


# ---------- Task 2: classify_and_cluster ----------

Stage = Literal["seed", "series-a", "series-b+", "growth", "exits", "irrelevant"]
Pillar = Literal[
    "fundraising", "gtm", "hiring", "product", "founder_psych", "exits", "regulation"
]


class ClassifiedItem(BaseModel):
    title: str
    url: str
    stage: Stage
    pillar: Pillar
    india_specific: bool
    persona_fit_score: float = Field(ge=0.0, le=1.0)


class TopicCluster(BaseModel):
    canonical_topic: str
    items: list[ClassifiedItem] = Field(min_length=1, max_length=5)
    avg_persona_fit_score: float = Field(ge=0.0, le=1.0)


class TopicClusterBatch(BaseModel):
    clusters: list[TopicCluster]


# ---------- Task 3: validate_and_rank ----------

class RankedTopic(BaseModel):
    canonical_topic: str
    items: list[ClassifiedItem]
    relevance_score: float = Field(ge=1.0, le=10.0)
    actionability_score: float = Field(ge=1.0, le=10.0)
    stage_fit_score: float = Field(ge=1.0, le=10.0)
    quality_score: float = Field(ge=1.0, le=10.0)
    final_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Weighted: relevance*0.3 + actionability*0.25 + stage*0.25 + quality*0.2",
    )
    score_breakdown: str
    persona_fit: float = Field(ge=0.0, le=1.0)


class RankedTopicBatch(BaseModel):
    topics: list[RankedTopic] = Field(max_length=3)


# ---------- Task 4: choose_format ----------

InstagramFormat = Literal["reel", "carousel", "story", "static"]
PrimaryMetric = Literal["saves", "watch_time", "shares", "comments"]


class FormatRecommendation(BaseModel):
    canonical_topic: str
    format: InstagramFormat
    expected_primary_metric: PrimaryMetric
    rubric_match: str
    reasoning: str


class FormatRecommendationBatch(BaseModel):
    recommendations: list[FormatRecommendation]


# ---------- Task 5: generate_schema_compliant_instagram_briefs (slimmed) ----------

PrimaryGoal = Literal[
    "awareness", "engagement", "saves", "shares", "follows", "leads"
]


class SourceCitation(BaseModel):
    """A real upstream URL the brief draws from. URLs MUST come from the cluster."""

    title: str
    url: str
    source: str

    @field_validator("url")
    @classmethod
    def reject_placeholder_urls(cls, v: str) -> str:
        bad_hosts = ("example.com", "example.org", "example.net", "test.com")
        if any(host in v.lower() for host in bad_hosts):
            raise ValueError(
                f"URL {v!r} is a placeholder. Use a real URL from the upstream RawItem cluster."
            )
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"URL {v!r} must include a scheme (http:// or https://).")
        return v


class HookOption(BaseModel):
    label: Literal["A", "B", "C"]
    angle: str = Field(description="Psychological angle. e.g. 'specific number', 'contrarian', 'insider language'.")
    text: str = Field(description="The hook itself, ≤15 words.")
    why_it_works: str = Field(description="1 sentence on why it drives the format's primary metric.")


class SlideOrShot(BaseModel):
    """One slide of a carousel, or one frame of a story. Mix as appropriate."""

    sequence_number: int = Field(ge=1)
    label: str = Field(description='"Slide 1" / "Frame 1" / "Slide 5" — match the format.')
    visual_concept: str = Field(description="What's literally on screen. Composition, B-roll, key imagery.")
    headline: str = Field(description="Big on-screen text. Carousel: ≤8 words. Story sticker: ≤6 words.")
    voiceover: str = Field(
        description=(
            "STORY: required, conversational founder-to-founder voice, MUST DIFFER from headline. "
            "CAROUSEL: typically empty string."
        )
    )
    design_notes: str = Field(description="Hex colors, typography weight, layout grid, motion notes.")


class InstagramBrief(BaseModel):
    """A production-ready Instagram brief. ~14 fields, slim by design.

    Source URLs MUST come from the upstream RankedTopic.items cluster.
    """

    # ---- Strategic frame (compact) ----
    topic_line: str = Field(description="The topic restated in one tight line.")
    thesis: str = Field(description="Single-sentence reason this content exists. Sharp, specific, no fluff.")
    target_subaudience: str = Field(
        description="Specific founder subsegment. e.g. 'Pre-seed B2B SaaS founders negotiating their first cap table.'"
    )
    why_now: str = Field(description="What makes this timely THIS WEEK. Cite the source date if known.")

    # ---- Format ----
    format: InstagramFormat
    primary_goal: PrimaryGoal = Field(description="Always 'awareness' for this crew.")
    specs: str = Field(description="Pipe-separated. Carousel: '8-10 slides | 4:5 | Slide 1 hook | 5 hashtags'. Story: '4 frames × 15s | 9:16 | Frame 1 hook | 1-2 hashtags'.")

    # ---- Fact-check + sources ----
    fact_check: str = Field(description="One of 'Verified', 'Evergreen', 'Unverified', followed by ' — ' and a one-sentence note.")
    source_citations: list[SourceCitation] = Field(
        min_length=1,
        max_length=4,
        description="Real URLs from the upstream cluster. NEVER invent. NEVER use example.com.",
    )

    # ---- The brief proper ----
    hooks: list[HookOption] = Field(min_length=3, max_length=3)
    slides_or_shots: list[SlideOrShot] = Field(min_length=3, max_length=10)
    caption: str = Field(
        description="80-180 word primary caption. Line breaks for readability. First line survives the Read More cut."
    )
    hashtags: list[str] = Field(
        min_length=3,
        max_length=7,
        description="Without # prefix. Mix broad + niche.",
    )

    # ---- CTA + production ----
    primary_cta: str = Field(description="The CTA matched to the primary_goal.")
    comment_cta: str = Field(description="A variant designed to provoke comments (great for stories).")

    # ---- Distribution (combined) ----
    audio_recommendation: str = Field(
        description="STORY: real trending sound name + BPM + mood, OR 'Original audio with voiceover'. CAROUSEL: 'No audio'."
    )
    distribution_notes: str = Field(
        description=(
            "Combined posting + engagement strategy. Cover: best time IST, best day, "
            "first-60-min engagement plan, follow-up post idea. 3-6 lines."
        )
    )


class InstagramBriefBatch(BaseModel):
    briefs: list[InstagramBrief] = Field(min_length=3, max_length=3)
