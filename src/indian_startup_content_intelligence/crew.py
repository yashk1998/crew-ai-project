import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileReadTool, TavilySearchTool

from indian_startup_content_intelligence.models import (
    FormatRecommendationBatch,
    RankedTopicBatch,
    RawItemBatch,
    TopicClusterBatch,
)
from indian_startup_content_intelligence.tools.rss_collector import (
    RSSFeedCollectorTool,
)


# ---- Azure OpenAI env-var bridging ----
# LiteLLM expects AZURE_API_KEY / AZURE_API_BASE / AZURE_API_VERSION,
# but it delegates to the OpenAI Python SDK which looks for
# AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_VERSION.
# Mirror whichever set the user provided into both naming conventions so
# auth works no matter which path the SDK takes internally.
def _bridge_azure_env() -> None:
    pairs = [
        ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY"),
        ("AZURE_API_BASE", "AZURE_OPENAI_ENDPOINT"),
        ("AZURE_API_VERSION", "AZURE_OPENAI_API_VERSION"),
    ]
    for litellm_name, openai_name in pairs:
        litellm_val = os.getenv(litellm_name)
        openai_val = os.getenv(openai_name)
        if litellm_val and not openai_val:
            os.environ[openai_name] = litellm_val
        elif openai_val and not litellm_val:
            os.environ[litellm_name] = openai_val


_bridge_azure_env()


_AZURE_GPT4O = os.getenv("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")
_AZURE_GPT4O_MINI = os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")


def _make_llm(deployment: str, max_tokens: int | None = None) -> LLM:
    """Build an Azure OpenAI LLM via LiteLLM, passing creds explicitly.

    Passing api_key / api_base / api_version directly to the LLM constructor
    avoids relying on LiteLLM's env-var lookup (which has been inconsistent
    between LiteLLM versions and the underlying OpenAI SDK). _bridge_azure_env
    is still called at import time as a belt-and-suspenders measure.

    `max_tokens` is needed for the brief generator (large markdown output);
    when None, the provider default applies (typically ~4096 for Azure).
    """
    api_key = os.getenv("AZURE_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    api_base = os.getenv("AZURE_API_BASE") or os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = (
        os.getenv("AZURE_API_VERSION")
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or "2024-08-01-preview"
    )

    kwargs = {"model": f"azure/{deployment}", "is_litellm": True}
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base
    if api_version:
        kwargs["api_version"] = api_version
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return LLM(**kwargs)


@CrewBase
class IndianStartupContentIntelligenceCrew:
    """IndianStartupContentIntelligence crew."""

    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # ---------- Agents ----------

    @agent
    def source_scout_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["source_scout_agent"],  # type: ignore[index]
            tools=[
                RSSFeedCollectorTool(),
                TavilySearchTool(
                    topic="news",
                    time_range="week",
                    search_depth="basic",
                    max_results=8,
                    days=7,
                    include_answer=False,
                    include_raw_content=False,
                ),
            ],
            llm=_make_llm(_AZURE_GPT4O),
        )

    @agent
    def content_classifier_and_topic_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["content_classifier_and_topic_analyzer"],  # type: ignore[index]
            llm=_make_llm(_AZURE_GPT4O),
        )

    @agent
    def topic_ranking_and_selection_specialist(self) -> Agent:
        return Agent(
            config=self.agents_config["topic_ranking_and_selection_specialist"],  # type: ignore[index]
            tools=[FileReadTool()],
            llm=_make_llm(_AZURE_GPT4O),
        )

    @agent
    def instagram_content_format_specialist_for_b2b_founder_audiences(self) -> Agent:
        return Agent(
            config=self.agents_config["instagram_content_format_specialist_for_b2b_founder_audiences"],  # type: ignore[index]
            llm=_make_llm(_AZURE_GPT4O),
        )

    @agent
    def senior_creative_director_for_instagram_b2b_content___schema_compliant(self) -> Agent:
        return Agent(
            config=self.agents_config["senior_creative_director_for_instagram_b2b_content___schema_compliant"],  # type: ignore[index]
            # Each brief task now generates ONE brief, so 8000 max_tokens is
            # plenty (~3000 tokens of markdown + buffer). Splitting the work
            # into 3 LLM calls eliminates the truncation risk that hit when a
            # single call had to produce all 3 briefs.
            llm=_make_llm(_AZURE_GPT4O, max_tokens=8000),
            max_retry_limit=1,
            max_iter=6,
            # Per-brief wall-clock cap — fails fast rather than burning tokens.
            max_execution_time=180,  # 3 min per brief
        )

    @agent
    def markdown_assembly_specialist(self) -> Agent:
        # Lightweight pass-through agent — concatenates the 3 brief outputs
        # into one markdown document. No content authorship, just stitching.
        # gpt-4o-mini is plenty for this; it's fast and cheap.
        return Agent(
            config=self.agents_config["markdown_assembly_specialist"],  # type: ignore[index]
            llm=_make_llm(_AZURE_GPT4O_MINI, max_tokens=12000),
            max_retry_limit=1,
            max_iter=3,
            max_execution_time=120,
        )

    # ---------- Tasks ----------

    @task
    def collect_indian_startup_content(self) -> Task:
        return Task(
            config=self.tasks_config["collect_indian_startup_content"],  # type: ignore[index]
            output_pydantic=RawItemBatch,
        )

    @task
    def classify_and_cluster(self) -> Task:
        return Task(
            config=self.tasks_config["classify_and_cluster"],  # type: ignore[index]
            output_pydantic=TopicClusterBatch,
        )

    @task
    def validate_and_rank(self) -> Task:
        return Task(
            config=self.tasks_config["validate_and_rank"],  # type: ignore[index]
            output_pydantic=RankedTopicBatch,
        )

    @task
    def choose_format(self) -> Task:
        return Task(
            config=self.tasks_config["choose_format"],  # type: ignore[index]
            output_pydantic=FormatRecommendationBatch,
        )

    # Brief generation is split into 3 separate LLM calls (one per brief)
    # so no single call has to fit all 3 elaborate briefs in its output
    # window. The final assemble_final_briefs task concatenates them into
    # one markdown document — the FINAL deliverable shown on the dashboard.

    @task
    def generate_carousel_brief(self) -> Task:
        return Task(
            config=self.tasks_config["generate_carousel_brief"],  # type: ignore[index]
        )

    @task
    def generate_reel_part1_brief(self) -> Task:
        return Task(
            config=self.tasks_config["generate_reel_part1_brief"],  # type: ignore[index]
        )

    @task
    def generate_reel_part2_brief(self) -> Task:
        return Task(
            config=self.tasks_config["generate_reel_part2_brief"],  # type: ignore[index]
        )

    @task
    def assemble_final_briefs(self) -> Task:
        return Task(
            config=self.tasks_config["assemble_final_briefs"],  # type: ignore[index]
        )

    # ---------- Crew ----------

    @crew
    def crew(self) -> Crew:
        """Creates the IndianStartupContentIntelligence crew."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            chat_llm=_make_llm(_AZURE_GPT4O_MINI),
        )
