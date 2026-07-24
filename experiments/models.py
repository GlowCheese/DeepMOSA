import ast
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    config_id: str
    argv: list[str]


class RunEntry(BaseModel):
    module_name: str
    run_config: RunConfig
    log_path: Path | None = None


class StatisticsRow(BaseModel):
    run_id: int = Field(alias="RunId")
    project_name: str = Field(alias="ProjectName")
    target_module: str = Field(alias="TargetModule")
    configuration_id: str = Field(alias="ConfigurationId")
    line_nos: int = Field(alias="LineNos")
    random_seed: int = Field(alias="RandomSeed")
    lines: int = Field(alias="Lines")
    predicates: int = Field(alias="Predicates")
    goals: int = Field(alias="Goals")
    mccabe_ast: list[int] = Field(alias="McCabeAST")
    mccabe_code_object: list[int] = Field(alias="McCabeCodeObject")
    code_objects: int = Field(alias="CodeObjects")
    accessible_objects_under_test: int = Field(alias="AccessibleObjectsUnderTest")
    llm_calls: int = Field(alias="LLMCalls")
    llm_query_time: float = Field(alias="LLMQueryTime")
    llm_stage_saved_tests: int = Field(alias="LLMStageSavedTests")
    llm_input_tokens: int = Field(alias="LLMInputTokens")
    llm_output_tokens: int = Field(alias="LLMOutputTokens")
    parsed_statements: int = Field(alias="ParsedStatements")
    parsable_statements: int = Field(alias="ParsableStatements")
    final_size: int = Field(alias="FinalSize")
    final_length: int = Field(alias="FinalLength")
    branch_coverage: float = Field(alias="BranchCoverage")
    coverage_timeline: list[float] = Field(alias="CoverageTimeline", exclude=True)

    @field_validator("mccabe_ast", "mccabe_code_object", mode="before")
    @classmethod
    def validate_mccabe(cls, v):
        return ast.literal_eval(v)

    @field_validator(
        "llm_calls",
        "llm_query_time",
        "llm_stage_saved_tests",
        "llm_input_tokens",
        "llm_output_tokens",
        "parsed_statements",
        "parsable_statements",
        mode="before",
    )
    @classmethod
    def empty_llm_stats(cls, v):
        if v == "":
            return 0
        return v
