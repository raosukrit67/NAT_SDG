# flake8: noqa: E501, BLE001
# pylint: disable=C0301
import os
import json
import logging
from typing import Dict, Any, List

from pydantic import Field
from pydantic import BaseModel

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class NDDClientConnectionConfig(BaseModel):
    """Configuration for NeMo Data Designer client connection."""
    ndd_client_url: str = Field(default="http://localhost:8000", description="URL of the NeMo Data Designer container")
    ndd_client_timeout: int = Field(default=600, description="Timeout for the NeMo Data Designer client")
    ndd_datastore_url: str = Field(default="http://localhost:3000", description="URL of the NMS Datastore container")


class NDDWorkflowConfig(FunctionBaseConfig, name="ndd_workflow"):
    """Configuration for the NeMo Data Designer workflow."""

    ndd_client_cfg: NDDClientConnectionConfig = Field(
        default_factory=NDDClientConnectionConfig,
        description=("Configuration dict for the NeMo Data Designer client.")
    )

    ndd_model_cfg: List[Dict[str, str]] = Field(
        default_factory=list,
        description=("List of model assignments. Each item in the list is a dict"
                     "each dict mapping model_alias -> llm_name"
                     "e.g., [{'candidate_llm': 'openai_llm'}]")
    )

    output_dir: str = Field(description="Directory to save the generated synthetic data.")


@register_function(config_type=NDDWorkflowConfig)
async def ndd_client_function(config: NDDWorkflowConfig, builder: Builder):
    """NeMo Data Designer client for synthetic data generation."""

    import uuid
    from datetime import datetime
    from datetime import timezone

    from nemo_microservices import NeMoMicroservices
    from nemo_microservices.beta.data_designer import (
        DataDesignerConfigBuilder,
        DataDesignerClient
    )

    from nemo_microservices.beta.data_designer.config import columns as C
    from nemo_microservices.beta.data_designer.config import params as P

    from sdg_workflow.data_models import AgentToolDetails
    from sdg_workflow.data_models import UserQuery
    from sdg_workflow.data_models import ScenarioDescription
    from sdg_workflow.data_models import AgentTrajectory

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    # Initialize NDD client and parse model configs
    ndd = DataDesignerClient(client=NeMoMicroservices(
        base_url=config.ndd_client_cfg.ndd_client_url,
        timeout=config.ndd_client_cfg.ndd_client_timeout
    ))

    async def parse_model_config(nat_llm_cfg: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Parse the NAT LLM config and create NDD model cfg"""
        ndd_model_configs = []

        for single_model_cfg in nat_llm_cfg:
            # Extract model alias and llm name from the dict
            model_alias = list(single_model_cfg.keys())[0]
            llm_name = single_model_cfg[model_alias]

            # Get the LLM config from NAT using the builder
            try:
                llm_config = builder.get_llm_config(llm_name)
                if not llm_config:
                    logger.warning("LLM '%s' not found in NAT config", llm_name)
                    continue

                # Get inference parameters
                inference_params = {
                    "temperature": getattr(llm_config, 'temperature', 0.7),
                    "max_tokens": getattr(llm_config, 'max_tokens', 1024),
                    "top_p": getattr(llm_config, 'top_p', 0.95),
                }

                # Get connection parameters
                model_id = getattr(llm_config, 'model_name', llm_name)
                base_url = getattr(llm_config, 'base_url', None)

                logger.info("Configuring NDD model: alias=%s, llm=%s, model_id=%s, base_url=%s, params=%s",
                           model_alias, llm_name, model_id, base_url, inference_params)

                # Create a NeMo Data Designer model config
                ndd_model_configs.append(
                    P.ModelConfig(
                        alias=model_alias,
                        inference_parameters=P.InferenceParameters(
                            **inference_params
                        ),
                        model=P.Model(
                            api_endpoint=P.ApiEndpoint(
                                model_id=model_id,
                                url=base_url,
                            ),
                            model_id=model_id
                        )
                    )
                )

            except Exception as e:
                logger.error("Failed to configure model '%s' with LLM '%s': %s", model_alias, llm_name, e)
                continue

        return ndd_model_configs

    async def get_tool_names(agent_tool_details: AgentToolDetails) -> List[str]:
        """Get the tool names from the agent tool details"""
        tool_names = [tool.name for tool in agent_tool_details.tools]

        tool_names.append("None")
        return tool_names

    async def convert_to_nat_eval_format(
        dataset_records: List[Dict[str, Any]],
        agent_tool_details: AgentToolDetails,
        output_dir: str,
        data_id: str,
        current_date: str
    ) -> str:
        """Convert NDD dataset to NAT TrajectoryEvaluator format."""

        # Convert to TrajectoryEvaluator format
        eval_items = []

        def convert_ndd_step_to_trajectory_step(step_data, step_index):
            """Convert NDD step to IntermediateStep format"""
            import time
            intermediate_steps = []

            if step_data.get("tool_calls"):
                for tool_idx, tool_call in enumerate(step_data["tool_calls"]):
                    tool_name = tool_call.get("name", "unknown_tool")
                    tool_args = tool_call.get("arguments", [])

                    # Convert arguments
                    tool_input = {}
                    for arg in tool_args:
                        if isinstance(arg, dict) and "name" in arg and "value" in arg:
                            tool_input[arg["name"]] = arg["value"]

                    # TOOL_START event
                    tool_start = {
                        "parent_id": "root",
                        "function_ancestry": {
                            "function_id": f"tool_{tool_name}_{step_index}_{tool_idx}",
                            "function_name": tool_name,
                            "parent_id": "root",
                            "parent_name": None
                        },
                        "payload": {
                            "event_type": "TOOL_START",
                            "event_timestamp": time.time() + step_index * 0.1 + tool_idx * 0.01,
                            "name": tool_name,
                            "data": {"input": tool_input, "output": None}
                        }
                    }

                    # TOOL_END event
                    tool_end = {
                        "parent_id": "root",
                        "function_ancestry": {
                            "function_id": f"tool_{tool_name}_{step_index}_{tool_idx}",
                            "function_name": tool_name,
                            "parent_id": "root",
                            "parent_name": None
                        },
                        "payload": {
                            "event_type": "TOOL_END",
                            "event_timestamp": time.time() + step_index * 0.1 + tool_idx * 0.01 + 0.005,
                            "name": tool_name,
                            "data": {
                                "input": None,
                                "output": {"status": "success", "result": f"Executed {tool_name}", "data": tool_input}
                            }
                        }
                    }

                    intermediate_steps.extend([tool_start, tool_end])

            if step_data.get("content"):
                # LLM_END event
                llm_response = {
                    "parent_id": "root",
                    "function_ancestry": {
                        "function_id": f"llm_response_{step_index}",
                        "function_name": "llm_response",
                        "parent_id": "root",
                        "parent_name": None
                    },
                    "payload": {
                        "event_type": "LLM_END",
                        "event_timestamp": time.time() + step_index * 0.1 + 0.05,
                        "name": "llm_response",
                        "data": {"input": None, "output": {"content": step_data["content"]}}
                    }
                }
                intermediate_steps.append(llm_response)

            return intermediate_steps

        # Process each record
        for i, record in enumerate(dataset_records):
            try:
                # Extract refined user query (final version after feedback)
                user_query_raw = record.get("refined_user_query", {})
                if isinstance(user_query_raw, str):
                    user_query = json.loads(user_query_raw)
                else:
                    user_query = user_query_raw
                question = user_query.get("user_query", "") if isinstance(user_query, dict) else str(user_query)

                # Extract refined agent trajectory (final version after feedback)
                trajectory_raw = record.get("refined_agent_trajectory", {})
                if isinstance(trajectory_raw, str):
                    trajectory_data = json.loads(trajectory_raw)
                else:
                    trajectory_data = trajectory_raw
                steps = trajectory_data.get("steps", []) if isinstance(trajectory_data, dict) else []

                # Convert steps
                expected_trajectory = []
                for step_idx, step in enumerate(steps):
                    intermediate_steps = convert_ndd_step_to_trajectory_step(step, step_idx)
                    expected_trajectory.extend(intermediate_steps)

                # Extract expected answer from refined trajectory
                final_answer = ""
                for step in reversed(steps):
                    if step.get("content"):
                        final_answer = step["content"]
                        break

                # Create EvalInputItem
                eval_item = {
                    "id": record.get("id", f"ndd_{uuid.uuid4().hex[:8]}"),
                    "input_obj": question,
                    "expected_output_obj": final_answer,
                    "output_obj": None,  # To be filled by real NAT agent
                    "expected_trajectory": expected_trajectory,  # Ground truth from refined NDD trajectory
                    "trajectory": [],  # To be filled by real NAT agent
                    "full_dataset_entry": record
                }

                eval_items.append(eval_item)
                logger.info("Converted record %d: %d trajectory steps", i+1, len(expected_trajectory))

            except Exception as e:
                logger.error("Error converting record %d: %s", i+1, e)

        # Save as JSONL format (one JSON object per line) for NAT evaluation
        ground_truth_path = os.path.join(output_dir, f"nat_eval_dataset_{data_id}.jsonl")
        with open(ground_truth_path, 'w', encoding='utf-8') as f:
            for eval_item in eval_items:
                nat_record = {
                    "id": eval_item["id"],
                    "question": eval_item["input_obj"],
                    "answer": eval_item["expected_output_obj"],
                    "expected_intermediate_steps": eval_item["expected_trajectory"]
                }
                f.write(json.dumps(nat_record, default=str) + '\n')

        logger.info("✅ TrajectoryEvaluator ground truth saved: %s", ground_truth_path)
        logger.info("📊 Converted %d cases ready for evaluation", len(eval_items))
        logger.info("🎯 Using refined user queries and trajectories as ground truth")

        return ground_truth_path

    async def generate_synthetic_data(agent_tool_details: AgentToolDetails) -> str:
        """Generate synthetic data using NeMo Data Designer patterns."""
        try:
            # Parse and configure models from NAT LLM configs
            ndd_model_configs = await parse_model_config(config.ndd_model_cfg)
            logger.info("Configured %d models for NDD workflow", len(ndd_model_configs))

            config_builder = DataDesignerConfigBuilder(model_configs=ndd_model_configs)

            # get the names of the tools that are available to the agent
            tool_names = await get_tool_names(agent_tool_details)

            #########################################################
            # COLUMN 1: TOOL CALLED; SAMPLED COLUMN
            #########################################################
            # add a column for the name of the tool that will be called in each scenario
            # we also add an emoty list so that we can generate examples where no tool is called
            config_builder.add_column(
                C.SamplerColumn(
                    name="tool_name",
                    type=P.SamplerType.CATEGORY,
                    params=P.CategorySamplerParams(values=tool_names),
                    description=("The tool that will be called in the given scenario."
                                 "Empty list means no tool is called."),
                ))

            #########################################################
            # COLUMN 2: SCENARIO DIFFICULTY; SAMPLED COLUMN
            #########################################################
            # add a column for the user query that will be called in each scenario
            config_builder.add_column(
                C.SamplerColumn(
                    name="scenario_difficulty",
                    type=P.SamplerType.CATEGORY,
                    params=P.CategorySamplerParams(values=["easy", "medium", "hard"]),
                    description=("The difficulty of the generated scenario."),
                ))

            #########################################################
            # COLUMN 3: SCENARIO DESCRIPTION; LLM GENERATED COLUMN
            #########################################################
            # add a column for the scenario description that will be generated by the LLM
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="scenario_description",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at generating DIVERSE and realistic evaluation scenarios for AI agents. "
                        "Your task is to create varied scenario descriptions that test different aspects of the agent's capabilities. "
                        "CRITICAL: Avoid repetitive patterns - create scenarios with different user personas, contexts, "
                        "complexity levels, edge cases, and problem types. Each scenario should feel distinctly different "
                        "from previous ones while remaining relevant to the agent's domain."
                    ),
                    prompt=(
                        "Create a UNIQUE and realistic scenario for evaluating an agent:\n\n"
                        f"**Agent Workflow**: {agent_tool_details.workflow_description}\n"
                        f"**Available Tools**: {[tool.name + ' - ' + tool.description for tool in agent_tool_details.tools]}\n\n"
                        "**Scenario Requirements:**\n"
                        "- Target tool: {{ tool_name }}\n"
                        "- Difficulty: {{ scenario_difficulty }}\n\n"
                        "**DIVERSITY REQUIREMENTS - Choose ONE approach:**\n"
                        "A) **Different User Personas**: novice vs expert, urgent vs exploratory, business vs personal\n"
                        "B) **Different Contexts**: time-sensitive, resource-constrained, collaborative, high-stakes\n"
                        "C) **Different Problem Types**: troubleshooting, research, creative, analytical, procedural\n"
                        "D) **Different Complexity**: multi-step processes, edge cases, incomplete information\n"
                        "E) **Different Industries/Domains**: healthcare, finance, education, entertainment, logistics\n\n"
                        "**AVOID**: Generic scenarios, 'user wants to...', repetitive setups\n"
                        "**CREATE**: Specific, contextual scenarios with clear motivations and constraints\n\n"
                        "Generate a scenario that:\n"
                        "1. Uses a DISTINCT approach from the list above\n"
                        "2. Includes specific context and user motivation\n"
                        "3. Requires the specified tool (if not 'None') naturally\n"
                        "4. Matches the difficulty through scenario complexity, not just tool usage\n"
                        "5. Feels realistic and specific to the domain\n\n"
                        "If tool_name is 'None', create scenarios where base knowledge suffices (trivia, definitions, general advice)."
                    ),
                    output_format=ScenarioDescription,
                    ))

            #########################################################
            # COLUMN 4: USER QUERY; LLM GENERATED COLUMN
            #########################################################
            # add a column for the user query that will be generated by the LLM
            config_builder.add_column(
                        C.LLMStructuredColumn(
            name="draft_user_query",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at generating DIVERSE, realistic user queries for AI agent evaluation. "
                        "Your task is to create varied user queries that reflect different communication styles, "
                        "experience levels, and urgency. CRITICAL: Avoid repetitive query patterns - vary the "
                        "formality, specificity, length, and approach. Real users communicate very differently "
                        "depending on their background, mood, and situation."
                    ),
                    prompt=(
                        "Generate a DIVERSE user query matching this scenario:\n\n"
                        f"**Agent Domain**: {agent_tool_details.workflow_description}\n"
                        f"**Available Tools**: {[tool.name + ' - ' + tool.description for tool in agent_tool_details.tools]}\n\n"
                        "**Scenario**: {{ scenario_description }}\n"
                        "**Target Tool**: {{ tool_name }}\n"
                        "**Difficulty**: {{ scenario_difficulty }}\n\n"
                        "**COMMUNICATION STYLE VARIETY - Choose ONE:**\n"
                        "A) **Casual/Informal**: \"hey\", \"can you help me\", contractions, informal language\n"
                        "B) **Professional/Formal**: complete sentences, technical terms, business language\n"
                        "C) **Urgent/Stressed**: \"need this ASAP\", \"urgent\", \"deadline\", fragmented thoughts\n"
                        "D) **Novice/Uncertain**: \"I think\", \"maybe\", \"not sure\", asks for guidance\n"
                        "E) **Expert/Specific**: technical jargon, specific requirements, assumes domain knowledge\n"
                        "F) **Conversational**: \"So I'm trying to...\", story-like, provides context\n"
                        "G) **Direct/Minimal**: short, to-the-point, \"Do X\", minimal explanation\n\n"
                        "**QUERY VARIATIONS:**\n"
                        "- Length: 5-50 words (vary dramatically)\n"
                        "- Clarity: crystal clear to somewhat ambiguous\n"
                        "- Context: minimal to extensive background\n"
                        "- Specificity: general request to precise requirements\n\n"
                        "**AVOID**: Starting with 'Can you', 'Please', 'I need' every time\n"
                        "**CREATE**: Authentic human communication with personality and context\n\n"
                        "Generate a query that fits the scenario while using a DISTINCT communication style."
                    ),
                    output_format=UserQuery,
                    )
                )

            #########################################################
            # COLUMN 5: AGENT TRAJECTORY; LLM GENERATED COLUMN
            #########################################################
            # add a column for the complete agent trajectory that will be generated by the LLM
            config_builder.add_column(
                        C.LLMStructuredColumn(
            name="draft_agent_trajectory",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at generating DIVERSE, realistic agent trajectories for evaluation. "
                        "Your task is to create varied problem-solving approaches that reflect different agent "
                        "behaviors, decision patterns, and execution styles. CRITICAL: Avoid repetitive trajectory "
                        "patterns - vary the approach, tool usage patterns, error handling, and response styles. "
                        "Real agents exhibit different behaviors based on confidence, available information, and context."
                    ),
                    prompt=(
                        "Generate a complete agent trajectory to solve this user query:\n\n"

                        "**USER QUERY**: {{ draft_user_query }}\n"
                        "**SCENARIO**: {{ scenario_description }}\n"
                        "**DIFFICULTY**: {{ scenario_difficulty }}\n\n"

                        f"**AVAILABLE TOOLS**:\n"
                        f"{chr(10).join([f'- {tool.name}: {tool.description}' for tool in agent_tool_details.tools])}\n\n"

                        "**DIVERSE TRAJECTORY APPROACHES - Choose ONE:**\n\n"

                        "A) **Confident/Direct**: Agent knows exactly what to do, executes efficiently\n"
                        "B) **Exploratory**: Agent tries multiple approaches, learns as it goes\n"
                        "C) **Cautious/Methodical**: Agent validates each step, asks clarifying questions\n"
                        "D) **Efficient/Minimal**: Agent takes shortest path, minimal tool usage\n"
                        "E) **Thorough/Comprehensive**: Agent gathers extensive information before responding\n"
                        "F) **Problem-solving**: Agent encounters issues and adapts strategy\n"
                        "G) **Collaborative**: Agent asks user for input or clarification during process\n\n"

                        "**EXECUTION VARIETY**:\n"
                        "- Tool usage: single tool vs multiple tools vs no tools\n"
                        "- Response length: brief vs detailed vs conversational\n"
                        "- Certainty level: confident vs uncertain vs seeking validation\n"
                        "- Error handling: smooth execution vs encountering/recovering from issues\n"
                        "- Information gathering: minimal vs comprehensive research\n\n"

                        "**TRAJECTORY RULES**:\n"
                        "1. **Choose a DISTINCT approach** from the list above\n"
                        "2. **Vary execution style** - avoid repetitive patterns\n"
                        "3. **Match agent persona** to the chosen approach\n"
                        "4. **Be realistic** - include appropriate struggles or smooth execution\n"
                        "5. **End appropriately** - complete task, ask for clarification, or admit limitations\n\n"

                        "**STEP TYPES & EXAMPLES**:\n\n"

                        "**Tool Call Step Example** (for wikipedia_search):\n"
                        "```json\n"
                        "{\n"
                        "  \"content\": null,\n"
                        "  \"tool_calls\": [\n"
                        "    {\n"
                        "      \"type\": \"function_call\",\n"
                        "      \"name\": \"wikipedia_search\",\n"
                        "      \"arguments\": [\n"
                        "        {\"name\": \"query\", \"value\": \"Great Fire of London history\"}\n"
                        "      ]\n"
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        "```\n\n"

                        "**Tool Call Step Example** (for webpage_query):\n"
                        "```json\n"
                        "{\n"
                        "  \"content\": null,\n"
                        "  \"tool_calls\": [\n"
                        "    {\n"
                        "      \"type\": \"function_call\",\n"
                        "      \"name\": \"webpage_query\",\n"
                        "      \"arguments\": [\n"
                        "        {\"name\": \"query\", \"value\": \"LangSmith pricing features\"}\n"
                        "      ]\n"
                        "    }\n"
                        "  ]\n"
                        "}\n"
                        "```\n\n"

                        "**Text Response Step Example**:\n"
                        "```json\n"
                        "{\n"
                        "  \"content\": \"Based on my search results, the Great Fire of London started on September 2, 1666, and was caused by a fire in a bakery on Pudding Lane.\",\n"
                        "  \"tool_calls\": null\n"
                        "}\n"
                        "```\n\n"

                        "**CRITICAL**: Always populate arguments with realistic values. \n\n"

                        "**TRAJECTORY PATTERNS (vary these)**:\n"
                        "- Direct: [tool_call] → [text_response with answer]\n"
                        "- Multi-step: [tool_call] → [tool_call] → [comprehensive text_response]\n"
                        "- Clarification: [text_response asking for more details]\n"
                        "- Research-heavy: [tool_call] → [tool_call] → [tool_call] → [synthesized response]\n"
                        "- Partial success: [tool_call] → [text_response with limitations/caveats]\n"
                        "- No-tool: [text_response based on knowledge] (if tool_name is 'None')\n\n"

                        "**FINAL INSTRUCTION**: Generate a UNIQUE trajectory that:\n"
                        "- Uses a distinct approach and execution style\n"
                        "- Follows the JSON structure exactly\n"
                        "- Feels different from typical AI agent responses\n"
                        "- Matches the scenario's specific context and user query style"
                    ),
                    output_format=AgentTrajectory,
                    )
                )

            #########################################################
            # LLM AS A JUDGE: SCORING RUBRICS
            #########################################################

            # Rubric 1: Scenario Realism and Relevance
            scenario_realism_rubric = P.Rubric(
                name="scenario_realism",
                description=(
                    f"CRITICALLY evaluate whether the scenario is realistic for {agent_tool_details.workflow_description}. "
                    f"Be skeptical - most AI-generated scenarios have flaws. Look for: overly complex setups, "
                    f"unrealistic user behaviors, artificial constraints, or scenarios that feel like textbook examples "
                    f"rather than real-world situations."
                ),
                scoring={
                    "4": "Exceptional realism - scenario feels completely natural and represents a common real-world use case (RARE)",
                    "3": "Good realism - scenario is plausible but may have 1-2 minor artificial elements",
                    "2": "Moderate realism - scenario is believable but feels somewhat constructed or has several unnatural aspects",
                    "1": "Poor realism - scenario is technically possible but highly contrived or unlikely in practice",
                    "0": "Unrealistic - scenario is artificial, overly complex, or impossible for this domain"
                })

                        # Rubric 2: Query-Scenario Alignment
            query_alignment_rubric = P.Rubric(
                name="query_alignment",
                description=(
                    "CRITICALLY assess whether the user query fits the scenario. Be harsh - look for: "
                    "queries that are too perfect/polished (real users are messy), queries that don't match "
                    "the scenario context, unnatural language, or queries that sound like they were written "
                    "to showcase the agent rather than solve a real problem."
                ),
                scoring={
                    "4": "Exceptional alignment - query sounds authentically human and perfectly matches scenario context (VERY RARE)",
                    "3": "Good alignment - query fits scenario well but may be slightly too polished or have minor gaps",
                    "2": "Moderate alignment - query generally matches scenario but feels artificial or has noticeable disconnects",
                    "1": "Poor alignment - query weakly connects to scenario or sounds unnatural for the context",
                    "0": "No alignment - query is completely mismatched to scenario or obviously artificial"
                })

            # Rubric 3: Ground Truth Trajectory Quality
            trajectory_quality_rubric = P.Rubric(
                name="trajectory_quality",
                description=(
                    "RIGOROUSLY evaluate the trajectory quality. Most AI-generated trajectories have flaws - "
                    "look for: unrealistic tool usage patterns, missing error handling, overly smooth execution "
                    "(real agents struggle), inappropriate tool choices, illogical step ordering, missing intermediate "
                    "reasoning, or trajectories that ignore real-world constraints and edge cases."
                ),
                scoring={
                    "4": "Exceptional trajectory - demonstrates nuanced understanding, handles edge cases, realistic agent behavior (EXTREMELY RARE)",
                    "3": "Good trajectory - logical flow with mostly realistic steps but may miss some real-world complexities",
                    "2": "Adequate trajectory - acceptable logic but somewhat idealized or missing important considerations",
                    "1": "Poor trajectory - has logical flaws, unrealistic assumptions, or inappropriate tool usage patterns",
                    "0": "Invalid trajectory - fundamentally flawed, illogical, or completely unsuitable as ground truth"
                })

            # Rubric 4: Difficulty Calibration
            difficulty_calibration_rubric = P.Rubric(
                name="difficulty_calibration",
                description=(
                    "STRICTLY evaluate difficulty calibration. Most AI systems struggle with this - look for: "
                    "'easy' scenarios that are actually complex, 'hard' scenarios that are trivial, mismatched "
                    "complexity between scenario/query/trajectory, or difficulty labels that don't reflect actual "
                    "cognitive load or skill requirements."
                ),
                scoring={
                    "4": "Perfect calibration - difficulty precisely matches all components and feels authentic (ALMOST NEVER occurs)",
                    "3": "Good calibration - difficulty generally appropriate but may have 1 component slightly off",
                    "2": "Moderate calibration - difficulty roughly matches but has noticeable misalignments across components",
                    "1": "Poor calibration - difficulty significantly mismatches scenario/query/trajectory complexity",
                    "0": "No calibration - difficulty label is completely wrong or nonsensical for the content"
                })

            config_builder.add_column(
                C.LLMJudgeColumn(
                    name="quality_assessment",
                    model_alias="judge_llm",
                    prompt=(
                        f"You are a HIGHLY CRITICAL evaluator assessing synthetic ground truth data for {agent_tool_details.workflow_description}. "
                        f"Your job is to be HARSH and SKEPTICAL - most AI-generated content has significant flaws.\n\n"

                        f"**CRITICAL EVALUATION MINDSET:**\n"
                        f"- ASSUME the data has problems until proven otherwise\n"
                        f"- SCORE 4 should be EXTREMELY RARE (less than 5% of cases)\n"
                        f"- LOOK FOR specific flaws and inconsistencies\n"
                        f"- BE HARSH - this data will be used to evaluate real agents\n\n"

                        f"**Domain Context:**\n"
                        f"- Target Agent Domain: {agent_tool_details.workflow_description}\n"
                        f"- Available Agent Tools: {[tool.name + ' (' + tool.description + ')' for tool in agent_tool_details.tools]}\n\n"

                        "**Data to Evaluate:**\n"
                        "- **Scenario:** {{ scenario_description }}\n"
                        "- **User Query:** {{ draft_user_query }}\n"
                        "- **Expected Agent Trajectory:** {{ draft_agent_trajectory }}\n"
                        "- **Claimed Difficulty:** {{ scenario_difficulty }}\n\n"

                        f"**RED FLAGS TO LOOK FOR:**\n"
                        f"- Scenarios that feel like textbook examples rather than messy real-world situations\n"
                        f"- User queries that are too polished or perfect (real users are messy and unclear)\n"
                        f"- Trajectories that execute too smoothly without realistic struggles or edge cases\n"
                        f"- Difficulty levels that don't match actual complexity\n"
                        f"- Generic or template-like content that could apply to any domain\n"
                        f"- Missing realistic constraints, errors, or complications\n\n"

                        f"**SCORING GUIDELINES:**\n"
                        f"- Score 4 ONLY if content is exceptionally realistic and shows deep domain understanding\n"
                        f"- Score 3 for good content that has minor but noticeable flaws\n"
                        f"- Score 2 for adequate content that feels somewhat artificial or generic\n"
                        f"- Score 1 for poor content with significant unrealistic elements\n"
                        f"- Score 0 for completely inappropriate or nonsensical content\n\n"

                        f"For each dimension, identify SPECIFIC issues and explain why the content falls short of perfection. "
                        f"Provide concrete examples of problems you observe."
                    ),
                    rubrics=[scenario_realism_rubric, query_alignment_rubric, trajectory_quality_rubric, difficulty_calibration_rubric],
                ))

            #########################################################
            # COLUMN 7: REFINED USER QUERY; LLM GENERATED COLUMN
            #########################################################
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="refined_user_query",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at refining user queries based on feedback. "
                        "Your task is to improve the draft user query based on the quality assessment feedback "
                        "to create a more realistic, context-appropriate query that better fits the scenario."
                    ),
                    prompt=(
                        "Refine this user query based on the quality assessment feedback:\n\n"

                        "**ORIGINAL SCENARIO**: {{ scenario_description }}\n"
                        "**DRAFT USER QUERY**: {{ draft_user_query }}\n"
                        "**QUALITY FEEDBACK**: {{ quality_assessment }}\n\n"

                        f"**Agent Context**: {agent_tool_details.workflow_description}\n"
                        f"**Available Tools**: {[tool.name + ': ' + tool.description for tool in agent_tool_details.tools]}\n\n"

                        "**Instructions**:\n"
                        "1. Address any issues identified in the quality feedback\n"
                        "2. Ensure the query is natural and realistic for this domain\n"
                        "3. Make sure it appropriately uses the agent's available tools\n"
                        "4. Keep the core intent but improve clarity and realism\n"
                        "5. If the original query is already high quality, you may keep it unchanged\n\n"

                        "Generate an improved user query that addresses the feedback while maintaining natural language."
                    ),
                    output_format=UserQuery,
                ))

            #########################################################
            # COLUMN 8: REFINED AGENT TRAJECTORY; LLM GENERATED COLUMN
            #########################################################
            config_builder.add_column(
                C.LLMStructuredColumn(
                    name="refined_agent_trajectory",
                    model_alias="candidate_llm",
                    system_prompt=(
                        "You are an expert at refining agent trajectories based on feedback. "
                        "Your task is to improve the draft agent trajectory based on the quality assessment feedback "
                        "to create a more realistic, high-quality trajectory that demonstrates expert agent behavior."
                    ),
                    prompt=(
                        "Refine this agent trajectory based on the quality assessment feedback:\n\n"

                        "**SCENARIO**: {{ scenario_description }}\n"
                        "**REFINED USER QUERY**: {{ refined_user_query }}\n"
                        "**DRAFT TRAJECTORY**: {{ draft_agent_trajectory }}\n"
                        "**QUALITY FEEDBACK**: {{ quality_assessment }}\n\n"

                        f"**Available Tools**:\n"
                        f"{chr(10).join([f'- {tool.name}: {tool.description}' for tool in agent_tool_details.tools])}\n\n"

                        "**Instructions**:\n"
                        "1. Address any issues identified in the quality feedback\n"
                        "2. Ensure the trajectory is realistic and demonstrates expert agent behavior\n"
                        "3. Use tools appropriately with realistic arguments\n"
                        "4. Maintain logical flow between steps\n"
                        "5. Ensure proper JSON structure with name-value argument pairs\n"
                        "6. If the original trajectory is already high quality, you may keep it unchanged\n\n"

                        "**Argument Structure**: Use [{\"name\": \"arg_name\", \"value\": \"arg_value\"}] format\n\n"

                        "Generate an improved agent trajectory that addresses the feedback."
                    ),
                    output_format=AgentTrajectory,
                ))

            # validate the various ops and columns that are to be generated
            config_builder.validate()

            # generate a preview of the synthetic data that will be generated
            preview = ndd.preview(config_builder, verbose_logging=True)

            preview.display_sample_record(0)
            preview.display_sample_record(1)

            # create uuid for the data
            data_id = str(uuid.uuid4())

            # get current time in UTC
            current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            output_dir = os.path.join(config.output_dir, current_date)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            save_path = os.path.join(output_dir, f"ndd_output_dataset_{data_id}.csv")

            preview.dataset.to_csv(save_path, index=True)

            # 🔄 Post-Process: Convert to TrajectoryEvaluator Ground Truth Format
            logger.info("Converting NDD output to TrajectoryEvaluator format...")

            try:
                # Save JSON version for post-processing
                json_path = save_path.replace('.csv', '.json')
                dataset_records = preview.dataset.to_dict('records')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(dataset_records, f, indent=2)
                logger.info("Dataset JSON saved to: %s", json_path)

                # Convert to NAT evaluation format using dedicated function
                ground_truth_path = await convert_to_nat_eval_format(
                    dataset_records=dataset_records,
                    agent_tool_details=agent_tool_details,
                    output_dir=output_dir,
                    data_id=data_id,
                    current_date=current_date
                )
                logger.info("NAT evaluation ground truth created: %s", ground_truth_path)

            except Exception as e:
                logger.error("Post-processing failed: %s", e)

            return save_path

        except Exception as e:
            logger.error("Data generation failed: %s", str(e))
            return str(e)

    try:
        yield FunctionInfo.create(
            single_fn=generate_synthetic_data,
            description="Generate synthetic data using NeMo Data Designer patterns",
        )

    except GeneratorExit:
        logger.warning("NDD Client function exited early!")
    finally:
        logger.info("Cleaning up NDD Client workflow.")
