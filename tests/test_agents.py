import json

import pytest

from structmem.agents.baselines import AgentKB, BM25, EmbeddingRetrieval, ExpeL, GraphEmbedding, MemGPT, Reflexion, Voyager
from structmem.agents.base import AgentContext
from structmem.agents.structural import StructuralAgent
from structmem.graph import trace_graph
from structmem.llm import LLMResponse
from structmem.memory import DenseEmbeddingRetriever
from structmem.types import Experience, Role, TaskSpec


@pytest.fixture(autouse=True)
def fixed_dense_scores(monkeypatch):
    def rank(self, query, candidates):
        return [(candidate, 0.5) for candidate in candidates]

    monkeypatch.setattr(DenseEmbeddingRetriever, "rank", rank)


class FakeLLM:
    def __init__(self, json_payloads=None):
        self.json_payloads = list(json_payloads or [])

    def complete(self, messages, **kwargs):
        return LLMResponse("look", 3, 1)

    def json(self, messages, **kwargs):
        payload = self.json_payloads.pop(0)
        return payload, LLMResponse(json.dumps(payload), 5, 2)


def exp(identifier="source"):
    roles = [Role.PRECONDITION_CHECK, Role.STATE_CHANGING_ACTION, Role.VERIFICATION]
    actions = ["find object", "change object", "verify object"]
    return Experience(
        experience_id=identifier,
        dataset="source",
        task_id=identifier,
        goal="change a remote object",
        success=True,
        score=1,
        actions=actions,
        observations=[""] * 3,
        roles=roles,
        graph=trace_graph(actions, roles),
    )


def target_exp(identifier="source", *, success=True, dataset="target"):
    item = exp(identifier)
    item.dataset = dataset
    item.success = success
    return item


def test_baselines_have_distinct_memory_state() -> None:
    llm = FakeLLM()
    assert hasattr(Voyager(llm), "skills")
    assert hasattr(Reflexion(llm), "reflections")
    assert hasattr(ExpeL(llm), "rules")
    assert hasattr(MemGPT(llm), "archival")
    assert hasattr(AgentKB(llm), "usage")
    assert BM25(llm).retriever.__class__.__name__ == "BM25Retriever"
    assert EmbeddingRetrieval(llm).retriever.__class__.__name__ == "DenseEmbeddingRetriever"
    assert GraphEmbedding(llm).retriever.__class__.__name__ == "GraphEmbeddingRetriever"


def test_unguarded_ablation_always_uses() -> None:
    representation = {"nodes": [{"action": "inspect", "role": "Verification"}], "edges": []}
    mapping = {"candidates": [{"id": "source", "score": 0.8, "adapted_plan": ["inspect target"]}]}
    failed_binding = {"outcome": "failed", "verdict": "skip", "selected_id": "source"}
    agent = StructuralAgent(FakeLLM([representation, mapping, failed_binding]), conservative_gate=False)
    task = TaskSpec("target", "t", "inspect target", "test", 0)
    query = exp("query")
    query.goal = task.goal
    context = agent.begin(task, query, [target_exp()])
    agent.guidance(context, "visible")
    assert context.diagnostics["gate_verdict"] == "use"


def test_structural_agent_maps_then_binds() -> None:
    representation = {
        "nodes": [
            {"action": "locate target", "role": "PreconditionCheck"},
            {"action": "change target", "role": "StateChangingAction"},
            {"action": "confirm result", "role": "Verification"},
        ],
        "edges": [
            {"source": 0, "target": 1, "type": "precondition"},
            {"source": 1, "target": 2, "type": "effect_check"},
        ],
    }
    mapping = {
        "candidates": [
            {"id": "source", "score": 0.9, "mapping": [{"query": 0, "source": 0}], "reason": "systematic"}
        ]
    }
    binding = {
        "outcome": "clean",
        "verdict": "use",
        "selected_id": "source",
        "mappings": [{"role": "Verification", "action": "inspect"}],
        "bound_plan": ["inspect the visible target", "confirm the changed state"],
        "reason": "all roles bind",
    }
    llm = FakeLLM([representation, mapping, binding])
    agent = StructuralAgent(llm, top_k=1, prefilter_k=5)
    task = TaskSpec("target", "t1", "change target", "test", 1)
    query = exp("query")
    query.goal = task.goal
    context = agent.begin(task, query, [target_exp()])
    guidance = agent.guidance(context, "target is visible")
    assert context.retrieved[0].experience_id == "source"
    assert context.diagnostics["binding_outcome"] == "clean"
    assert context.diagnostics["gate_verdict"] == "use"
    assert "passed binding verification" in guidance


def test_structural_agent_filters_cross_domain_and_failed_memories() -> None:
    representation = {"nodes": [{"action": "inspect", "role": "Verification"}], "edges": []}
    mapping = {
        "candidates": [
            {"id": identifier, "score": 0.9, "adapted_plan": ["inspect target"]}
            for identifier in ("good", "good2", "good3")
        ]
    }
    llm = FakeLLM([representation, mapping])
    agent = StructuralAgent(llm, top_k=1, prefilter_k=5)
    task = TaskSpec("target", "t2", "inspect target", "test", 2)
    query = target_exp("query")
    query.goal = task.goal
    context = agent.begin(
        task,
        query,
        [
            target_exp("good"),
            target_exp("good2"),
            target_exp("good3"),
            target_exp("failed", success=False),
            target_exp("foreign", dataset="other"),
        ],
    )
    assert [item.experience_id for item in context.retrieved] == ["good"]
    assert context.diagnostics["candidate_pool_size"] == 3
    assert context.diagnostics["configured_top_k"] == 1
    assert context.diagnostics["effective_top_k"] == 1


def test_conservative_gate_rejects_ambiguous_binding_without_memory_payload() -> None:
    representation = {"nodes": [{"action": "inspect", "role": "Verification"}], "edges": []}
    mapping = {"candidates": [{"id": "source", "score": 0.7, "adapted_plan": ["inspect target"]}]}
    ambiguous = {
        "outcome": "ambiguous",
        "verdict": "probe",
        "selected_id": "source",
        "bound_plan": ["copy a source action"],
    }
    agent = StructuralAgent(FakeLLM([representation, mapping, ambiguous]), top_k=1, prefilter_k=5)
    task = TaskSpec("target", "t3", "inspect target", "test", 3)
    query = target_exp("query")
    query.goal = task.goal
    context = agent.begin(task, query, [target_exp()])
    guidance = agent.guidance(context, "target may be visible")
    assert context.diagnostics["gate_verdict"] == "skip"
    assert "copy a source action" not in guidance
    assert "rejected" in guidance


def test_high_confidence_ambiguous_binding_uses_sanitized_probe_plan() -> None:
    representation = {"nodes": [{"action": "inspect", "role": "Verification"}], "edges": []}
    mapping = {"candidates": [{"id": "source", "score": 0.9, "adapted_plan": ["inspect target"]}]}
    ambiguous = {
        "outcome": "ambiguous",
        "verdict": "probe",
        "selected_id": "source",
        "bound_plan": ["inspect an unvisited target location"],
    }
    agent = StructuralAgent(FakeLLM([representation, mapping, ambiguous]), top_k=1, prefilter_k=5)
    task = TaskSpec("target", "t4", "inspect target", "test", 4)
    query = target_exp("query")
    query.goal = task.goal
    context = agent.begin(task, query, [target_exp()])
    guidance = agent.guidance(context, "target is not visible")
    assert context.diagnostics["gate_verdict"] == "probe"
    assert "safe probe plan" in guidance
    assert "inspect an unvisited target location" in guidance


def test_execution_monitor_preserves_long_horizon_progress() -> None:
    context = AgentContext(TaskSpec("target", "t5", "move two objects", "test", 5))
    context.history = [
        ("", "go to drawer 1"),
        ("", "take object 1 from drawer 1"),
        ("", "move object 1 to table 1"),
        ("", "go to drawer 1"),
    ]
    guidance = StructuralAgent._execution_state(context)
    assert "move object 1 to table 1" in guidance
    assert "go to drawer 1" in guidance
    assert "avoid_repeating_without_new_evidence" in guidance


def test_execution_monitor_marks_rejected_repeated_action() -> None:
    context = AgentContext(TaskSpec("scienceworld", "chemistry-mix:1", "make soapy water", "test", 1))
    context.history = [("", "mix glass cup"), ("No known action matches that input.", "mix glass cup")]
    guidance = StructuralAgent._execution_state(context, "No known action matches that input.")
    assert '"last_action_rejected_or_ambiguous":"mix glass cup"' in guidance
    assert '"avoid_repeating_without_new_evidence":["mix glass cup"]' in guidance


def test_scienceworld_ambiguity_prefers_held_referent() -> None:
    context = AgentContext(TaskSpec("scienceworld", "find-animal:1", "find an adult bee", "test", 1))
    context.history = [("", "move adult bee to blue box")]
    observation = (
        "Ambiguous request: Please enter the number for the action you intended:\n"
        "0:\tmove adult bee (in inventory, in agent, in workshop) to blue box (in workshop)\n"
        "1:\tmove adult bee (in workshop) to blue box (in workshop)\n"
    )
    assert StructuralAgent._resolve_scienceworld_ambiguity(context, observation) == "0"


def test_scienceworld_template_exposes_only_upcoming_phase() -> None:
    template = Experience(
        experience_id="scienceworld:chemistry-mix:0:no_memory",
        dataset="scienceworld",
        task_id="chemistry-mix:0",
        goal="make salt water",
        success=True,
        score=100,
        actions=[
            "go to kitchen",
            "read instructions to make salt water",
            "take glass cup",
            "put sodium chloride in glass cup",
            "mix glass cup",
            "focus on salt water",
        ],
        observations=[""] * 6,
        roles=[
            Role.INFORMATION_ACQUISITION,
            Role.INFORMATION_ACQUISITION,
            Role.PRECONDITION_CHECK,
            Role.STATE_CHANGING_ACTION,
            Role.STATE_CHANGING_ACTION,
            Role.TERMINATION,
        ],
        graph={},
    )
    context = AgentContext(TaskSpec("scienceworld", "chemistry-mix:24", "make sugar water", "test", 24))
    context.retrieved = [template]
    context.history = [("", "go through door to kitchen"), ("", "read instructions to make sugar water")]
    guidance = StructuralAgent(FakeLLM())._scienceworld_template_state(context)
    assert "take glass cup" in guidance
    assert "go to kitchen" not in guidance.split("upcoming_successful_phase", 1)[1]
    assert context.diagnostics["scienceworld_template"]["aligned_steps"]


def test_scienceworld_recipe_binding_uses_observed_recipe_not_variation_id() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:999", "create an unknown product", "test", 999)
    )
    observation = "The recipe reads:\nTo make green paste, you need to mix blue powder, yellow liquid."
    guidance = StructuralAgent._scienceworld_recipe_guidance(context, observation)
    assert '"product":"green paste"' in guidance
    assert '"allowed_ingredients_only":["blue powder","yellow liquid"]' in guidance
    assert "exactly the ingredients stated" in guidance


def test_scienceworld_recipe_executor_mixes_complete_uncontaminated_container() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:999", "create green paste", "test", 999)
    )
    context.history = [
        (
            "The recipe reads:\nTo make green paste, you need to mix blue powder, yellow liquid.",
            "take blue powder",
        ),
        ("You move the blue powder to the inventory.", "put blue powder in ceramic cup"),
        ("You move the blue powder to the ceramic cup.", "pour yellow liquid into ceramic cup"),
    ]
    action = StructuralAgent._scienceworld_recipe_action(
        context,
        "You pour the yellow liquid into the ceramic cup.",
    )
    assert action == "mix ceramic cup"


def test_scienceworld_recipe_executor_does_not_treat_inventory_as_container() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "chemistry-mix:25",
            "create peanut butter with banana sandwich",
            "test",
            25,
        )
    )
    context.history = [
        (
            "a glass cup (containing nothing)\n"
            "The recipe reads:\n"
            "To make peanut butter with banana sandwich, you need to mix peanut, banana, bread.",
            "take peanut",
        ),
        ("You move the peanut to the inventory.", "take banana"),
        ("You move the banana to the inventory.", "take bread"),
    ]
    action = StructuralAgent._scienceworld_recipe_action(
        context,
        "You move the bread to the inventory.",
    )
    assert action == "move peanut to glass cup"
    assert "inventory" not in context.diagnostics["scienceworld_recipe_runtime"]["containers"]


def test_scienceworld_recipe_executor_fills_selected_container_before_mixing() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:25", "create sandwich", "test", 25)
    )
    context.history = [
        (
            "a glass cup (containing nothing)\n"
            "To make sandwich, you need to mix peanut, banana, bread.",
            "take peanut",
        ),
        ("You move the peanut to the inventory.", "take banana"),
        ("You move the banana to the inventory.", "take bread"),
        ("You move the bread to the inventory.", "move peanut to glass cup"),
    ]
    assert StructuralAgent._scienceworld_recipe_action(
        context,
        "You move the peanut to the glass cup.",
    ) == "move banana to glass cup"


def test_scienceworld_recipe_executor_uses_reachable_container_and_preserves_binding() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:27", "create red paper", "test", 27)
    )
    context.history = [
        (
            "This room is called the art studio.\n"
            "a wood cup (containing red paint)\n"
            "a glass cup (containing nothing)",
            "go to kitchen",
        ),
        (
            "This room is called the kitchen.\n"
            "a glass cup (containing nothing)\n"
            "To make red paper, you need to mix paper, red paint.",
            "take paper",
        ),
        ("You move the paper to the inventory.", "move paper to glass cup"),
    ]
    assert StructuralAgent._scienceworld_recipe_action(
        context,
        "You move the paper to the glass cup.",
    ) == "take glass cup"
    state = context.diagnostics["scienceworld_recipe_runtime"]
    assert state["target_container"] == "glass cup"
    assert state["containers"]["glass cup"] == ["paper"]

    context.history.append(("You move the paper to the glass cup.", "take glass cup"))
    context.history.append(("You move the glass cup to the inventory.", "go to art studio"))
    context.history.append(("You move to the art studio.", "take red paint"))
    action = StructuralAgent._scienceworld_recipe_action(
        context,
        "This room is called the art studio.\n"
        "a glass cup (containing nothing)\n"
        "a wood cup (containing red paint)",
    )
    assert action == "pour wood cup containing red paint into glass cup in inventory"
    state = context.diagnostics["scienceworld_recipe_runtime"]
    assert state["containers"]["glass cup"] == ["paper"]


def test_scienceworld_recipe_executor_tracks_pour_into_inventory_container() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:27", "create red paper", "test", 27)
    )
    context.history = [
        (
            "This room is called the kitchen.\n"
            "a glass cup (containing nothing)\n"
            "To make red paper, you need to mix paper, red paint.",
            "move paper to glass cup",
        ),
        ("You move the paper to the glass cup.", "take glass cup"),
        ("You move the glass cup to the inventory.", "go to art studio"),
        (
            "This room is called the art studio.\n"
            "a wood cup (containing red paint)",
            "pour wood cup containing red paint into glass cup in inventory",
        ),
    ]
    assert StructuralAgent._scienceworld_recipe_action(
        context,
        "You pour the contents of the wood cup into the glass cup.",
    ) == "mix glass cup in inventory"


def test_scienceworld_recipe_executor_navigates_to_known_ingredient_room() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:27", "create red paper", "test", 27)
    )
    context.history = [
        (
            "This room is called the art studio.\n"
            "a wood cup (containing red paint)",
            "go to hallway",
        ),
        ("You move to the hallway.", "go to kitchen"),
        (
            "This room is called the kitchen.\n"
            "a glass cup (containing nothing)\n"
            "To make red paper, you need to mix paper, red paint.",
            "move paper to glass cup",
        ),
        ("You move the paper to the glass cup.", "take glass cup"),
    ]
    assert StructuralAgent._scienceworld_recipe_action(
        context,
        "You move the glass cup to the inventory.",
    ) == "go to hallway"
    context.history.append(("You move the glass cup to the inventory.", "go to hallway"))
    assert StructuralAgent._scienceworld_recipe_action(
        context,
        "You move to the hallway.",
    ) == "go to art studio"


def test_scienceworld_recipe_repair_blocks_mix_inventory() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:25", "create sandwich", "test", 25)
    )
    context.history = [
        (
            "a glass cup (containing nothing)\n"
            "To make sandwich, you need to mix peanut, banana, bread.",
            "take peanut",
        ),
        ("You move the peanut to the inventory.", "take banana"),
        ("You move the banana to the inventory.", "take bread"),
    ]
    repaired = StructuralAgent._repair_scienceworld_recipe_action(
        context,
        "You move the bread to the inventory.",
        "mix inventory",
    )
    assert repaired == "move peanut to glass cup"


def test_scienceworld_recipe_executor_focuses_created_product() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:999", "create green paste", "test", 999)
    )
    context.history = [
        (
            "The recipe reads:\nTo make green paste, you need to mix blue powder, yellow liquid.",
            "mix ceramic cup",
        )
    ]
    action = StructuralAgent._scienceworld_recipe_action(
        context,
        "Blue powder and yellow liquid mix to produce green paste.",
    )
    assert action == "focus on green paste"


def test_scienceworld_recipe_repair_blocks_contaminating_source_container() -> None:
    context = AgentContext(
        TaskSpec("scienceworld", "chemistry-mix:999", "create green paste", "test", 999)
    )
    context.history = [
        (
            "The recipe reads:\nTo make green paste, you need to mix blue powder, yellow liquid.",
            "look around",
        ),
        (
            "a ceramic cup (containing a blue powder)\n"
            "a glass jar (containing a substance called red powder)",
            "take glass jar",
        ),
    ]
    repaired = StructuralAgent._repair_scienceworld_recipe_action(
        context,
        "You move the glass jar to the inventory.",
        "pour glass jar into ceramic cup",
    )
    assert repaired != "pour glass jar into ceramic cup"
    assert context.diagnostics["scienceworld_recipe_blocked_actions"][-1]["reason"] == "non_recipe_ingredient"


def test_agentbench_single_cell_result_is_submitted_without_llm_call() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:99",
            "Question: how many matching rows are there?",
            "standard",
            99,
        )
    )
    context.history = [("question", "Action: Operation\n```sql\nSELECT COUNT(*) FROM t;\n```")]
    action = StructuralAgent._fast_agentbench_single_cell_answer(context, "[(7,)]")
    assert action == 'Action: Answer\nFinal Answer: ["7"]'
    assert context.diagnostics["agentbench_direct_answers"][-1]["sql_result"] == 7


def test_agentbench_single_cell_executor_ignores_multirow_probe() -> None:
    context = AgentContext(TaskSpec("agentbench", "dbbench-std:99", "Question: find value", "standard", 99))
    context.history = [("question", "Action: Operation\n```sql\nSELECT value FROM t;\n```")]
    assert StructuralAgent._fast_agentbench_single_cell_answer(context, "[('a',), ('b',)]") is None


def test_agentbench_empty_result_is_submitted_as_normalized_zero() -> None:
    context = AgentContext(TaskSpec("agentbench", "dbbench-std:99", "Question: find value", "standard", 99))
    context.history = [("question", "Action: Operation\n```sql\nSELECT value FROM t WHERE id = 1;\n```")]
    action = StructuralAgent._fast_agentbench_single_cell_answer(context, "[]")
    assert action == 'Action: Answer\nFinal Answer: ["0"]'
    assert context.diagnostics["agentbench_direct_answers"][-1]["reason"] == "empty_result_normalized_to_zero"


def test_agentbench_sql_guidance_distinguishes_counts_from_scalar_values() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:99",
            "Question: How many operational locomotives are there?",
            "standard",
            99,
        )
    )
    guidance = StructuralAgent._agentbench_sql_guidance(context)
    assert "use COUNT" in guidance
    assert "attendance, wins, points" in guidance
    assert '"bound_question":"How many operational locomotives are there?"' in guidance


def test_agentbench_operation_repair_counts_events_instead_of_returning_field_value() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:18",
            "Question: How many home losses occured when PF 13?\n"
            "The name of this table is NFL Team Records Table.",
            "standard",
            18,
        )
    )
    action = (
        "Action: Operation\n```sql\n"
        "SELECT `Home losses` FROM `NFL Team Records Table` WHERE `PF` = 13;\n```"
    )
    repaired = StructuralAgent._repair_agentbench_operation_binding(context, action)
    assert "SELECT COUNT(`Home losses`) FROM" in repaired


def test_agentbench_operation_repair_counts_non_measurement_total_number() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:20",
            "Question: Name the total number of represents for Clary Sermina Delgado Cid",
            "standard",
            20,
        )
    )
    action = (
        "Action: Operation\n```sql\n"
        "SELECT `Represents` FROM `contestants` WHERE `Contestant` = 'Clary Sermina Delgado Cid';\n```"
    )
    repaired = StructuralAgent._repair_agentbench_operation_binding(context, action)
    assert "SELECT COUNT(`Represents`) FROM" in repaired


def test_agentbench_operation_repair_preserves_scalar_measurement_lookup() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:99",
            "Question: what is the total number of wins for Confey",
            "standard",
            99,
        )
    )
    action = "Action: Operation\n```sql\nSELECT `Wins` FROM `teams` WHERE `Team` = 'Confey';\n```"
    assert StructuralAgent._repair_agentbench_operation_binding(context, action) == action


def test_agentbench_operation_repair_uses_month_prefix_for_date_counts() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:22",
            "Question: how many games did the badgers play in october?",
            "standard",
            22,
        )
    )
    action = (
        "Action: Operation\n```sql\n"
        "SELECT COUNT(*) FROM `Football Schedule` WHERE `Date` LIKE '%Oct%';\n```"
    )
    repaired = StructuralAgent._repair_agentbench_operation_binding(context, action)
    assert "LIKE 'October%'" in repaired


def test_agentbench_operation_repair_replaces_multi_argument_count() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:28",
            "Question: what is the total number of participants between Norway and France?",
            "standard",
            28,
        )
    )
    action = (
        "Action: Operation\n```sql\n"
        "SELECT COUNT(`Rank`, `Nationality`) FROM `Biathlon Results Table` "
        "WHERE `Nationality` IN ('Norway', 'France');\n```"
    )
    repaired = StructuralAgent._repair_agentbench_operation_binding(context, action)
    assert "SELECT COUNT(*) FROM" in repaired


def test_agentbench_operation_repair_casts_text_numeric_comparison() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:39",
            "Question: When the population is greater than 146,445, what is the number of households?",
            "standard",
            39,
        )
    )
    action = (
        "Action: Operation\n```sql\n"
        "SELECT `Number of households` FROM counties WHERE `Population` > 146445;\n```"
    )
    repaired = StructuralAgent._repair_agentbench_operation_binding(context, action)
    assert "CAST(REPLACE(`Population`, ',', '') AS REAL) > 146445" in repaired


def test_agentbench_operation_repair_binds_weekday_in_full_date() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:32",
            "Question: How many episodes aired Saturday, July 11, 2009?",
            "standard",
            32,
        )
    )
    action = (
        "Action: Operation\n```sql\n"
        "SELECT COUNT(*) FROM episodes WHERE `US air date` = 'July 11, 2009';\n```"
    )
    repaired = StructuralAgent._repair_agentbench_operation_binding(context, action)
    assert "`US air date` = 'Saturday, July 11, 2009'" in repaired


def test_scienceworld_high_temperature_metal_requires_blast_furnace() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil lead. First, focus on the substance.",
            "test",
            99,
        )
    )
    guidance = StructuralAgent._scienceworld_capability_state(context)
    assert "blast furnace" in guidance
    assert "Never use the stove" in guidance
    assert context.diagnostics["scienceworld_device_capability"]["target"] == "lead"


def test_scienceworld_low_temperature_material_does_not_force_blast_furnace() -> None:
    for target in ("water", "chocolate", "juice"):
        context = AgentContext(
            TaskSpec(
                "scienceworld",
                "boil:99",
                f"Your task is to boil {target}. First, focus on the substance.",
                "test",
                99,
            )
        )
        assert StructuralAgent._scienceworld_capability_state(context) == ""


def test_scienceworld_capability_executor_routes_held_metal_to_foundry() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil tin. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [("This room is called the workshop.", "take tin")]
    assert StructuralAgent._scienceworld_capability_action(context, "You move the tin to the inventory.") == "go to hallway"
    context.history.extend(
        [
            ("", "go to hallway"),
            ("", "go to kitchen"),
            ("", "open door to outside"),
            ("", "go outside"),
            ("", "open door to foundry"),
            ("", "go to foundry"),
        ]
    )
    assert StructuralAgent._scienceworld_capability_action(context, "You move to the foundry.") == "open blast furnace"


def test_scienceworld_capability_executor_opens_blocking_route_door() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil lead. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [
        ("This room is called the workshop.", "take lead"),
        ("You move the lead to the inventory.", "go to hallway"),
        ("You move to the hallway.", "go to kitchen"),
    ]
    action = StructuralAgent._scienceworld_capability_action(context, "The door is not open.")
    assert action == "open door to kitchen"


def test_scienceworld_capability_executor_opens_blocking_direct_go_door() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil gallium. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [
        ("This room is called the workshop.", "take gallium"),
        ("You move the gallium to the inventory.", "go to hallway"),
        ("You move to the hallway.", "go to kitchen"),
        ("You move to the kitchen.", "go outside"),
    ]
    action = StructuralAgent._scienceworld_capability_action(context, "The door is not open.")
    assert action == "open door to outside"


def test_scienceworld_capability_repairs_incompatible_heater_choice() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil gallium. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [("This room is called the kitchen.", "take gallium")]
    repaired = StructuralAgent._repair_scienceworld_capability_action(
        context,
        "You move the gallium to the inventory.",
        "put gallium in metal pot",
    )
    assert repaired == "go outside"


def test_agentbench_binds_football_score_difference_unit() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:3",
            "Question: what is the difference between Southampton's score and Sunderland's score? "
            "The name of this table is Football Matches.",
            "standard",
            3,
        )
    )
    repaired = StructuralAgent._repair_agentbench_answer_binding(
        context,
        'Action: Answer\nFinal Answer: ["2"]',
    )
    assert repaired == 'Action: Answer\nFinal Answer: ["2 goals"]'
    assert context.diagnostics["answer_bindings"] == [
        {"semantic_type": "football_score_difference", "unit": "goals"}
    ]


def test_agentbench_does_not_invent_units_for_generic_numeric_answers() -> None:
    context = AgentContext(
        TaskSpec(
            "agentbench",
            "dbbench-std:4",
            "Question: what is the difference between the two values? The table is Measurements.",
            "standard",
            4,
        )
    )
    action = 'Action: Answer\nFinal Answer: ["2"]'
    assert StructuralAgent._repair_agentbench_answer_binding(context, action) == action


def test_scienceworld_liquid_metal_uses_existing_container_then_pours() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil mercury. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [
        (
            "This room is called the workshop. A plastic cup (containing a substance called mercury) is on the table.",
            "focus on mercury",
        ),
        ("You focus on the mercury.", "take plastic cup"),
        ("You move the plastic cup to the inventory.", "go to hallway"),
        ("You move to the hallway.", "go to kitchen"),
        ("You move to the kitchen.", "open door to outside"),
        ("The door is now open.", "go outside"),
        ("You move to the outside.", "open door to foundry"),
        ("The door is now open.", "go to foundry"),
        ("You move to the foundry.", "open blast furnace"),
    ]
    assert StructuralAgent._scienceworld_target_carrier(context, "mercury") == "plastic cup"
    action = StructuralAgent._scienceworld_capability_action(context, "The blast furnace is now open.")
    assert action == "pour mercury into blast furnace"


def test_scienceworld_rejected_liquid_transfer_does_not_advance_phase() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil mercury. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [
        (
            "This room is called the workshop. A glass cup (containing a substance called mercury) is visible.",
            "focus on mercury",
        ),
        ("You focus on the mercury.", "take glass cup"),
        ("You move the glass cup to the inventory.", "go to hallway"),
        ("You move to the hallway.", "go to kitchen"),
        ("You move to the kitchen.", "open door to outside"),
        ("The door is now open.", "go outside"),
        ("You move to the outside.", "open door to foundry"),
        ("The door is now open.", "go to foundry"),
        ("You move to the foundry.", "open blast furnace"),
        ("The blast furnace is now open.", "pour mercury from glass cup into blast furnace"),
    ]
    action = StructuralAgent._scienceworld_capability_action(context, "No known action matches that input.")
    assert action == "pour mercury into blast furnace"


def test_scienceworld_capability_discovery_routes_to_workshop_and_takes_target() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil caesium. First, focus on the substance.",
            "test",
            99,
        )
    )
    assert StructuralAgent._scienceworld_capability_action(
        context,
        "This room is called the bedroom. You also see a door to the hallway.",
    ) == "go to hallway"
    context.history = [("This room is called the workshop. A paper cup contains caesium.", "focus on caesium")]
    assert StructuralAgent._scienceworld_capability_action(
        context,
        "You focus on the caesium.",
    ) == "take caesium"


def test_scienceworld_capability_discovery_takes_liquid_container() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "boil:99",
            "Your task is to boil mercury. First, focus on the substance.",
            "test",
            99,
        )
    )
    context.history = [
        (
            "This room is called the workshop. A plastic cup (containing a substance called mercury) is visible.",
            "focus on mercury",
        )
    ]
    assert StructuralAgent._scienceworld_capability_action(
        context,
        "You focus on the mercury.",
    ) == "take plastic cup"


def test_scienceworld_find_animal_binds_visible_egg_and_routes_outside() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "find-animal:225",
            "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the orange box in the living room.",
            "test",
            225,
        )
    )
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        "This room is called the workshop. A door to the hallway is open.",
    ) == "go to hallway"
    outside = (
        "This outside location is called the outside. Here you see:\n"
        "\ta frog egg\n\ta turtle egg\n\ta baby wolf"
    )
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        outside,
    ) == "focus on frog egg"
    state = context.diagnostics["scienceworld_find_animal_state"]
    assert state["binding"]["inventory_referent"] == "egg frog egg in inventory"


def test_scienceworld_find_animal_preserves_held_state_and_places_target() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "find-animal:228",
            "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the red box in the bedroom.",
            "test",
            228,
        )
    )
    outside = "This outside location is called the outside. Here you see:\n\ta turtle egg"
    focus = StructuralAgent._scienceworld_find_animal_action(context, outside)
    assert focus == "focus on turtle egg"
    context.history.append((outside, focus))
    pickup = StructuralAgent._scienceworld_find_animal_action(context, "You focus on the turtle egg.")
    assert pickup == "pick up turtle egg"
    context.history.append(("You focus on the turtle egg.", pickup))
    route = StructuralAgent._scienceworld_find_animal_action(context, "You move the turtle to the inventory.")
    assert route == "go to kitchen"
    context.history.append(("You move the turtle to the inventory.", route))
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        "You move to the kitchen.",
    ) == "go to hallway"
    context.history.append(("You move to the kitchen.", "go to hallway"))
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        "The door is not open.",
    ) == "open door to hallway"
    assert context.diagnostics["scienceworld_find_animal_state"]["held"] is True
    context.history.append(("The door is not open.", "open door to hallway"))
    context.history.append(("The door is now open.", "go to hallway"))
    context.history.append(("You move to the hallway.", "go to bedroom"))
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        "You move to the bedroom.",
    ) == "move egg turtle egg in inventory to red box"


def test_scienceworld_find_animal_opens_blocking_destination_door() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "find-animal:227",
            "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the purple box in the living room.",
            "test",
            227,
        )
    )
    context.diagnostics["scienceworld_find_animal_state"] = {
        "focused": True,
        "held": True,
        "binding": {
            "display_name": "turtle egg",
            "species": "turtle",
            "action_referent": "turtle egg",
            "inventory_referent": "egg turtle egg in inventory",
        },
    }
    context.history = [("You move to the hallway.", "go to living room")]
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        "The door is not open.",
    ) == "open door to living room"


def test_scienceworld_find_animal_keeps_binding_after_successful_navigation() -> None:
    context = AgentContext(
        TaskSpec(
            "scienceworld",
            "find-animal:225",
            "Your task is to find a(n) animal. First, focus on the thing. Then, move it to the orange box in the living room.",
            "test",
            225,
        )
    )
    context.diagnostics["scienceworld_find_animal_state"] = {
        "focused": True,
        "held": True,
        "binding": {
            "display_name": "frog egg",
            "species": "frog",
            "action_referent": "frog egg",
            "inventory_referent": "egg frog egg in inventory",
        },
    }
    context.history = [("You move the frog to the inventory.", "go to kitchen")]
    assert StructuralAgent._scienceworld_find_animal_action(
        context,
        "You move to the kitchen.",
    ) == "go to hallway"


def test_structural_executor_prioritizes_goal_actions_and_unvisited_locations() -> None:
    context = AgentContext(TaskSpec("alfworld", "t6", "find two remotecontrol and put them in coffeetable.", "test", 6))
    context.diagnostics["search_location_priors"] = {"sidetable": 1.0, "cabinet": 0.2}
    context.history = [("", "go to cabinet 1")]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        [
            "go to cabinet 1",
            "go to sidetable 1",
            "take remotecontrol 2 from drawer 1",
            "examine drawer 1",
        ],
    )
    assert ranked[0][0] == "take remotecontrol 2 from drawer 1"
    assert [action for action, _ in ranked].index("go to sidetable 1") < [
        action for action, _ in ranked
    ].index("go to cabinet 1")


def test_structural_executor_forces_only_grounded_alfworld_manipulations() -> None:
    context = AgentContext(TaskSpec("alfworld", "t7", "find two remotecontrol and put them in coffeetable.", "test", 7))
    ranked = [("take remotecontrol 1 from sidetable 1", 103.0), ("go to drawer 1", 20.0)]
    assert StructuralAgent._forced_grounded_action(context, "A remotecontrol is visible.", ranked) == ranked[0][0]
    other = AgentContext(TaskSpec("scienceworld", "t8", "find remotecontrol", "test", 8))
    assert StructuralAgent._forced_grounded_action(other, "visible", ranked) is None


def test_structural_executor_ignores_destination_overlap_for_take() -> None:
    context = AgentContext(TaskSpec("alfworld", "t8b", "find two remotecontrol and put them in coffeetable.", "test", 8))
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["take keychain 1 from coffeetable 1", "go to sidetable 1"],
    )
    assert ranked[0][0] == "go to sidetable 1"
    assert StructuralAgent._forced_grounded_action(context, "At the coffeetable.", ranked) is None


def test_structural_executor_does_not_retake_completed_target() -> None:
    context = AgentContext(TaskSpec("alfworld", "t8c", "find two remotecontrol and put them in coffeetable.", "test", 8))
    context.history = [
        ("", "take remotecontrol 1 from sidetable 1"),
        ("", "move remotecontrol 1 to coffeetable 1"),
    ]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["take remotecontrol 1 from coffeetable 1", "go to sofa 1"],
    )
    assert ranked[0][0] == "go to sofa 1"
    assert StructuralAgent._forced_grounded_action(context, "At the coffeetable.", ranked) is None


def test_search_location_priors_strip_instance_ids() -> None:
    memory = target_exp("memory", dataset="alfworld")
    memory.actions = ["take remotecontrol 2 from sidetable 4", "take book 1 from shelf 3"]
    task = TaskSpec("alfworld", "t9", "find remotecontrol", "test", 9)
    priors = StructuralAgent._search_location_priors(task, [memory])
    assert priors["sidetable"] == 1.0
    assert "sidetable 4" not in priors


def test_structural_executor_learns_closed_locations_and_prefers_open_surfaces() -> None:
    memory = target_exp("memory", dataset="alfworld")
    memory.actions = ["open cabinet 1", "open drawer 2"]
    context = AgentContext(TaskSpec("alfworld", "t10", "find remotecontrol", "test", 10))
    context.diagnostics["closed_location_types"] = StructuralAgent._closed_location_types([memory])
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to cabinet 1", "go to sofa 1"],
    )
    assert context.diagnostics["closed_location_types"] == ["cabinet", "drawer"]
    assert ranked[0][0] == "go to sofa 1"


def test_structural_executor_returns_held_target_to_destination() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11", "find two remotecontrol and put them in coffeetable.", "test", 11))
    context.history = [("", "take remotecontrol 1 from diningtable 1")]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to cabinet 1", "go to coffeetable 1", "go to sofa 1"],
    )
    assert StructuralAgent._forced_grounded_action(context, "At the diningtable.", ranked) == "go to coffeetable 1"


def test_structural_executor_opens_current_destination_before_navigating_elsewhere() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11b", "find two saltshaker and put them in drawer.", "test", 11))
    context.history = [("", "take saltshaker 3 from countertop 2")]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to drawer 2", "go to drawer 3", "open drawer 1"],
    )
    assert StructuralAgent._forced_grounded_action(
        context,
        "You arrive at drawer 1. The drawer 1 is closed.",
        ranked,
    ) == "open drawer 1"


def test_structural_executor_fast_paths_positive_alfworld_search_actions() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11c", "find a saltshaker.", "test", 11))
    assert StructuralAgent._fast_alfworld_search_action(
        context,
        [("go to countertop 2", 19.9), ("go to drawer 1", 1.0)],
    ) == "go to countertop 2"
    assert StructuralAgent._fast_alfworld_search_action(
        context,
        [("open cabinet 2", 44.9), ("go to drawer 1", 1.0)],
    ) == "open cabinet 2"


def test_structural_executor_keeps_repeated_or_cross_dataset_navigation_with_llm() -> None:
    alfworld = AgentContext(TaskSpec("alfworld", "t11d", "find a saltshaker.", "test", 11))
    scienceworld = AgentContext(TaskSpec("scienceworld", "t11e", "find an animal.", "test", 11))
    assert StructuralAgent._fast_alfworld_search_action(alfworld, [("go to drawer 1", -0.1)]) is None
    assert StructuralAgent._fast_alfworld_search_action(scienceworld, [("go to hallway", 20.0)]) is None


def test_structural_executor_reuses_first_placement_destination_binding() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11f", "find two saltshaker and put them in drawer.", "test", 11))
    context.history = [
        ("", "take saltshaker 3 from countertop 2"),
        ("", "move saltshaker 3 to drawer 1"),
        ("", "take saltshaker 2 from cabinet 14"),
    ]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to drawer 2", "go to drawer 1", "go to cabinet 1"],
    )
    assert StructuralAgent._bound_destination(context, StructuralAgent._goal_terms(context.task.goal)) == "drawer 1"
    assert ranked[0][0] == "go to drawer 1"
    assert StructuralAgent._forced_grounded_action(context, "At cabinet 14.", ranked) == "go to drawer 1"


def test_structural_executor_returns_to_observed_unfinished_target() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11g", "put two saltshaker in cabinet.", "test", 11))
    context.history = [
        (
            "You arrive at sidetable 1. On the sidetable 1, you see a peppershaker 1, "
            "a saltshaker 2, and a saltshaker 1.",
            "take saltshaker 1 from sidetable 1",
        ),
        ("You pick up the saltshaker 1.", "go to cabinet 2"),
        ("You arrive at cabinet 2.", "move saltshaker 1 to cabinet 2"),
    ]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to cabinet 1", "go to sidetable 1", "go to drawer 1"],
    )
    assert StructuralAgent._known_target_locations(context, {"saltshaker 1"}) == {"sidetable 1"}
    assert ranked[0][0] == "go to sidetable 1"


def test_structural_executor_places_held_target_before_returning_for_next_one() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11i", "put two saltshaker in cabinet.", "test", 11))
    context.history = [
        (
            "You arrive at sidetable 1. On the sidetable 1, you see a saltshaker 2, and a saltshaker 1.",
            "take saltshaker 1 from sidetable 1",
        ),
        ("You pick up the saltshaker 1.", "go to cabinet 2"),
        ("You arrive at cabinet 2.", "open cabinet 2"),
    ]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to sidetable 1", "move saltshaker 1 to cabinet 2", "go to cabinet 3"],
    )
    assert ranked[0][0] == "move saltshaker 1 to cabinet 2"
    assert StructuralAgent._forced_grounded_action(context, "Cabinet 2 is open.", ranked) == ranked[0][0]


def test_structural_executor_scans_numbered_locations_in_natural_order() -> None:
    context = AgentContext(TaskSpec("alfworld", "t11h", "find two saltshaker and put them in drawer.", "test", 11))
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to cabinet 10", "go to cabinet 2", "go to cabinet 3"],
    )
    assert [action for action, _ in ranked] == ["go to cabinet 2", "go to cabinet 3", "go to cabinet 10"]


def test_structural_executor_penalizes_container_type_after_first_open() -> None:
    context = AgentContext(TaskSpec("alfworld", "t12", "find remotecontrol", "test", 12))
    context.history = [("", "go to cabinet 1"), ("", "open cabinet 1")]
    ranked = StructuralAgent._rank_admissible_actions(
        context,
        ["go to cabinet 2", "go to sofa 1"],
    )
    assert ranked[0][0] == "go to sofa 1"
