"""
The state machine pattern describes workflows where an agent’s behavior changes as it moves
through different states of a task. The state can be determined from multiple sources: the
agent’s past actions (tool calls), external state (such as API call results), or even initial
user input (for example, by running a classifier to determine user intent).

The state machine pattern uses a single agent whose configuration changes based on workflow
progress. Each “step” is just a different configuration (system prompt + tools) of the same
underlying agent, selected dynamically based on state.
"""


from typing import Callable, Literal

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, SummarizationMiddleware, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing_extensions import NotRequired

load_dotenv()


# Define the possible workflow steps
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]

model = init_chat_model("openai:gpt-5-nano-2025-08-07")


class SupportState(AgentState):
    """State for customer support workflow."""

    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    # In a real system, this would create a ticket, notify staff, etc.
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"


@tool
def go_back_to_warranty(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to warranty verification step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Success",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "warranty_collector",
        }
    )


@tool
def go_back_to_classification(runtime: ToolRuntime[None, SupportState]) -> Command:
    """Go back to issue classification step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="Success",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "current_step": "issue_classifier",
        }
    )


# Define prompts as constants for easy reference
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Warranty verification

At this step, you need to:
1. Greet the customer warmly
2. Ask if their device is under warranty
3. Use record_warranty_status to record their response and move to the next step

Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Issue classification
CUSTOMER INFO: Warranty status is {warranty_status}

At this step, you need to:
1. Ask the customer to describe their issue
2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
3. Use record_issue_type to record the classification and move to the next step

If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

If the customer indicates any information was wrong, use:
- go_back_to_warranty to correct warranty status
- go_back_to_classification to correct issue type

Be specific and helpful in your solutions."""


# Step configuration: maps step name to (prompt, tools, required_state)
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human, go_back_to_warranty, go_back_to_classification],
        "requires": ["warranty_status", "issue_type"],
    },
}


@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "warranty_collector")

    # Look up step configuration
    stage_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values (supports {warranty_status}, {issue_type}, etc.)
    system_prompt = stage_config["prompt"].format(**request.state)

    # Inject system prompt and step-specific tools
    request = request.override(
        system_prompt=system_prompt,
        tools=stage_config["tools"],
    )

    return handler(request)


# Collect all tools from all step configurations
all_tools = [record_warranty_status, record_issue_type, provide_solution, escalate_to_human, go_back_to_warranty, go_back_to_classification]

# Create the agent with step-based configuration
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config, SummarizationMiddleware(model="gpt-4.1-mini", trigger=("tokens", 4000), keep=("messages", 10))],
    checkpointer=InMemorySaver(),
)


if __name__ == "__main__":
    import uuid

    from langchain.messages import HumanMessage

    # Configuration for this conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Turn 1: Initial message - starts with warranty_collector step
    print("=== Turn 1: Warranty Collection ===")
    result = agent.invoke({"messages": [HumanMessage("Hi, my phone screen is cracked")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()

    # Turn 2: User responds about warranty
    print("\n=== Turn 2: Warranty Response ===")
    result = agent.invoke({"messages": [HumanMessage("Yes, it's still under warranty")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"Current step: {result.get('current_step')}")

    # Turn 3: User describes the issue
    print("\n=== Turn 3: Issue Description ===")
    result = agent.invoke({"messages": [HumanMessage("The screen is physically cracked from dropping it")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()
    print(f"Current step: {result.get('current_step')}")

    # Turn 4: Go back
    print("\n=== Turn 4: Go Back ===")
    result = agent.invoke({"messages": [HumanMessage("Actually, I made a mistake - my device is out of warranty")]}, config)
    for msg in result["messages"]:
        msg.pretty_print()

    # Turn 4: Resolution
    # print("\n=== Turn 4: Resolution ===")
    # result = agent.invoke({"messages": [HumanMessage("What should I do?")]}, config)
    # for msg in result["messages"]:
    #     msg.pretty_print()
