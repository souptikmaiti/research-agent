import logging
import uvicorn
import asyncio
import sys
import os
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig
from google.adk.tools import google_search
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentstack_sdk.server.context import RunContext
from agentstack_sdk.server.store.platform_context_store import PlatformContextStore
from agentstack_sdk.a2a.extensions.ui.agent_detail import EnvVar
from agentstack_sdk.a2a.types import AgentMessage
from agentstack_sdk.a2a.extensions import (
    AgentDetail, AgentDetailTool,
    CitationExtensionServer, CitationExtensionSpec, 
    TrajectoryExtensionServer, TrajectoryExtensionSpec, 
    LLMServiceExtensionServer, LLMServiceExtensionSpec
)
from agentstack_sdk.server import Server
from a2a.types import AgentSkill
from typing import Annotated
from a2a.types import Message, Part
from a2a.utils.message import get_message_text


load_dotenv()

# Configure logging
# Configure logging to output to stdout
logger = logging.getLogger("healthcare-research-agent")
# set root logger
logging.getLogger().setLevel(logging.DEBUG)

SYSTEM_PROMPT = """You are a healthcare research assistant. Your role is to:

1. Search for reliable information about diseases, symptoms, and health conditions
2. Provide clear, evidence-based insights
3. Identify common symptoms and warning signs
4. Recommend which type of medical specialist to consult
5. Suggest general health recommendations

Always emphasize that users should consult qualified healthcare professionals for diagnosis and treatment.
Be concise, accurate, and empathetic in your responses."""


class HealthcareResearchAgent:
    def __init__(self, model: str = "gemini-3-flash-preview", temperature: float = 0.2):
        """Initialize the Healthcare Research Agent."""

        # Strip 'gemini:models/' prefix if present
        if model.startswith("gemini:models/"):
            model = model.replace("gemini:models/", "models/")
        elif model.startswith("gemini:"):
            model = model.replace("gemini:", "")

        self.agent = LlmAgent(
            model=model,
            name="HealthcareResearchAgent",
            description="A healthcare research assistant that provides reliable information about diseases, symptoms, and health conditions.",
            instruction=SYSTEM_PROMPT,
            generate_content_config=GenerateContentConfig(temperature=temperature),
            tools=[google_search],
        )
        
        logger.info(f"Initialized HealthcareResearchAgent with model={model}, temperature={temperature}")
    
    async def ask_async(self, query: str, user_id: str = "user", session_id: str = "default") -> str:
        """Ask the agent a healthcare-related question (async version)."""
        session_service = InMemorySessionService()
        
        # Create session asynchronously
        await session_service.create_session(
            app_name="healthcare",
            user_id=user_id,
            session_id=session_id
        )
        
        runner = Runner(agent=self.agent, app_name="healthcare", session_service=session_service)

        logger.info(f"Received query: {query}")
        
        content = types.Content(role="user", parts=[types.Part(text=query)])
        
        try:
            # Use run_async instead of run
            events_async = runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            )
            
            async for event in events_async:
                logger.debug(f"Event: {event}")
                
                if event.is_final_response() and event.content:
                    final_answer = event.content.parts[0].text.strip()
                    logger.info(f"Final Answer: {final_answer}")
                    return final_answer
        
        except Exception as e:
            logger.error(f"ERROR during agent run: {e}", exc_info=True)
            raise
        
        return "No response generated."



server = Server()

@server.agent(
    name="HealthcareResearchAgent",
    detail=AgentDetail(
        interaction_mode="multi-turn",
        # Add enviorment variables through the CLI so the agent has the credentials it needs to use its tools
        variables=[
            EnvVar(
                name="GOOGLE_API_KEY",
                description="Google API Key",
                required=True
            )
        ],
        user_greeting="Hi! I'm a Health Research Agent using Google search.",
        # version="1.0.0",
        tools=[
            AgentDetailTool(
                name="Google Search", 
                description="Intelligent web search powered by Google Search. Automatically extracts optimal search terms from conversational queries."
            )
        ],
        framework="google-adk"
    ),
    skills=[
        AgentSkill(
            id="google-search-agent",
            name="Google Search Agent",
            description="Provides healthcare information about symptoms, health conditions, treatments, and procedures using up-to-date web resources.",
            tags=["Search", "Web", "Research"],
            examples=[
                "My shoulder is swollen and my arm hurts to move, what doctor should I see?",
                "What are the common symptoms of diabetes?",
                "I have a rash and no fever, what kind of doctor should I see?",
            ]
        ),
    ],
)
async def healthcare_research_wrapper(
    input: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(
            suggested=("google/gemini-3-flash-preview",)
        )
    ],
):
    """Healthcare research assistant"""
    
    await context.store(input)
    prompt = get_message_text(input)
    logger.info(f"Received prompt: {prompt}")
    print(f"Received prompt: {prompt}")

    if not prompt:
        yield AgentMessage(text="No input provided.")
        return
    
    yield trajectory.trajectory_metadata(title="User Query", content=f"Received: '{prompt}'")

    if llm and llm.data and llm.data.llm_fulfillments:
        llm_config = llm.data.llm_fulfillments.get("default")
    else:
        yield AgentMessage(text="LLM selection is required.")
        return

    if not llm_config:
        yield AgentMessage(text="No LLM configuration available from the extension.")
        return
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        yield AgentMessage(text="GOOGLE_API_KEY is required.")
        return
    
    yield trajectory.trajectory_metadata(title="Agent Setup", content="Initializing HealthcareResearchAgent with Google search")
    
    agent = HealthcareResearchAgent(model=llm_config.api_model, temperature=0.2)
    
    session_id = context.context_id or "default"
    user_id = "user"
    
    # Use async version
    response = await agent.ask_async(prompt, user_id=user_id, session_id=session_id)
    yield AgentMessage(text=response)

def run():
    server.run(
        host="0.0.0.0", 
        port=8000, 
        context_store=PlatformContextStore(),
    )
