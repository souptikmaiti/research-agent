import logging
import uvicorn
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        """Initialize the Healthcare Research Agent.
        
        Args:
            model: The Gemini model to use
            temperature: Temperature for response generation (0.0-1.0)
        """
        self.agent = LlmAgent(
            model=model,
            name="HealthcareResearchAgent",
            description="A healthcare research assistant that provides reliable information about diseases, symptoms, and health conditions.",
            instruction=SYSTEM_PROMPT,
            generate_content_config=GenerateContentConfig(temperature=temperature),
            tools=[google_search],
        )
        
        # Initialize session service and runner
        self.session_service = InMemorySessionService()
        self.runner = Runner(agent=self.agent, app_name="healthcare", session_service=self.session_service)
        self.session = None
        
        logger.info(f"Initialized HealthcareResearchAgent with model={model}, temperature={temperature}")
    
    async def initialize_session(self, user_id: str = "user", session_id: str = "default"):
        """Initialize a session for the agent.
        
        Args:
            user_id: User identifier
            session_id: Session identifier
        """
        self.session = await self.session_service.create_session(
            app_name="healthcare",
            user_id=user_id,
            session_id=session_id
        )
        logger.info(f"Session initialized: user_id={user_id}, session_id={session_id}")

    async def ask(self, query: str, user_id: str = "user", session_id: str = "default") -> str:
        """Ask the agent a healthcare-related question.
        
        Args:
            query: The user's question
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            The agent's final response text
        """
        # Initialize session if not already done
        if not self.session:
            await self.initialize_session(user_id, session_id)
        
        logger.info(f"Received query: {query}")
        
        content = types.Content(role="user", parts=[types.Part(text=query)])
        final_response_text = "No final text response captured."
        
        try:
            async for event in self.runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content
            ):
                logger.info(f"Event ID: {event.id}, Author: {event.author}")
                
                # Check for specific parts
                has_specific_part = False
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.executable_code:
                            logger.debug(f"Agent generated code:\n{part.executable_code.code}")
                            has_specific_part = True
                        elif part.code_execution_result:
                            logger.debug(f"Code Execution Result: {part.code_execution_result.outcome} - Output:\n{part.code_execution_result.output}")
                            has_specific_part = True
                        elif part.text and not part.text.isspace():
                            logger.debug(f"Text: '{part.text.strip()}'")
                
                # Check for final response
                if not has_specific_part and event.is_final_response():
                    if event.content and event.content.parts and event.content.parts[0].text:
                        final_response_text = event.content.parts[0].text.strip()
                        logger.info(f"Final Agent Response: {final_response_text}")
                    else:
                        logger.warning("Final Agent Response: [No text content in final event]")
        
        except Exception as e:
            logger.error(f"ERROR during agent run: {e}")
            raise
        
        return final_response_text

    

server = Server()

@server.agent(
    name="ResearchAgent",
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
                "I have a white rash and no feeling in a leg, who should I see?",
                "I have a rash and no fever, what kind of doctor should I see?",
            ]
        ),
    ],
)
async def google_search_agent(
    input: Message,
    context: RunContext,
    trajectory: Annotated[TrajectoryExtensionServer, TrajectoryExtensionSpec()],
    citation: Annotated[CitationExtensionServer, CitationExtensionSpec()],
    llm: Annotated[
        LLMServiceExtensionServer, 
        LLMServiceExtensionSpec.single_demand(
            suggested=("google/gemini-2.0-flash-exp",)
        )
    ],
):
    """Healthcare research assistant"""
    
    # Record the incoming message to context history
    await context.store(input)

    prompt = get_message_text(input)

    if not prompt:
        yield AgentMessage(text="No input provided.")
        return
    
    # Record the incoming message to trajectory
    yield trajectory.trajectory_metadata(title="User Query", content=f"Received: '{prompt}'")

    # Select the default LLM fulfillment from the extension
    if llm and llm.data and llm.data.llm_fulfillments:
        llm_config = llm.data.llm_fulfillments.get("default")
    else:
        yield AgentMessage(text="LLM selection is required.")
        return

    # Ensure we have a valid LLM config to create the client
    if not llm_config:
        yield AgentMessage(text="No LLM configuration available from the extension.")
        return
    
    yield trajectory.trajectory_metadata(title="Agent Setup", content="Initializing RequirementAgent with Google search")
    
    # Create agent instance with the selected model
    agent = HealthcareResearchAgent(model=llm_config.api_model, temperature=0.2)
    
    # Get unique session ID from context
    session_id = context.run_id or "default"
    user_id = "user"
    
    # Ask the agent and stream the response
    response = await agent.ask(prompt, user_id=user_id, session_id=session_id)
    yield AgentMessage(text=response)

def run():
    server.run(host="127.0.0.1", port=8001, context_store=PlatformContextStore())
