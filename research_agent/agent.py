import logging
import uvicorn
from dotenv import load_dotenv
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents import LlmAgent
from google.genai.types import GenerateContentConfig
from google.adk.tools import google_search

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

research_agent = LlmAgent(
    model="gemini-3-flash-preview",
    name="HealthcareResearchAgent",
    description="A healthcare research assistant that provides reliable information about diseases, symptoms, and health conditions. It can search for information, identify common symptoms and warning signs, recommend medical specialists, and suggest general health recommendations.",
    instruction=SYSTEM_PROMPT,
    generate_content_config=GenerateContentConfig(temperature=0.2),
    tools=[google_search],
)

def run_research_agent():
    app = to_a2a(research_agent, host="0.0.0.0", port=8001)
    print("Healthcare Research Agent is running on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    run_research_agent()
