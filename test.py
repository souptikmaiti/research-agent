import logging
import asyncio
from dotenv import load_dotenv
from research_agent.agentstack_reserach import HealthcareResearchAgent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)

async def test_agent():
    """Test the Healthcare Research Agent"""
    
    # Initialize agent
    logger.info("Initializing HealthcareResearchAgent...")
    agent = HealthcareResearchAgent(
        model="gemini-3-flash-preview",
        temperature=0.2
    )
    
    # Test queries
    test_queries = [
        "What are the common symptoms of diabetes?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test Query {i}: {query}")
        logger.info(f"{'='*60}")
        
        try:
            response = await agent.ask_async(
                query=query,
                user_id="test_user",
                session_id=f"test_session_{i}"
            )
            
            logger.info(f"\n✅ Response:\n{response}\n")
            
        except Exception as e:
            logger.error(f"\n❌ Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(test_agent())