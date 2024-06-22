# Agent 1: Product Price Checker
from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
import os


openai_api_key = "Your OpenAI API"
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = "your SERPER_API_KEY

"

# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
search = Agent(
    role="Product Price Checker",
    goal="Find and compare prices of a specific product across multiple e-commerce websites.",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        """The Product Price Checker agent is designed to help users find the best deals for their desired products.
        Upon receiving a product query from the user, the agent searches for the product on various e-commerce websites.
        It then gathers and compares the prices, providing a comprehensive report on the availability and cost of the product across different platforms.
        This helps users make informed purchasing decisions and ensures they get the best possible price."""
    )
)
# Agent 2: Price Comparison Specialist
comparison = Agent(
    role='Price Comparison Specialist',
    goal=(
        "Compare the prices of a specific product across all e-commerce websites and return the website offering the lowest price, adjusted to the specified country's currency."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=True,
    backstory=(
        """The Price Comparison Specialist agent is designed to ensure users get the best deal on their desired products.
        After receiving a list of product prices from various e-commerce websites, the agent compares these prices and converts them into the user's specified currency.
        It then identifies the most trusted website offering the lowest price for the product, helping users make cost-effective purchasing decisions."""
    )
)
search_task = Task(
    description="Find out the {product}",
    expected_output="All the details of a specifically chosen product, including its availability and prices on various e-commerce websites.",
    human_input=True,
    agent=search
)
comparison_task = Task(
    description="Compare prices for {product} across different e-commerce websites and find the lowest price adjusted to {country}'s currency.",
    expected_output="The website with the lowest price for the product, adjusted to the specified country's currency.",
    human_input=True,
    agent=comparison
)
# Define the crew with agents and tasks
event_management_crew = Crew(
    agents=[search, 
            comparison],    
    tasks=[search_task, 
           comparison_task],
    
    verbose=True
)
event_details = {
    'product': "lenovo earbuds lp40",
    'country': "Pakistan"
    
}
result = event_management_crew.kickoff(inputs=event_details)
