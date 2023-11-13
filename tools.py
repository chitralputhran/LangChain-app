
from langchain.llms import OpenAI
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.utilities import PythonREPL
from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool


def initialize_tools():
    # Initializing tools
    search = DuckDuckGoSearchAPIWrapper()
    wikipedia = WikipediaAPIWrapper()
    python_repl = PythonREPL()
    yahoo_finance_news = YahooFinanceNewsTool()
    
    return search, wikipedia, python_repl, yahoo_finance_news

def get_tools():
    search, wikipedia, python_repl, yahoo_finance_news = initialize_tools()
    
    tools = [
        Tool(
            name="Online Search",
            func=search.run,
            description="Useful for when you need to answer questions about current events.",
        ),
        Tool(
            name = "Python Interpreter",
            func=python_repl.run,
            description="Useful for when you need to use Python to answer a question."
        ), 
        Tool(
            name = "Wikipedia Search",
            func=wikipedia.run,
            description="Useful for when you need to answer questions about a topic, country, person from Wikipedia."
        ), 
        Tool(
            name = "Yahoo Finance News Search",
            func=yahoo_finance_news.run,
            description="Useful for when you need to answer questions about stocks and news from Yahoo Finance."
        )
    ]
    return tools