import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = "<YOUR_API_KEY>"

template = """
Based on the table schema below, write a SQL query to answer the question.
{schema}

Question: {question}
SQL Query:
"""

prompt = ChatPromptTemplate.from_template(template)
prompt.format(schema="my schema", question="how many users are there?")
"Human: \nBased on the table schema below, write a SQL query that would answer the user's question.\nmy schema\n\nQuestion: how many users are there?\nSQL Query:\n"


db_uri = "mysql+mysqlconnector://root:admin@localhost:3306/Chinook"
db = SQLDatabase.from_uri(db_uri)


def get_schema(_):
    return db.get_table_info()

llm = ChatOpenAI()

sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop="\nSQL Result:")
        | StrOutputParser()
)
sql_chain.invoke({"question": "how many artists are there?"})
'SELECT COUNT(*) AS TotalArtists\nFROM Artist;'
template = """
Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

prompt = ChatPromptTemplate.from_template(template)


def run_query(query):
    return db.run(query)


run_query("SELECT COUNT(ArtistId) AS TotalArtists FROM Artist;")
'[(275,)]'
full_chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=get_schema,
            response=lambda vars: run_query(vars["query"])
        )
        | prompt
        | llm
        | StrOutputParser()
)
full_chain.invoke({"question": "how many artists are there?"})
