CSV AI Reader
Overview
The CSV AI Reader is a Streamlit-based web application that allows users to upload a CSV file and query its contents using plain English questions. Powered by LangChain's create_csv_agent and an AI language model, the app translates natural language into pandas queries, displaying results as interactive tables. It's designed for general data exploration, enabling users to filter, summarize, or analyze tabular data without writing code.
Code Functionality
The Python script (app.py) provides the following core functions:

CSV Upload:

Accepts a CSV file via Streamlit's file uploader.
Loads the CSV using pandas for querying.


AI-Powered Querying:

Uses langchain_experimental.agents.create_csv_agent to create an agent that interprets plain English questions.
Leverages an AI language model (configured with zero temperature for consistent responses) to convert questions into pandas operations (e.g., filtering rows, counting values, calculating averages).
Supports basic parsing of direct pandas queries (e.g., df[df['column'] == 'value']), though natural language is recommended.


Result Display:

Renders query results as tables in Streamlit using st.dataframe, typically limited to 10 rows for clarity.
Offers a download button to save full result sets as CSV files for larger queries.
Handles common query types (e.g., filtering by column values) with optimized logic to ensure accurate table output.


Error Handling:

Detects and processes pandas queries using Python's ast module.
Catches errors from invalid queries or processing issues, displaying user-friendly messages.
Cleans up temporary files to manage disk space.


Debugging:

Enables verbose logging to print executed pandas queries and results in the terminal.
Supports optional debug output in the web interface for inspecting raw responses.



Dependencies

Python 3.8 or higher
langchain, langchain_experimental: For AI-driven CSV querying.
openai: For the language model.
pandas: For data processing.
tabulate: For table formatting.
streamlit: For the web interface.
python-dotenv: For environment variable management.

Install with:
pip install langchain langchain_experimental openai pandas tabulate streamlit python-dotenv

Usage

Configure the Environment:

Set an AI API key (e.g., OpenAI) as an environment variable (OPENAI_API_KEY) in a .env file:OPENAI_API_KEY=your-api-key




Run the App:
streamlit run app.py

Launches the web interface at http://localhost:8501.

Query the CSV:

Upload any CSV file with structured data (e.g., columns for IDs, descriptions, categories).
Enter a plain English question in the text input (e.g., "List items in category X").
View results as a table, with an option to download the full data.
Check the terminal for query details if verbose mode is enabled.



Example Questions
These generic questions show what the app can do:

"List the first 10 items in category 'X' with their IDs and names."
Filters rows by a category column and shows specific fields.


"How many items have a status of 'Active'?"
Counts rows matching a condition.


"Show items with descriptions containing 'keyword'."
Searches for a term in a text column.


"What is the average price of items in group 'Y'?"
Computes a numerical summary for a subset.


"List the first 5 active items with unit 'Each'."
Combines multiple filters for targeted results.



Security Considerations

Code Execution Risk: The app uses allow_dangerous_code=True for pandas query execution, which could be exploited with malicious inputs.
Best Practices:
Run in a sandboxed environment (e.g., Docker).
Validate CSV files and user inputs.
Monitor queries via terminal logs.
Use predefined queries in production for safety.



Troubleshooting

No Results Shown:
Ensure questions are in plain English, not code.
Check terminal logs for query details (verbose mode).
Add debug output in app.py:st.write(f"Raw response: {response}")




Dependency Errors:
Install missing packages, e.g., pip install tabulate.


API Key Issues:
Verify OPENAI_API_KEY in .env or environment.


Large Outputs:
Use "first 5" or "first 10" in questions to limit results.
Download full results via the button.


Unexpected Results:
Share the question and terminal output for debugging.



