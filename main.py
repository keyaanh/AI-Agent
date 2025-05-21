from langchain_experimental.agents import create_csv_agent
import streamlit as st
from streamlit_chat import message
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import pandas as pd
import importlib.util
import re
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.agent import AgentAction, AgentFinish
from langchain.agents.agent import AgentExecutor
from typing import Dict, List, Any, Optional, Union

def check_dependency(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None

def extract_filters(query, df_columns):
    """Extract filtering conditions from queries containing 'all'."""
    query_lower = query.lower()
    filters = []

    # Pattern 1: "in the '[value]' product group minor"
    match_product_group = re.search(r"in the ['\"](.+?)['\"] product group minor", query_lower)
    if match_product_group:
        value = match_product_group.group(1).upper()  # Match CSV case (e.g., 'SHOP')
        filters.append(("PrdGrpMinorVal", value))
        print(f"Extracted filter: {('PrdGrpMinorVal', value)}")  # Debug

    # Pattern 2: "with [column] of '[value]'"
    match_with = re.search(r"with (\w+) of ['\"](.+?)['\"]", query_lower)
    if match_with:
        column = match_with.group(1)
        value = match_with.group(2)
        if column in df_columns:
            filters.append((column, value))
            print(f"Extracted filter: {(column, value)}")  # Debug

    # Pattern 3: "[category] items" (e.g., "office items")
    match_category = re.search(r"all (.+?) items", query_lower)
    if match_category:
        category = match_category.group(1).strip().upper()
        if category in df_columns.get('PrdGrpMinorVal', set()):
            filters.append(("PrdGrpMinorVal", category))
            print(f"Extracted category filter: {('PrdGrpMinorVal', category)}")  # Debug

    return filters

def apply_filters(df, filters):
    """Apply the parsed filters to the DataFrame."""
    filtered_df = df.copy()
    for column, value in filters:
        filtered_df = filtered_df[filtered_df[column] == value]
    return filtered_df

class StreamlitAgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler that displays the agent's thought process in Streamlit."""
    def __init__(self, container):
        """Initialize with a Streamlit container to write output to."""
        self.container = container
        self.chain_starts = 0
        self.already_displayed_header = False
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Run when LLM starts running."""
        # Skip displaying "Thinking..." message to keep the output cleaner
        pass
    
    def on_llm_new_token(self, token, **kwargs):
        """Run on new LLM token. Only available when streaming is enabled."""
        pass
    
    def on_llm_end(self, response, **kwargs):
        """Run when LLM ends running."""
        pass
    
    def on_llm_error(self, error, **kwargs):
        """Run when LLM errors."""
        with self.container:
            st.error(f"LLM Error: {error}")
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Run when chain starts running."""
        # Only display the "Entering new AgentExecutor chain..." message once
        if not self.already_displayed_header:
            with self.container:
                st.markdown("**Agent started reasoning about your question...**")
            self.already_displayed_header = True
    
    def on_chain_end(self, outputs, **kwargs):
        """Run when chain ends running."""
        pass
    
    def on_chain_error(self, error, **kwargs):
        """Run when chain errors."""
        with self.container:
            st.error(f"Chain Error: {error}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        """Run when tool starts running."""
        pass
    
    def on_agent_action(self, action: AgentAction, **kwargs) -> Any:
        """Run on agent action."""
        with self.container:
            thought_text = action.log.split("Action:")[0].strip() if hasattr(action, "log") and action.log else ""
            if thought_text:
                st.markdown(f"**Thought:** {thought_text}")
            
            # Display the action and input
            st.code(f"Action: {action.tool}\nAction Input: {action.tool_input}", language="python")
    
    def on_tool_end(self, output, **kwargs):
        """Run when tool ends running."""
        with self.container:
            st.code(str(output), language="python")
    
    def on_tool_error(self, error, **kwargs):
        """Run when tool errors."""
        with self.container:
            st.error(f"Tool Error: {error}")
    
    def on_text(self, text, **kwargs):
        """Run on text."""
        # Don't display raw text callbacks to avoid redundant "Entering new AgentExecutor chain..." messages
        pass
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Run on agent end."""
        with self.container:
            st.markdown(f"**Final Answer:** {finish.return_values['output']}")

def main():
    load_dotenv()

    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        st.error("OPENAI_API_KEY is not set")
        return
    else:
        print("OPENAI_API_KEY is set")

    # Check for required dependencies
    if not check_dependency("tabulate"):
        st.error("Missing required dependency 'tabulate'. Please install it using 'pip install tabulate'.")
        return
    if not check_dependency("streamlit_chat"):
        st.error("Missing required dependency 'streamlit_chat'. Please install it using 'pip install streamlit_chat'.")
        return

    # Center the title and subtitle using custom CSS
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>Chat with CSV 📈</h1>
            <h3 style='color: white;'>Ask questions about your CSV data</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # File uploader in sidebar
    csv_file = st.sidebar.file_uploader("Upload your CSV", type="csv")

    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Upload a CSV and ask me anything about it 🤗"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

    def process_csv_and_query(query):
        """Process the CSV and run the user's query."""
        if csv_file is None:
            return "Please upload a CSV file to proceed."
        
        # Initialize variables that will be used in finally block
        tmp_path = None
        filtered_tmp_path = None
        
        try:
            # Save the uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(csv_file.read())
                tmp_path = tmp.name
            
            # Define the filtered temp path
            filtered_tmp_path = tmp_path.replace(".csv", "_filtered.csv")
            
            # Define relevant columns
            relevant_columns = [
                "Item", "Size", "Description", "Stocking UOM", "Costing UOM", "Extended Description",
                "Product Group Major", "Product Group Minor", "Price Code Major", "Price Code Minor"
            ]
            # Read the CSV with pandas
            df = pd.read_csv(tmp_path, sep=";", encoding="utf-8", on_bad_lines="skip")
            # Select only relevant columns
            df = df[relevant_columns]
            # Save the filtered CSV to a new temporary file
            filtered_tmp_path = tmp_path.replace(".csv", "_filtered.csv")
            df.to_csv(filtered_tmp_path, sep=";", index=False)

            # Create a Streamlit container for displaying the agent's thoughts
            thought_process_container = st.container()
            
            with thought_process_container:
                st.markdown("### Agent's Thinking Process")
                # Create the callback handler
                st_callback = StreamlitAgentCallbackHandler(thought_process_container)
                
                # Add a line to display thinking in a cleaner way
                st.info("Analyzing your query... results will appear below.")
                
                # Create the agent with the filtered CSV
                agent = create_csv_agent(
                    OpenAI(temperature=0),
                    filtered_tmp_path,
                    verbose=True,
                    pandas_kwargs={
                        "sep": ";",
                        "encoding": "utf-8",
                        "on_bad_lines": "skip"
                    },
                    allow_dangerous_code=True
                )
                
                # Run the query with the callback handler
                if isinstance(agent, AgentExecutor):
                    response = agent.run(query, callbacks=[st_callback])
                else:
                    # If agent is not an AgentExecutor, fall back to standard run
                    response = agent.run(query)
                    st.write("Agent's thinking process not available")

            # Load the filtered DataFrame for manual processing
            df_filtered = pd.read_csv(filtered_tmp_path, sep=";")[relevant_columns]

            # Get unique values for columns to assist filtering
            df_columns = {col: set(df_filtered[col].dropna().astype(str)) for col in relevant_columns}

            # Check if the query contains 'all'
            query_lower = query.lower()
            if "all" in query_lower:
                # Extract filters based on the query
                filters = extract_filters(query, df_columns)
                if filters:
                    filtered_df = apply_filters(df_filtered, filters)
                    if not filtered_df.empty:
                        st.table(filtered_df)
                        filter_desc = " and ".join([f"{col} of '{val}'" for col, val in filters])
                        output = f"Items with {filter_desc} are displayed in the table above."
                    else:
                        filter_desc = " and ".join([f"{col} of '{val}'" for col, val in filters])
                        output = f"No items found with {filter_desc}."
                else:
                    # Fallback: Use the agent's response
                    output = response
            else:
                output = response

            # Handle specific case for StockUOMVal == 'EA' with download option
            if "stockuomval" in query_lower and "'ea'" in query_lower:
                ea_items = df_filtered.query('StockUOMVal == "EA"')[["ItemVal", "ItemDescVal"]]
                st.table(ea_items.head(10))
                output = f"Found {len(ea_items)} items with StockUOMVal of 'EA'. Showing the first 10 in the table above."
                if len(ea_items) > 10:
                    csv = ea_items.to_csv(index=False)
                    st.download_button(
                        label="Download full list as CSV",
                        data=csv,
                        file_name="ea_items.csv",
                        mime="text/csv",
                        key=f"download_{query}"
                    )

            return output

        except Exception as e:
            return f"Error: {e}"
        finally:
            # Clean up temporary files
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if filtered_tmp_path and os.path.exists(filtered_tmp_path):
                os.unlink(filtered_tmp_path)

    # Containers for chat history and user input
    response_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Query:", placeholder="Talk to your CSV data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner("Processing..."):
                output = process_csv_and_query(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)
                st.session_state['history'].append((user_input, output))

    # Display chat history
    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

if __name__ == "__main__":
    main()