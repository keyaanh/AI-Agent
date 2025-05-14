from langchain_experimental.agents import create_csv_agent
import streamlit as st
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import pandas as pd
import importlib.util

def check_dependency(module_name):
    """Check if a Python module is installed."""
    return importlib.util.find_spec(module_name) is not None

def main():
    load_dotenv()

    # Check for OpenAI API key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # Check for required dependencies
    if not check_dependency("tabulate"):
        st.error("Missing required dependency 'tabulate'. Please install it using 'pip install tabulate'.")
        return

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(csv_file.read())
            tmp_path = tmp.name

        # Preprocess CSV to select relevant columns
        try:
            # Define relevant columns
            relevant_columns = [
                "ItemVal", "ItemDescVal", "StockUOMVal", "ItemWeightVal",
                "PrdGrpMinorVal", "ActiveVal"
            ]
            # Read the CSV with pandas
            df = pd.read_csv(tmp_path, sep=";", encoding="utf-8", on_bad_lines="skip")
            # Select only relevant columns
            df = df[relevant_columns]
            # Save the filtered CSV to a new temporary file
            filtered_tmp_path = tmp_path.replace(".csv", "_filtered.csv")
            df.to_csv(filtered_tmp_path, sep=";", index=False)

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

            user_question = st.text_input("Ask a question about your CSV: ")
            if user_question and user_question.strip():
                with st.spinner(text="In progress..."):
                    # Run the query
                    response = agent.run(user_question)
                    # Display results
                    try:
                        # Check if the question is about StockUOMVal == 'EA'
                        if "StockUOMVal" in user_question and "'EA'" in user_question:
                            # Re-run the query to get the DataFrame
                            df_response = pd.read_csv(filtered_tmp_path, sep=";").query('StockUOMVal == "EA"')[["ItemVal", "ItemDescVal"]]
                            # Display the first 10 rows (adjustable)
                            st.write(f"Found {len(df_response)} items with StockUOMVal of 'EA'. Showing the first 10:")
                            st.dataframe(df_response.head(10))
                            # Option to download full results
                            if len(df_response) > 10:
                                csv = df_response.to_csv(index=False)
                                st.download_button(
                                    label="Download full list as CSV",
                                    data=csv,
                                    file_name="ea_items.csv",
                                    mime="text/csv"
                                )
                        else:
                            # Handle other queries
                            if isinstance(response, (pd.Series, pd.DataFrame)):
                                st.dataframe(response.head(10))
                            else:
                                st.write(response)
                    except Exception as e:
                        st.write(response)
                        st.warning(f"Could not display results as a table: {e}")
        except Exception as e:
            st.error(f"Error processing CSV: {e}")
        finally:
            # Clean up temporary files
            os.unlink(tmp_path)
            if os.path.exists(filtered_tmp_path):
                os.unlink(filtered_tmp_path)

if __name__ == "__main__":
    main()