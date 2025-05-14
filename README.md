# 📊 CSV Chat Agent using LangChain + Streamlit

This project allows you to **upload a CSV file** and **ask natural language questions** about your data using OpenAI's LLM — powered by [LangChain](https://github.com/langchain-ai/langchain) and [Streamlit](https://streamlit.io/).

## 🚀 Features

- 🧠 Built with `langchain_experimental.create_csv_agent` and OpenAI
- 📁 Upload CSV files (semicolon-separated)
- 💬 Ask natural language questions
- 📉 Automatically filters for relevant columns:
- 🖼️ Displays structured or text responses
- 📥 Option to download results as a CSV
- ⚠️ Dependency and API key checks

---

## 📦 Requirements

Install dependencies:

```bash
pip install streamlit langchain openai python-dotenv pandas tabulate
