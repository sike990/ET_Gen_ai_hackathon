# 🚀 Anti-Gravity Financial AI System

An intelligent, multi-agent financial command center powered by Google's **Gemini LLM, LangChain, LangGraph, Streamlit, and ChromaDB RAG**. 

It takes your personal financial context and unleashes 4 targeted AI specialist agents to build a highly personalized "Financial Flight Plan".

---

## 🛠 Features
- **Multi-Agent Workflow (`agents.py`)**: 
  - 🏦 **Savings Architect**: Calculates emergency fund gaps.
  - 🔥 **Debt Shredder**: Targets high-interest debt with Avalanche/Snowball.
  - 🛡️ **Insurance Expert**: Evaluates coverage and calculates precise risks.
  - 📈 **Investment Scout**: Aligns portfolio with the 100-Age Rule.
  - ✈️ **Final Planner**: Merges agent advice into a prioritized, budgeted timeline.
- **Dynamic Local RAG (`rag_setup.py`)**: Vector retrieval of PDFs and Text files locally via ChromaDB so agents stay highly knowledgeable.
- **Terminal Rate-Limit Handler**: Includes an automatic suspension wrapper that intercepts Gemini Free-Tier `429 RESOURCE_EXHAUSTED` quotas and waits dynamically instead of crashing.
- **Streamlit UI (`app.py`)**: A single-page dashboard with spider-charts, dynamic health scoring (`scoring_engine.py`), and a conversational memory interface.

---

## ⚙️ Quick Start Installation

1. **Activate your Python Environment**
   ```bash
   python -m venv et
   .\et\Scripts\activate
   ```

2. **Install Dependencies**
   Install all the packages required to run the AI system:
   ```bash
   pip install streamlit langchain langchain-google-genai langgraph chromadb pypdf python-dotenv matplotlib
   ```

3. **Configure Environment Variables**
   Create a `.env` file in your root folder and paste in your Gemini key:
   ```env
   GOOGLE_API_KEY=AIzaSy_your_gemini_api_key_here
   ```
   *(You can get a free api key from [Google AI Studio](https://aistudio.google.com/).)*

4. **Populate the Knowledge Base**
   Ensure any PDF or text files representing financial knowledge are placed in the `knowledge_base/` folder. The app will ingest them on the first run automatically.

---

## 🏃‍♂️ How to Run

Launch the Streamlit dashboard by running:

```bash
streamlit run app.py
```

### Usage Instructions
1. A browser window will open automatically at `http://localhost:8501`.
2. Enter your financial details (income, saved cash, debts) in the **Sidebar Forms**.
3. Click the **🚀 Analyze My Finances** button.
4. **Watch your terminal!** The console will log exactly which agent is currently "thinking", and will automatically pause for ~60s if the Google API rate-limits the execution.
5. In the browser, you will receive a comprehensive, graded Financial Flight Plan dynamically assembled by your AI team.
