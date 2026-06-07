# Multi-Agent-Clinical-Workflow-Engine

This system was designed, engineered, and deployed as a comprehensive Graduate Capstone Project to demonstrate senior-level competencies in multi-agent orchestration systems, LLM state management, and production-grade data pipelines.

An enterprise-grade, stateful multi-agent healthcare assistant engineered to autonomously coordinate complex patient care pipelines. Developed as a capstone project utilizing LangGraph, LangChain, and OpenAI GPT-4o-mini, the workflow dynamically interprets multi-step natural language queries, maps relational medical entities via SQLite3, performs high-precision semantic searches over unstructured Electronic Health Records (EHR) using FAISS, and grounds clinical summaries using official National Library of Medicine (NLM) APIs.

## Executive Project Overview

### The Challenge
Modern digital health ecosystems suffer from deep operational friction due to siloed relational databases, unstructured PDF health charts, and disconnected external medical literature. Manually coordinating appointment bookings, synthesizing medical records, and validating clinical treatments leads to massive administrative burnout and highly fragmented patient experiences.

### The Solution
A multi-agent system built on a unified directed acyclic graph (DAG) state machine. The system abstracts engineering silos into **six specialized agent personas** that handle real-time slot discovery, EHR vector indexing, and XML research parsing via a centralized memory state—eliminating manual clinical routing.

---

## Core Architecture & Dataflow

```
[User Query] ──> (Patient Finder) ──> (The Planner)
│
┌──────────────────────┼──────────────────────┐
▼                      ▼                      ▼
(The Scheduler)     (Medical History RAG)     (Research Agent)
[SQLite3 Relational]   [FAISS Vector DB]       [NLM Medline API]
│                      │                      │
└──────────────────────┼──────────────────────┘
▼
(The Aggregator) ──> [Streamlit UI Dashboard]

```
### Logic Flow Orchestration & State Machine
* **LangGraph DAG Routing:** Implements conditional edge evaluations rather than fragile, linear prompt chains, guaranteeing structured state updates across nodes.
* **Shared Memory State (`PATIENTREQUEST_STATE`):** Employs an append-only transaction state where specialized agents iteratively dump vectorized contexts, execution payloads, and lookup parameters.

```
class PatientRequest_State(TypedDict):
  query: str
  category: str
  task_list: str
  patient: list[dict]
  category: list[dict]
  appointment: list[dict]
  document: list[dict]
  medical_history_summary: str
  qa_evaluation: str
  per_evaluation: str
  research_summary: str
  final_response: str
  error_message: str
```

---

## Specialized Agent Microservices

### 1. Agent 1: The Patient Finder (Demographics Lookup)
* Extracts core patient parameters directly from unstructured strings to perform zero-shot deterministic relational lookups inside an SQLite3 instance.

```
def patientfinder_agent(state):
  # Determine criteria: Name, ID, Relationship
  sql_query = ttm.invoke(messages).content
  results = get_db_results(sql_query) # Return person_id and name for state
  state['patient'] = results
  return state
```
### 2. Agent 2: The Planner (The Orchestration Core)
* Evaluates complex multi-step queries (e.g., matching a patient's age and diagnosis to a specialist) and decomposes them into explicit `APPOINTMENT` or `MEDICAL_RECORD` task lists.

```
def planner_agent(state):
  # System classifies intent into task
  list response = ttm.invoke(ctassification_prompt)
  category_dict = json.loads(response.content)
  state['category'] = category_dict
  return state
```

### 3. Agent 3: The Scheduler (Relational Slot Discovery)
* Employs schema-injected prompting to programmatically generate syntactically correct SQL queries over the `Doctor Schedule Schema` to claim open booking slots.

```
def scheduler_agent(state):
  # Use person_id from state to query schedule
  sql_query = construct_sql(state['query'])
  results = get_db_resutls(sql_query)
  # Store appointment results metadata in state
  state(['appointment'] = results
  return state
```
### 4. Agent 4: Medical History Agent (FAISS RAG Pipeline)
* Orchestrates semantic search pipelines over vectorized patient history charts. Generates text chunking arrays using `OpenAI Embeddings` and caches indices locally via `FAISS`.

```
def medical_history_agent(state):
  # Load PDF, Split into chunks
  vectorstore = FAISS.from_documents(chunks, emb)
  retriever = vectorstore.as_retriever(k=3)
  # Identify primary disease via RAG chain
  result = rag_chain.invoke({})
  state ["medical_history_summary] = result
  return state
```

### 5. Agent 5: The Research Agent (API Grounding Engine)
* Eliminates clinical hallucinations by executing live web requests against the `NLM Medline HealthTopics API`, parsing dense XML dataframes with `BeautifulSoup` to anchor answers in verified facts.

```
def research_agent(state):
  url = "ntm.nih.gov/ws/query"
  response = requests.get(url, diag})
  # Parse Full Summary from XML with BeautifutSoup
  full_summary = get_first_full_summary(response.text)
  state ('research_summary'] = full_summary
  return state
```

### 6. Agent 6: The Aggregator (Clinical Synthesizer)
* Collects parallel asynchronous payloads from the shared state (SQL execution logs, FAISS RAG context windows, and XML research nodes) to render an un-hallucinated, patient-friendly diagnostic brief.

```
def research_agent(state):
  url = "nlm.nih.gov/ws/query"
  response = requests.get(url, diag})
  # Parse Full Summary from XML with BeautifutSoup
  full_summary = (response.text)
  state ([research_summary] = full_summary
  return state
```

---

## Technical Tool Integration Stack

* **Orchestration Engine:** LangGraph (Stateful Directed Graphs).
* **LLM Framework & Embeddings:** LangChain / OpenAI GPT-4o-mini & Text-Embedding-3.
* **Vector Architecture:** FAISS (Facebook AI Similarity Search) for high-precision EHR lookups.
* **Relational Layer:** SQLite3 with pre-configured schemas for provider availability.
* **External Clinical Trust:** NLM Medline HealthTopics API (Live XML Scraper / BeautifulSoup Parsing).
* **Frontend Delivery:** Streamlit UI supporting dual Patient/Doctor views.

---

## LLMOps: Monitoring, Evaluation & Dashboard

The platform exposes a bifurcated **Streamlit Dashboard UI** that provides end-to-end data visualization and model evaluation matrices:

### Patient Portal Interface
* **Natural Language Queries:** Allows inputting compound prompts (e.g., *"Book a nephrologist for my 70-year-old father with chronic kidney disease and summarize his treatment methods"*).
* **Real-time Summaries:** Provides instantaneous delivery of booking block structures and medical risk factors.

### Doctor & Admin Performance Dashboard
* **Clinical Transparency Ledger:** Side-by-side comparative viewport detailing local internal patient histories alongside external Medline data matrices.
* **Agent Execution Tracing:** High-granularity execution tracking highlighting individual agent node success/failure rates (e.g., SQL execution accuracy vs FAISS vector hit rates).
* **Memory Trace Audit Logs:** Deep inspection windows showcasing the underlying LLM planning breakdowns and state memory mutations for compliance and logging.

