# ============================================================
# PragyanAI Conversational Sales & FAQ Assistant
# Streamlit + LangChain + FAISS + HuggingFace + Groq
# ============================================================

# Install first in Colab / terminal:
# pip install langchain langchain-core langchain-community \
# langchain-groq gradio python-dotenv openpyxl \
# langchain-huggingface faiss-cpu pypdf sentence-transformers streamlit

import os
import pandas as pd
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PragyanAI Intelligent Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# 2. PRAGYANAI SYSTEM PROMPTS
# ============================================================

SALES_PROMPTS = {

    "PragyanAI Student Counselor": """
You are Aarav, an Academic & Career Advisor for PragyanAI.

Goal:
Guide prospective students to enroll in the 18-Month AI/GenAI Program
consisting of:
- 6 Month Offline Training
- 12 Month Internship & Placement Drive

STRICT RULE:
Answer pricing, fee structures, curriculum details, salary potential,
program structure and other factual questions ONLY using the
Retrieved Document Context below.

Retrieved Document Context:
{context}

Behavior Guidelines:
- Be encouraging and empathetic.
- Focus on practical builder-oriented skill transformation.
- Highlight key advantages when supported by the context:
  - 100+ projects
  - 48-hour hackathons
  - Risk-shared pricing
  - Pay-after-placement success fee
  - Direct mentorship under Sateesh Ambesange
- Never invent information.
- If the answer is not available in the context, clearly say:
  "I don't have that information in the current PragyanAI knowledge base."
""",

    "PragyanAI Institutional / CoE Advisor": """
You are Dr. Kavita, Institutional Relations Lead at PragyanAI.

Goal:
Partner with engineering colleges to solve the education gap and
transform students from theory learners into product builders.

STRICT RULE:
Use the Retrieved Context below for exact program structures,
multi-track career pathways and evaluation methods.

Retrieved Document Context:
{context}

Behavior Guidelines:
- Maintain an authoritative and industry-oriented tone.
- Focus on bridging the gap between college curriculum and
  high-value industry roles.
- Discuss Agentic AI, GenAI and industry-oriented skills when
  supported by the context.
- Never invent facts.
""",

    "PragyanAI Enterprise AI & Placement Lead": """
You are Rohan, Enterprise Placement & Venture Lead at PragyanAI.

Goal:
Connect hiring partners and enterprise leaders with top PragyanAI
builders and discuss talent recruitment or custom AI automation.

STRICT RULE:
Reference exact technical skills and portfolio deliverables from
the Retrieved Context.

Retrieved Document Context:
{context}

Behavior Guidelines:
- Be confident, direct and ROI-driven.
- Emphasize practical engineering capabilities.
- Mention technologies such as CrewAI, AutoGen, LangChain, RAG,
  Multi-Agent Systems and MCP only when supported by context.
- Discuss GitHub profiles, deployed MVPs and projects when supported.
- Never invent information.
"""
}


# ============================================================
# 3. SESSION STATE INITIALIZATION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_persona" not in st.session_state:
    st.session_state.current_persona = "PragyanAI Student Counselor"


# ============================================================
# 4. LOAD EMBEDDINGS
# ============================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


embeddings = get_embeddings()


# ============================================================
# 5. CREATE DEFAULT EXCEL FAQ
# ============================================================

def create_default_excel():

    faq_data = {

        "Category": [
            "Program Overview",
            "Program Structure",
            "Program Structure",
            "Pricing & Fees",
            "Pricing & Fees",
            "Curriculum & Skills",
            "Curriculum & Skills",
            "Evaluation & Projects",
            "Career & Placement",
            "Leadership & Contact"
        ],

        "Question": [
            "What is the total duration and structure of the PragyanAI program?",
            "What happens in Phase 1 (First 6 Months)?",
            "What happens in Phase 2 (12 Months)?",
            "What is the fee structure for the Founding Batch?",
            "What is the salary potential after completing the program?",
            "What modules are covered in Months 1-3 (Foundational Core)?",
            "What modules are covered in Months 4-6 (Advanced Frontier)?",
            "How are students evaluated during the 6-month offline training?",
            "What career tracks or roles are unlocked?",
            "Who leads PragyanAI and how can I contact them?"
        ],

        "Answer": [

            "The PragyanAI AI GenAI program is an 18-month journey comprising 6 Months of Fully Offline Training followed by a 12-Month Internship & Placement Drive.",

            "Phase 1 (6 Months) consists of intensive offline training with half-day classroom sessions, half-day hands-on labs, real-time projects, monthly hackathons, and technical seminars.",

            "Phase 2 (12 Months) includes an extended internship, live client assignments, technical mock interviews, resume building, and startup/product development exposure.",

            "Founding Batch (First 100 students): Initial Training Fee is ₹50,000 + Success Fee of ₹50,000 after placement (Total ₹1,00,000, discounted from standard ₹1,50,000).",

            "Target packages: AI Engineer (₹6–₹15 LPA), GenAI Engineer (₹8–₹18 LPA), and Agentic AI Engineer (₹10–₹25 LPA).",

            "Month 1: Python Full Stack & Analytics. Month 2: Data Science & BI Analytics. Month 3: Machine Learning Frameworks (AutoML, Streamlit deployment).",

            "Month 4: Deep Learning & Computer Vision (CNNs, PyTorch, YOLO). Month 5: NLP & Generative AI (LLMs, RAG, LangChain, Fine-tuning). Month 6: Agentic AI (CrewAI, AutoGen, Multi-Agent Systems, MCP).",

            "Students participate in 1 Technical Seminar per skill (evaluated out of 100 marks) and 1 Skill-wise 48-Hour Hackathon with cash prizes (₹5,000 winner, ₹3,000 runner-up).",

            "7 Multi-Track Pathways: Data Analyst, Data Scientist & ML, AI Engineer, GenAI Engineer, Agentic AI Engineer, Product/MVP Engineer, and Software Engineer.",

            "Led by Sateesh Ambesange (Co-Founder, NITK alumnus, 25+ years IT exp). Phone: +91-9741007422 | Email: sateesh.ambesange@pragyanai.com / pragyan.ai.school@gmail.com"
        ]
    }

    df = pd.DataFrame(faq_data)

    file_path = "pragyan_faq_prices.xlsx"

    df.to_excel(
        file_path,
        index=False
    )

    return file_path


# ============================================================
# 6. LOAD DOCUMENTS
# ============================================================

def load_documents_into_vectorstore(uploaded_files=None):

    docs = []

    # --------------------------------------------------------
    # A. Process uploaded files
    # --------------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_name = uploaded_file.name

            # Save uploaded file temporarily
            with open(file_name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ---------------- PDF ----------------

            if file_name.lower().endswith(".pdf"):

                loader = PyPDFLoader(file_name)

                pdf_docs = loader.load()

                docs.extend(pdf_docs)

            # ---------------- Excel ----------------

            elif (
                file_name.lower().endswith(".xlsx")
                or file_name.lower().endswith(".xls")
            ):

                excel_df = pd.read_excel(file_name)

                for _, row in excel_df.iterrows():

                    content = " | ".join(
                        [
                            f"{col}: {val}"
                            for col, val in row.items()
                        ]
                    )

                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_name
                            }
                        )
                    )

    # --------------------------------------------------------
    # B. Load default Excel FAQ
    # --------------------------------------------------------

    default_file = "pragyan_faq_prices.xlsx"

    if not os.path.exists(default_file):

        create_default_excel()

    excel_df = pd.read_excel(default_file)

    for _, row in excel_df.iterrows():

        content = " | ".join(
            [
                f"{col}: {val}"
                for col, val in row.items()
            ]
        )

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": default_file
                }
            )
        )

    # --------------------------------------------------------
    # C. Fallback knowledge
    # --------------------------------------------------------

    if not docs:

        docs = [

            Document(
                page_content=
                "PragyanAI Program: 6 Months Offline Training + "
                "12 Months Placement Drive. "
                "Led by Sateesh Ambesange."
            ),

            Document(
                page_content=
                "Founding Batch Fee: ₹50,000 initial training + "
                "₹50,000 success fee post placement."
            )
        ]

    # --------------------------------------------------------
    # D. Create FAISS vector store
    # --------------------------------------------------------

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore, len(docs)


# ============================================================
# 7. INITIALIZE DEFAULT VECTOR STORE
# ============================================================

if st.session_state.vectorstore is None:

    with st.spinner("Loading PragyanAI Knowledge Base..."):

        vectorstore, doc_count = load_documents_into_vectorstore()

        st.session_state.vectorstore = vectorstore

        st.session_state.doc_count = doc_count


# ============================================================
# 8. GROQ API KEY
# ============================================================

# Recommended:
# Set GROQ_API_KEY in Streamlit secrets or environment variables.

groq_api_key = None

try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    groq_api_key = os.getenv("GROQ_API_KEY")


# ============================================================
# 9. GROQ LLM
# ============================================================

@st.cache_resource
def get_llm(api_key):

    if not api_key:

        return None

    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )


llm = get_llm(groq_api_key)


# ============================================================
# 10. CREATE RAG CHAIN
# ============================================================

def create_rag_chain(
    persona_name,
    retrieved_context,
    chat_history
):

    system_instruction = SALES_PROMPTS.get(
        persona_name,
        SALES_PROMPTS["PragyanAI Student Counselor"]
    ).format(
        context=retrieved_context
    )

    prompt = ChatPromptTemplate.from_messages([

        (
            "system",
            system_instruction
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            "{input}"
        )
    ])

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({

        "input": st.session_state.current_question,

        "history": chat_history
    })

    return response


# ============================================================
# 11. RAG RESPONSE FUNCTION
# ============================================================

def get_response(
    message,
    persona_name
):

    if not llm:

        return (
            "⚠️ Groq API key is missing.\n\n"
            "Please configure `GROQ_API_KEY` in your "
            "Streamlit secrets or environment variables."
        )

    if st.session_state.vectorstore is None:

        return "⚠️ Knowledge base is not loaded."

    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    retriever = st.session_state.vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    relevant_docs = retriever.invoke(message)

    context_str = "\n\n".join(
        [
            f"- {doc.page_content}"
            for doc in relevant_docs
        ]
    )

    # --------------------------------------------------------
    # Convert Streamlit messages into LangChain history
    # --------------------------------------------------------

    from langchain_core.messages import HumanMessage, AIMessage

    chat_history = []

    for msg in st.session_state.messages:

        if msg["role"] == "user":

            chat_history.append(
                HumanMessage(
                    content=msg["content"]
                )
            )

        elif msg["role"] == "assistant":

            chat_history.append(
                AIMessage(
                    content=msg["content"]
                )
            )

    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    st.session_state.current_question = message

    response = create_rag_chain(
        persona_name,
        context_str,
        chat_history
    )

    return response


# ============================================================
# 12. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🤖 PragyanAI")

    st.subheader("Assistant Persona")

    persona_selector = st.selectbox(

        "Select Persona",

        list(SALES_PROMPTS.keys()),

        index=0
    )

    # Detect persona change
    if persona_selector != st.session_state.current_persona:

        st.session_state.current_persona = persona_selector

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # --------------------------------------------------------
    # Upload Files
    # --------------------------------------------------------

    st.subheader("📚 Knowledge Base")

    uploaded_files = st.file_uploader(

        "Upload PDF / Excel files",

        type=[
            "pdf",
            "xlsx",
            "xls"
        ],

        accept_multiple_files=True
    )

    if st.button(
        "🔄 Update Knowledge Base",
        use_container_width=True
    ):

        with st.spinner(
            "Processing documents..."
        ):

            vectorstore, count = (
                load_documents_into_vectorstore(
                    uploaded_files
                )
            )

            st.session_state.vectorstore = vectorstore

            st.session_state.doc_count = count

        st.success(
            f"Knowledge Base updated with {count} documents."
        )

    st.info(
        f"📄 Documents indexed: "
        f"{st.session_state.get('doc_count', 0)}"
    )

    st.divider()

    # --------------------------------------------------------
    # Clear Chat
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat Memory",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# 13. MAIN UI
# ============================================================

st.title(
    "🤖 PragyanAI Conversational Sales & FAQ Assistant"
)

st.caption(
    "AI-powered assistant grounded in the PragyanAI "
    "program knowledge base."
)


# ============================================================
# 14. PERSONA INFORMATION
# ============================================================

persona_descriptions = {

    "PragyanAI Student Counselor":
        "🎓 Aarav — Academic & Career Advisor",

    "PragyanAI Institutional / CoE Advisor":
        "🏫 Dr. Kavita — Institutional Relations Lead",

    "PragyanAI Enterprise AI & Placement Lead":
        "💼 Rohan — Enterprise Placement & Venture Lead"
}


st.info(
    persona_descriptions[
        st.session_state.current_persona
    ]
)


# ============================================================
# 15. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 16. CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Ask anything about PragyanAI..."
)


if user_message:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": user_message
    })

    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            response = get_response(
                user_message,
                st.session_state.current_persona
            )

        st.markdown(
            response
        )

    # --------------------------------------------------------
    # Save response
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": response
    })
