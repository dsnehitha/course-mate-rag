# CourseMateRAG 📖
A Retrieval-Augmented Generation (RAG) system for answering student queries based on course materials.

## Features ✨
- 📂 Loads all PDFs in `course_materials/`
- 🧠 Uses FAISS for fast retrieval
- 🤖 Generates answers with OpenAI GPT-4
- 🔄 Auto-detects new PDFs and updates the index

## Installation 💻
```sh
git clone https://github.com/dsnehitha/course-mate-rag.git
cd course-mate-rag
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
pip install -r requirements.txt