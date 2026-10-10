import pandas as pd
from langchain_core.documents import Document

chapter_titles = {
    "2": "Roles, Responsibilities, and Principles Related to Tailoring of the Requirements",
    "3": "Software Management Requirements",
    "4": "Software Engineering (Life Cycle) Requirements",
    "5": "Supporting Software Life Cycle Requirements",
}

section_titles = {
    # Chapter 2
    "2.1": "Roles and Responsibilities",
    "2.2": "Principles Related to Tailoring of the Requirements",
    
    # Chapter 3
    "3.1": "Software Life Cycle Planning",
    "3.2": "Software Cost Estimation",
    "3.3": "Software Schedules",
    "3.4": "Software Training",
    "3.5": "Software Classification Assessments",
    "3.6": "Software Assurance and Software Independent Verification & Validation",
    "3.7": "Safety-Critical Software",
    "3.8": "Automatic Generation of Software Source Code",
    "3.9": "Software Development Processes and Practices",
    "3.10": "Software Reuse",
    "3.11": "Software Cybersecurity",
    "3.12": "Software Bi-Directional Traceability",
    
    # Chapter 4
    "4.1": "Software Requirements",
    "4.2": "Software Architecture",
    "4.3": "Software Design",
    "4.4": "Software Implementation",
    "4.5": "Software Testing",
    "4.6": "Software Operations, Maintenance, and Retirement",
    
    # Chapter 5
    "5.1": "Software Configuration Management",
    "5.2": "Software Risk Management",
    "5.3": "Software Peer Reviews/Inspections",
    "5.4": "Software Measurements",
    "5.5": "Software Non-conformance or Defect Management",
}


def load_requirements(csv_path):
    df = pd.read_csv(csv_path)

    df["chapter_title"] = (
        df["clause"].str.extract(r"^(\d+)", expand=False).map(chapter_titles)
    )

    df["section_title"] = (
        df["clause"].str.extract(r"^(\d+\.\d+)", expand=False).map(section_titles)
    )

    return df


def create_documents(df):
    documents = []

    for _, row in df.iterrows():
        documents.append(
            Document(
                id=str(row["swe_id"]),
                page_content=(
                    f"{row['section_title']}\n\n"
                    f"{row['requirement']}"
                ),
                metadata={
                    "swe_id": str(row["swe_id"]),
                    "clause": str(row["clause"]),
                    "chapter_title": str(row["chapter_title"]),
                    "section_title": str(row["section_title"]),
                    "source": "NASA NPR 7150.2D",
                },
            )
        )

    return documents
