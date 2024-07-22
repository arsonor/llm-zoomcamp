import streamlit as st


from openai import OpenAI
from elasticsearch import Elasticsearch


client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

es = Elasticsearch("http://localhost:9200")
index_name = "course-questions"


context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()


prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def retrieve_documents(
        query,
        index_name="course-questions",
        max_results=5,
        course=None
    ):
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents


def build_context(documents):
    context_result = ""
    
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt


def llm(prompt, model="phi3"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer


def qa_bot(user_question, course):
    context_docs = retrieve_documents(user_question, course=course)
    prompt = build_prompt(user_question, context_docs)
    answer = llm(prompt)
    return answer



def main():
    st.title("DTC Q&A System")

    courses = [
    "data-engineering-zoomcamp",
    "machine-learning-zoomcamp",
    "mlops-zoomcamp"
    ]
    zoomcamp_option = st.selectbox("Select a zoomcamp", courses)

    with st.form(key='rag_form'):
        user_question = st.text_input("Enter your question")
        response_placeholder = st.empty()
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        response_placeholder.markdown("Loading...")
        response = qa_bot(user_question, zoomcamp_option)
        response_placeholder.markdown(response)

    
if __name__ == "__main__":
    main()