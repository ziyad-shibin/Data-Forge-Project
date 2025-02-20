import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io

GOOGLE_API_KEY = "AIzaSyAT6PmJVITWLxrZxYa7qcfP-oB828211Eg"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

def analyze_dataframe(df):
    structure = []
    for column in df.columns:
        dtype = str(df[column].dtype)
        sample = df[column].head(3).tolist()
        structure.append(f"Column: {column}, Type: {dtype}, Sample: {sample}")
    return "\n".join(structure)

def generate_dataframe_query(user_input, df_structure):
    prompt = f"""
    You are given a DataFrame with the following structure:
    {df_structure}

    The user input is: "{user_input}"
    Based on this information, generate code that performs the required operation(s) to address the user's request.
    The code should use the columns and data types provided in the structure and can include various types of operations such as filtering, aggregation, transformation, or visualization.
    If the user requests a visualization, use matplotlib or seaborn to create the plot.You should never use the import statement and labelling of grapgh in the code.Make use of simple code structure for visualization.
    For visualizations, return both the DataFrame and the plot object.
    Provide the code in a format that can be directly executed on the DataFrame. Do not include any additional text or explanations or comments or import statements for code.
    Ensure that the code accurately performs the requested operations using the available columns.
    """
    response = model.generate_content(prompt)
    return response.text.strip().replace('```python', '').replace('```', '').strip()

def execute_query(query, df):
    local_vars = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns}
    lines = query.strip().split('\n')
    for line in lines[:-1]:
        exec(line, globals(), local_vars)
    result = eval(lines[-1], globals(), local_vars)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], plt.Figure):
        return result
    elif isinstance(result, plt.Figure):
        return df, result
    else:
        return result, plt.gcf()

def generate_sql_query(text_input):
    template = """
    Create a SQL query snippet using the below text:
    '''
        {text_input}
    '''
    I want a SQL Query.
    """
    formatted_template = template.format(text_input=text_input)
    response = model.generate_content(formatted_template)
    return response.text.strip().lstrip("```sql").rstrip("```")

def generate_expected_output(sql_query):
    expected_output = """
    What would be the expected response of this SQL query snippet:
        '''
        {sql_query}
        '''
    Provide sample tabular Response with no explanation
    """
    expected_output_formatted = expected_output.format(sql_query=sql_query)
    eoutput = model.generate_content(expected_output_formatted)
    return eoutput.text

def generate_explanation(sql_query):
    explanation = """
    Explain the SQL Query
        '''
        {sql_query}
        '''
    Please provide with simplest of explanation:
    """
    explanation_formatted = explanation.format(sql_query=sql_query)
    exp = model.generate_content(explanation_formatted)
    return exp.text

def generate_documentation(sql_query):
    documentation_template = """
    Create documentation for the SQL query:
    '''
        {sql_query}
    '''
    Provide detailed documentation including descriptions of tables, columns, and the query logic.
    """
    formatted_doc_template = documentation_template.format(sql_query=sql_query)
    doc_response = model.generate_content(formatted_doc_template)
    return doc_response.text

# New Python-related functions
def generate_python_code(description):
    prompt = f"""
    Generate Python code based on the following description:
    {description}

    Provide only the code, without any additional explanations or comments.
    """
    response = model.generate_content(prompt)
    return response.text.strip().replace('```python', '').replace('```', '').strip()

def explain_python_code(code):
    prompt = f"""
    Explain the following Python code in simple terms:
    ```python
    {code}
    ```
    Provide a clear and concise explanation of what the code does.
    """
    response = model.generate_content(prompt)
    return response.text.strip().replace('```python', '').replace('```', '').strip()


def generate_python_documentation(topic):
    prompt = f"""
    Generate comprehensive Python documentation for the following topic:
    {topic}

    Include explanations, examples, and best practices.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# Page Functions
def sql_query_generator_page():
    st.title("Advanced SQL Query Generator")


    with st.form("sql_query_form"):
        text_input = st.text_input("Enter your query here in English")
        submit = st.form_submit_button("Generate SQL Query")

    if submit and text_input:
        with st.spinner("Generating SQL Query..."):
            sql_query = generate_sql_query(text_input)
            expected_output = generate_expected_output(sql_query)
            explanation = generate_explanation(sql_query)

        st.success("SQL Query Generated Successfully")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Generated SQL Query")
            st.code(sql_query, language='sql')

        with col2:
            st.subheader("Explanation")
            st.markdown(explanation)

        st.subheader("Expected Output")
        st.markdown(expected_output)

def dataframe_query_generator_page():
    st.title("Analyze Your Dataset Using Gemini Pro")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head())

        with col2:
            st.subheader("Dataframe Structure")
            st.text(analyze_dataframe(df))

        with st.form("dataframe_query_form"):
            user_input = st.text_input("Enter your query (you can also ask for visualizations):")
            submit = st.form_submit_button("Execute Query")

        if submit and user_input:
            query = generate_dataframe_query(user_input, analyze_dataframe(df))

            st.subheader("Generated Query")
            st.code(query, language='python')

            try:
                with st.spinner("Executing query..."):
                    result, plot = execute_query(query, df)

                st.subheader("Results")
                if isinstance(result, pd.DataFrame):
                    st.dataframe(result)
                elif isinstance(result, (pd.Series, list, tuple)):
                    st.write(result)
                else:
                    st.write(str(result))

                if plot:
                    st.pyplot(plot)
                    plt.close(plot)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Generated Query:", query)

def sql_documentation_page():
    st.title("SQL Documentation Generator")

    with st.form("sql_documentation_form"):
        topic = st.text_input("Enter a SQL topic for documentation:")
        submit = st.form_submit_button("Generate Documentation")
    if submit and topic:
        documentation = generate_documentation(topic)
        st.markdown(documentation)

def python_code_generator_page():
    st.title("Python Code Generator and Analyzer")

    with st.form("python_code_geneartion_form"):

        description = st.text_input("Describe the Python code you want to generate:")
        submit = st.form_submit_button("Generate Code")
    if submit and description:
        code = generate_python_code(description)
        st.subheader("Generated Code")
        st.code(code, language='python')

        explanation = explain_python_code(code)
        st.subheader("Code Explanation")
        st.write(explanation)

        st.subheader("Code Output")
        try:
            with st.spinner("Executing code..."):
                old_stdout = sys.stdout
                redirected_output = sys.stdout = io.StringIO()

                exec(code)

                sys.stdout = old_stdout
                output = redirected_output.getvalue()

            if output:
                st.text_area("Output:", value=output, height=100, disabled=True)
            else:
                st.info("The code didn't produce any output.")
        except Exception as e:
            st.error(f"The code didn't produce any output.")

def python_documentation_page():
    st.title("Python Documentation Generator")

    with st.form("python_documentation_form"):
        topic = st.text_input("Enter a Python topic for documentation:")
        submit = st.form_submit_button("Generate Documentation")
    if submit and topic:
        documentation = generate_python_documentation(topic)
        st.markdown(documentation)

# Main App
def main():
    st.set_page_config(page_title="SQL & Python Analysis Assistant", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:",["SQL Query Generator","SQL Documentation","Python Code Generator","Python Documentation" ,"Analyze Your Dataset"])

    if page == "SQL Query Generator":
        sql_query_generator_page()
    elif page == "Analyze Your Dataset":
        dataframe_query_generator_page()
    elif page == "SQL Documentation":
        sql_documentation_page()
    elif page == "Python Code Generator":
        python_code_generator_page()
    elif page == "Python Documentation":
        python_documentation_page()

if __name__ == "__main__":
    main()




