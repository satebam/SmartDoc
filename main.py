
# Import necessary libraries
import os
import PyPDF2
from strands import Agent, tool
from strands.models import BedrockModel
import streamlit as st


# Initializing the Streamlit app
st.title("Amazon Content Summarizer")

@tool
def file_read_txt(file_path: str) -> str:
    """Read a file and return its content.

    Args:
        file_path (str): Path to the file to read

    Returns:
        str: Content of the file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        with open(file_path, "r", encoding='UTF-8')  as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def file_read_pdf(file_path, encoding='utf-8'):
    """Read a file and return its content.

    Args:
        file_path (str): Path to the file to read

    Returns:
        str: Content of the file

    Raises:
        FileNotFoundError: If the file doesn't exist
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text()
            return text
    except Exception as e:
        print(f"An error occurred: {e}")

@tool
def file_write(file_path: str, content: str) -> str:
    """Write content to a file.

    Args:
        file_path (str): The path to the file
        content (str): The content to write to the file

    Returns:
        str: A message indicating success or failure
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

        with open(file_path, "w") as file:
            file.write(content)
        return f"File '{file_path}' written successfully."
    except Exception as e:
        return f"Error writing to file: {str(e)}"

def initialize_agent():
# system prompt for the Document Summary agent
    system_prompt = """
    You are a document summarizing agent. Please read the document and summarize it.
    After summarizing create and write the content to a file if requested.
    
    When using tools:
    - Always verify file paths before operations
    - Be careful with system commands
    - Provide clear explanations of what you're doing
    - If a task cannot be completed, explain why and suggest alternatives
    - Ignore any non-ASCII characters
    
    When giving the output:
    - Please include a brief introduction of the document.
    - Include the recommendations.
    - Give bullet point responses.
    - Have conclusion at the end of the document.
    
    Please restrict your answers to summarization of documents.
    """

    model = BedrockModel(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        additional_request_fields={
            "thinking": {
                "type": "disabled",
                # "budget_tokens": 2048,
            }
        },
    )


    # Create the agent with granting access to tools defined above
    local_agent = Agent(
        system_prompt=system_prompt,
        model=model,
        tools=[file_read_txt,file_read_pdf,file_write],
    )

    return local_agent

def main():
    try:
        # Initialize the agent
        local_agent = initialize_agent()
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        return


    # Create two columns for file upload and prompt input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Enter Your Prompt")
        custom_prompt = st.text_area(
            "Enter prompt",
            height=150,
            placeholder="Example: Read the files and provide a summary..."
        )

    with col2:
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['txt', 'pdf'],
            accept_multiple_files=True
        )

        # Display uploaded files
        if uploaded_files:
            st.write("Uploaded files:")
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # Size in KB
                size_text = f"{file_size:.1f} KB" if file_size < 1024 else f"{file_size / 1024:.1f} MB"
                st.write(f"📄 {file.name} ({size_text})")

    # Output options
    st.subheader("Output Options")
    output_type = st.radio("Select output type", ["Display", "Save to file"])

    if st.button("Process"):
        if uploaded_files:
            try:
                # Save all uploaded files temporarily
                file_paths = []
                for uploaded_file in uploaded_files:
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                        file_paths.append(uploaded_file.name)

                # Construct the prompt with all file paths
                files_string = ", ".join(file_paths)
                prompt = custom_prompt + f". File paths are: {files_string}"

                # Add file output instruction if needed
                if output_type == "Save to file":
                    prompt += " and write it to a file named output.txt"

                # Show the final prompt
                # st.info(f"Processing with prompt: {prompt}")

                # Process with agent
                with st.spinner('Processing...'):
                    result = local_agent(prompt)
                    st.success("Processing complete!")

                    # Display result in a box
                    st.markdown("""
                        <style>
                            .result-box {
                                border: 2px solid #007bff;
                                border-radius: 10px;
                                padding: 20px;
                                background-color: #f8f9fa;
                                margin: 10px 0;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                        </style>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                        <div class="result-box">
                            {result.message["content"][0]['text']}
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

        else:
            st.warning("Please upload at least one file")


if __name__ == "__main__":
    main()
