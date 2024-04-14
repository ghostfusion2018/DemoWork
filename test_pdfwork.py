from turtle import st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from PDFWorks import main, handle_userinput

class TestMain:

    #  User inputs a valid question and PDFs, and clicks on 'Process'.
    #  The function successfully processes the PDFs, creates a conversation chain, and responds to the user's question with relevant information from the PDFs.
    def test_valid_question_and_PDFs(self, mocker):
        # Mocking file uploader
        mocker.patch('streamlit.file_uploader', return_value=['pdf1.pdf', 'pdf2.pdf'])
    
        # Mocking text input
        mocker.patch('streamlit.text_input', return_value='What is the capital of France?')
    
        # Mocking conversation chain
        mocker.patch('langchain.chains.ConversationalRetrievalChain.from_llm')
    
        # Mocking handle_userinput function
        mocker.patch('handle_userinput')
    
        # Invoke main function
        main()
    
        # Assertions
        assert st.file_uploader.called_once_with("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        assert st.text_input.called_once_with("Ask a question about your documents:")
        assert handle_userinput.called_once_with('What is the capital of France?')
        assert ConversationalRetrievalChain.from_llm.called_once_with(llm=mocker.ANY, retriever=mocker.ANY, memory=mocker.ANY)

    #  User inputs an invalid question (e.g. empty string, non-text input). The function does not respond and waits for a valid input.
    def test_invalid_question(self, mocker):
        # Mocking file uploader
        mocker.patch('streamlit.file_uploader', return_value=['pdf1.pdf', 'pdf2.pdf'])
    
        # Mocking text input
        mocker.patch('streamlit.text_input', return_value='')
    
        # Mocking conversation chain
        mocker.patch('langchain.chains.ConversationalRetrievalChain.from_llm')
    
        # Mocking handle_userinput function
        mocker.patch('handle_userinput')
    
        # Invoke main function
        main()
    
        # Assertions
        assert st.file_uploader.called_once_with("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        assert st.text_input.called_once_with("Ask a question about your documents:")
        assert not handle_userinput.called
        assert not ConversationalRetrievalChain.from_llm.called
