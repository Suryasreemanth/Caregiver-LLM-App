# event_handler.py

import streamlit as st
from openai import AssistantEventHandler

class MyCustomEventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.full_response = ""  # Initialize to capture the full assistant response
        # Initialize session state variables if they don't exist
        if "text_boxes" not in st.session_state:
            st.session_state.text_boxes = []
        if "assistant_text" not in st.session_state:
            st.session_state.assistant_text = []

    def on_text_created(self, text):
        """
        Handler for when text is first created.
        """
        st.session_state.text_boxes.append(st.empty())
        st.session_state.assistant_text.append("")  # Prepare for new text input

    def on_text_delta(self, delta, snapshot):
        """
        Handler for when a text delta is received.
        """
        if delta.value:
            st.session_state.assistant_text[-1] += delta.value
            st.session_state.text_boxes[-1].markdown(st.session_state.assistant_text[-1])
            self.full_response += delta.value  # Capture the full response incrementally

    def on_text_done(self, text):
        """
        Handler for when the assistant finishes generating text.
        """
        st.session_state.text_boxes[-1].markdown(st.session_state.assistant_text[-1])

    def on_tool_call_created(self, tool_call):
        """
        Handler for when a tool is called.
        """
        st.session_state.text_boxes.append(st.empty())

    def on_tool_call_delta(self, delta, snapshot):
        """
        Handler for updates during a tool call.
        """
        if delta.code_interpreter and delta.code_interpreter.input:
            st.session_state.text_boxes[-1].code(delta.code_interpreter.input)

    def on_tool_call_done(self, tool_call):
        """
        Handler for when a tool call is completed.
        """
        st.session_state.text_boxes.append(st.empty())

    def on_timeout(self):
        """
        Handler for when the API call times out.
        """
        st.error("The API call timed out.")
        st.stop()
