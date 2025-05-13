from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace
from evaluator import Evaluator
from runner import Runner
from settings import Settings
import time
import os
import pandas as pd
import gradio as gr
import logging
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)
settings = Settings()
evaluator = Evaluator(settings)
runner = Runner(settings)


# Create a TracerProvider for OpenTelemetry
trace_provider = TracerProvider()

# Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

# Set the global default tracer provider
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# Instrument smolagents with the configured provider
SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

LOGIN_MESSAGE = "Please Login to Hugging Face with the button."
EMPTY_RESULTS_TABLE = pd.DataFrame(columns=['task_id', 'question', 'answer'])
        
def _format_elapsed_time(elapsed_time):
    minutes = int(elapsed_time // 60)  # Get the whole number of minutes
    seconds = elapsed_time % 60  # Get the remaining seconds

    if minutes > 0:
        return f"Elapsed time: {minutes} minutes {seconds:.2f} seconds"
    else:
        return f"Elapsed time: {seconds:.2f} seconds"
    
def _run(questions: list, username: str) -> pd.DataFrame:
    start_time = time.time()
    question_answer_pairs = runner.run_agent(questions, username)
    end_time = time.time()
    message = f"Complete. {_format_elapsed_time(end_time - start_time)}"
    return message, question_answer_pairs
    
def run_one(profile: gr.OAuthProfile | None) -> pd.DataFrame:
    if profile: 
        return _run([evaluator.get_one_question()], profile.username)
    else:
        return LOGIN_MESSAGE, EMPTY_RESULTS_TABLE

def run_all(profile: gr.OAuthProfile | None) -> pd.DataFrame:
    if profile: 
        return _run(evaluator.get_questions(), profile.username)
    else:
        return LOGIN_MESSAGE, EMPTY_RESULTS_TABLE

def submit(profile: gr.OAuthProfile | None) -> str:
    if profile: 
        return evaluator.submit_answers(profile.username)
    else:
        return LOGIN_MESSAGE


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Log in to your Hugging Face account using the button below. 
        2.  Click 'Get One Answer' to run the agent on a random question or 'Get All Answers' to run all. 
        3.  Click 'Submit Answers' to submit answers for evaluation. This will NOT submit your HF username.

        ---
        **Disclaimers:**
        Once clicking 'Get All Answers', it can take quite some time (this is the time for the agent to go through all 20 questions).
        The agent(s) will run question tasks in parallel making the logs hard to follow. Langfuse instrumentation has been configured. 
        The 'Submit All Answers' button will use the most recent agent answers cached in the space for your username.
        """
    )

    gr.LoginButton()

    run_one_button = gr.Button("Get One Answer")
    run_all_button = gr.Button("Get All Answers")
    submit_button = gr.Button("Submit Answers")

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(
        label="Questions and Agent Answers", wrap=True)

    run_one_button.click(
        fn=run_one, outputs=[status_output, results_table]
    )
    run_all_button.click(
        fn=run_all, outputs=[status_output, results_table]
    )
    submit_button.click(
        fn=submit, outputs=[status_output]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(
            f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
