import gradio as gr
from generate import ask


def handle_query(question):
    """
    Takes a question from the Gradio UI, runs it through
    the full RAG pipeline, and returns the answer and sources
    as separate strings for display.
    """

    # Don't process empty questions
    if not question.strip():
        return "Please enter a question.", "No sources."

    # Run the full RAG pipeline
    result = ask(question)

    # Format the answer
    answer = result["answer"]

    # Format the sources nicely
    if result["sources"]:
        sources_text = "\n".join(f"• {s}" for s in result["sources"])
    else:
        sources_text = "No sources — question was out of scope."

    return answer, sources_text


# --- Build the Gradio UI ---
with gr.Blocks() as demo:

    # Header
    gr.Markdown("""
    # 🎓 FIU Unofficial Professor Guide
    **Your AI-powered guide to real student experiences at Florida International University.**
    
    Ask anything about professors in Computer Science, Computer Engineering, or Information Systems.
    Answers are grounded exclusively in real student reviews from Rate My Professors.
    """)

    # Example questions
    gr.Markdown("""
    ---
    ### 💡 Example Questions
    - What do students say about Gregory Reis?
    - Which Computer Science professor should I avoid?
    - Is Vladimir Pozdin recommended for beginners?
    - What is the workload like for Andres Rodriguez?
    - How do students describe Rehan Akbar's teaching style?
    - Which Information Systems professor is easiest?
    ---
    """)

    # Input row
    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a Question",
            placeholder="e.g. What do students say about Gregory Reis exams?",
            lines=2,
            scale=5
        )

    # Button row
    with gr.Row():
        submit_btn = gr.Button(
            "🔍 Ask",
            variant="primary",
            scale=2
        )
        clear_btn = gr.Button(
            "🗑️ Clear",
            variant="secondary",
            scale=1
        )

    # Answer output
    with gr.Row():
        answer_output = gr.Textbox(
            label="📝 Answer",
            lines=10,
            interactive=False
        )

    # Sources output
    with gr.Row():
        sources_output = gr.Textbox(
            label="📚 Retrieved From",
            lines=3,
            interactive=False
        )

    # Footer
    gr.Markdown("""
    ---
    *This system answers only from collected student reviews.
    For questions outside this scope, it will say so honestly.
    Reviews sourced from Rate My Professors — Florida International University.*
    """)

    # Submit button click
    submit_btn.click(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    # Press Enter to submit
    question_input.submit(
        fn=handle_query,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

    # Clear button resets everything
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question_input, answer_output, sources_output]
    )


# --- Launch the app ---
if __name__ == "__main__":
    print("Starting FIU Unofficial Professor Guide...")
    print("Open your browser at: http://localhost:7860")
    demo.launch(theme=gr.themes.Soft())