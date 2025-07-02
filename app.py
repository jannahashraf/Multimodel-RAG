import gradio as gr
from qa_model import MultimodalRAG
import os
import traceback

# Initialize the RAG system
rag = MultimodalRAG()

def upload_file(files):
    results = {}
    results["status"] = []
    
    try:
        if files:
            for file in files:
                file_path = file.name  # Get the temporary file path
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension == '.pdf':
                    # Process PDF and add to knowledge base
                    success = rag.process_upload(file_path, file_type="pdf")
                    if success:
                        results["status"].append(f"✅ {os.path.basename(file_path)} processed successfully!")
                    else:
                        results["status"].append(f"❌ Failed to process {os.path.basename(file_path)}")
                
                elif file_extension in ['.jpg', '.jpeg', '.png']:
                    # Image search
                    image_results = rag.hybrid_search(query_image=file_path)
                    if image_results.get("images"):
                        results["status"].append(f"🖼️ Found visual matches in {os.path.basename(file_path)}")
                    else:
                        results["status"].append(f"🖼️ No similar images found in {os.path.basename(file_path)}")
        
    except Exception as e:
        error_msg = f"🚨 Error: {str(e)}\n\n{traceback.format_exc()}"
        results["status"].append(error_msg)
    
    finally:
        return "\n".join(results["status"])

def run_search(query):
    answer = rag.generate_answer(query)["answer"]
    return answer if answer else {"Status": "⚠️ Please provide a query or upload files"}

# Custom CSS for better UI
css = """
.important-button {
    background: linear-gradient(45deg, #FF3366, #BA265D) !important;
    border: none !important;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 🧠 CS Knowledge RAG System")
    gr.Markdown("Upload multiple PDFs or images to build your knowledge base. Then enter a query to search for answers.")

    with gr.Row():
        file_input = gr.File(
            label="📁 Upload PDFs/Images", 
            file_types=[".pdf", ".jpg", ".jpeg", ".png"], 
            type="filepath",
            file_count="multiple"  # This enables multiple file selection
        )
        upload_output = gr.Text(label="📥 Upload Status", lines=4)

    upload_button = gr.Button("📤 Upload & Embed", elem_classes="important-button")
    upload_button.click(fn=upload_file, inputs=[file_input], outputs=[upload_output])

    gr.Markdown("---")

    query_input = gr.Textbox(label="🔍 Search Query", placeholder="Ask about CS concepts...", lines=2)
    query_output = gr.Text(label="📊 Answer", lines=6)

    search_button = gr.Button("🔎 Run Search")
    search_button.click(fn=run_search, inputs=[query_input], outputs=[query_output])

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)