import gradio as gr
import torch
from transformers import AutoTokenizer

# Import your modularized code
from model import MultiModalDetector
from utils import collate_twibot_batch

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing Application on {device}...")

# 2. Build Model & Load Weights
model = MultiModalDetector(text_dim=768, vision_dim=512, graph_dim=128)
model_path = "phase2_production_model.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✅ Success: Weights loaded from {model_path}")
except Exception as e:
    print(f"❌ Error: Could not load weights. Ensure '{model_path}' is in the same directory.")
    print(e)

model = model.to(device)
model.eval()

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 4. Define the Inference Function
def gradio_predict(profile_id, bio, tweets, following_count, follower_count, image_url):
    following_list = [f"f_{i}" for i in range(int(following_count))]
    follower_list = [f"fl_{i}" for i in range(int(follower_count))]
    
    custom_data = {
        "ID": profile_id,
        "profile": {
            "description": bio,
            "profile_image_url": image_url
        },
        "tweet": [t.strip() for t in tweets.split("|") if t.strip()],
        "neighbor": {
            "following": following_list,
            "follower": follower_list
        },
        "label": 0 
    }
    
    try:
        text_inputs, images, graphs, _ = collate_twibot_batch([custom_data], tokenizer)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        images = images.to(device)
        graphs = graphs.to(device)
        
        with torch.no_grad():
            logits = model(text_inputs, images, graphs)
            logits = torch.clamp(logits, min=-10.0, max=10.0) 
            probability = torch.sigmoid(logits).item()
            
        is_bot = probability >= 0.5
        prediction = "🤖 BOT (Fake Profile)" if is_bot else "👤 HUMAN (Real Profile)"
        confidence = (probability if is_bot else (1.0 - probability)) * 100
        
        return f"{prediction}\nConfidence Score: {confidence:.2f}%"
    except Exception as e:
        return f"Error analyzing profile: {e}"

# 5. Build and Launch the Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️ Multi-Modal Fake Profile Detector")
    gr.Markdown("Enter the details of a social media account below to analyze if it is a Bot or a Human.")
    
    with gr.Row():
        with gr.Column():
            prof_id = gr.Textbox(label="Username", placeholder="@username")
            bio = gr.Textbox(label="Bio / Description", lines=2, placeholder="e.g., Crypto enthusiast. DM for 100x returns!")
            tweets = gr.Textbox(label="Recent Tweets (separate with | )", lines=3, placeholder="Tweet 1 | Tweet 2 | Tweet 3")
            img_url = gr.Textbox(label="Profile Image URL", placeholder="https://...")
            
            with gr.Row():
                following = gr.Number(label="Following Count", value=100, precision=0)
                followers = gr.Number(label="Follower Count", value=10, precision=0)
                
            submit_btn = gr.Button("🔍 Analyze Profile", variant="primary")
            
        with gr.Column():
            output_res = gr.Textbox(label="Detection Result", lines=3)
            
    submit_btn.click(gradio_predict, 
                     inputs=[prof_id, bio, tweets, following, followers, img_url], 
                     outputs=output_res)

# Standard Python execution block
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
