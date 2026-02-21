import torch
import requests
from PIL import Image
from io import BytesIO
from torchvision import transforms
from torch_geometric.data import Data, Batch

def fetch_image(url):
    default_tensor = torch.zeros((3, 224, 224))
    if not isinstance(url, str) or not url.strip():
        return default_tensor
    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(img)
    except Exception:
        return default_tensor

def build_ego_graph(user_id, neighbor_dict):
    followers = neighbor_dict.get("follower", []) if neighbor_dict else []
    following = neighbor_dict.get("following", []) if neighbor_dict else []
    
    node_to_idx = {user_id: 0}
    curr_idx = 1
    edges = []
    
    for f in followers:
        if f not in node_to_idx:
            node_to_idx[f] = curr_idx
            curr_idx += 1
        edges.append([node_to_idx[f], 0])
        
    for f in following:
        if f not in node_to_idx:
            node_to_idx[f] = curr_idx
            curr_idx += 1
        edges.append([0, node_to_idx[f]])
        
    if not edges:
        edges = [[0, 0]]
        
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.ones((curr_idx, 16), dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

def collate_twibot_batch(batch_data, tokenizer):
    texts, images, graphs, labels = [], [], [], []
    for item in batch_data:
        profile = item.get("profile") or {}
        neighbor = item.get("neighbor") or {}
        tweet_list = item.get("tweet") or []
        
        bio = profile.get("description") or ""
        tweets = " ".join(tweet_list[:5])
        texts.append(f"{bio} {tweets}")
        
        img_url = profile.get("profile_image_url") or ""
        images.append(fetch_image(img_url))
        
        graphs.append(build_ego_graph(item.get("ID", ""), neighbor))
        
        raw_label = item.get("label", 0)
        try:
            if isinstance(raw_label, str):
                if raw_label.lower() == 'bot':
                    clean_label = 1
                elif raw_label.lower() == 'human':
                    clean_label = 0
                else:
                    clean_label = int(raw_label)
            else:
                clean_label = int(raw_label)
        except ValueError:
            clean_label = 0 
            
        labels.append(clean_label)
        
    text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    image_tensor = torch.stack(images)
    graph_batch = Batch.from_data_list(graphs)
    label_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
    
    return text_inputs, image_tensor, graph_batch, label_tensor
