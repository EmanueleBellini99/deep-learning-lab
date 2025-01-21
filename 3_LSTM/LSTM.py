'''
Assignment 3
EMANUELE BELLINI
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import collections
import string

# Constants
EOS = "<EOS>"
PAD = "<PAD>"
BATCH_SIZE = 32
# Target threshold for training loss
LOSS_VALID_THRESHOLD = 1.5 

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_index_map(sequence, map_dict: dict, default=None):
    """Question 5 helper function"""
    return [map_dict.get(key, default) for key in sequence]

class NewsDataset(Dataset):
    """Question 5"""
    def __init__(self, tokenized_sequences, word_to_int):
        self.data_as_int = []
        # Convert sequences to integers, using random choice for unknown words
        choice = np.random.choice(list(word_to_int.values()))
        for sequence in tokenized_sequences:
            self.data_as_int.append(create_index_map(sequence, word_to_int, choice))

    def __len__(self):
        return len(self.data_as_int)

    def __getitem__(self, idx):
        item = self.data_as_int[idx]
        x = item[:-1]  # All but last
        y = item[1:]   # All but first
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    """Question 6"""
    sequences, targets = zip(*batch)
    pad_value = word_to_int[PAD]
    
    # Pad sequences to max length in batch
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, 
                                               padding_value=pad_value)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, 
                                             padding_value=pad_value)
    return padded_sequences, padded_targets

class NewsLSTM(nn.Module):
    """Model"""
    def __init__(self, vocab_size, hidden_size=1024, emb_dim=150, n_layers=1, dropout_p=0.2):
        super(NewsLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim,
            padding_idx=word_to_int[PAD]
        )

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p if n_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        output = self.dropout(output)
        output = self.fc(output)
        return output, state

    def init_state(self, batch_size=1, device=None):
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if device:
            h0 = h0.to(device)
            c0 = c0.to(device)
        return (h0, c0)

def random_sample_next(model, x, prev_state, device):
    """Sample next word randomly according to model probabilities."""
    x = x.to(device)
    prev_state = tuple(s.to(device) for s in prev_state)
    
    output, state = model(x, prev_state)
    last_output = output[0, -1, :]
    
    # Apply softmax and convert to probabilities
    probs = F.softmax(last_output.detach(), dim=-1).cpu().numpy()
    sampled_idx = np.random.choice(len(probs), p=probs)
    
    return sampled_idx, state

def sample_argmax(model, x, prev_state, device):
    """Sample next word using greedy strategy (highest probability)."""
    x = x.to(device)
    prev_state = tuple(s.to(device) for s in prev_state)
    
    output, state = model(x, prev_state)
    last_output = output[0, -1, :]
    return torch.argmax(last_output).item(), state
        
        

def generate_text(model, seed_text, word_to_int, int_to_word, device, 
                 max_len=20, sampling_func=sample_argmax):
    """Generate text from a seed prompt."""
    model.eval()
    words = seed_text.lower().split()
    state = model.init_state(1, device=device)
    
    # Initialize with seed text
    for word in words[:-1]:
        x = torch.tensor([[word_to_int.get(word, 0)]]).to(device)
        _, state = model(x, state)
    
    result = words.copy()
    curr_word = words[-1]
    
    # Generate new words
    with torch.no_grad():
        for _ in range(max_len - len(words)):
            x = torch.tensor([[word_to_int.get(curr_word, 0)]]).to(device)
            idx, state = sampling_func(model, x, state, device)
            curr_word = int_to_word[idx]
            result.append(curr_word)
            
            if curr_word == EOS:
                break
    
    return ' '.join(result)


def generate_multiple_sentences(model, prompt, word_to_int, int_to_word, device, num_sentences=3):
    """Function used for the evaluation of the model."""
    print(f"\nPrompt: {prompt}")
    
    print("\nRandom sampling strategy:")
    for i in range(num_sentences):
        print(f"{i+1}. {generate_text(model, prompt, word_to_int, int_to_word, device, sampling_func=random_sample_next)}")
    
    print("\nGreedy sampling strategy:")
    for i in range(num_sentences):
        print(f"{i+1}. {generate_text(model, prompt, word_to_int, int_to_word, device)}")

def train_standard(model, dataloader, criterion, optimizer, device, 
                  n_epochs=12, clip=1.0, print_every=1, target_loss=1.5):
    """Standard training loop"""
    model.to(device)
    model.train()
    
    losses = []
    perplexities = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        model.train()
        
        for batch_num, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            state = model.init_state(inputs.size(0), device)
            output, _ = model(inputs, state)
            
            loss = criterion(output.transpose(1, 2), targets)
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        perplexity = np.exp(avg_loss)
        losses.append(avg_loss)
        perplexities.append(perplexity)
        
        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}')
            print("\nSample generation:")
            print(generate_text(model, "the president announces", word_to_int, int_to_word, device))
            print()

        # Early stopping when target loss is reached
        if avg_loss < target_loss:
            print(f"\nReached target loss of {target_loss} at epoch {epoch}")
            break
    
    return losses, perplexities

def train_tbptt(model, dataloader, criterion, optimizer, device, 
                chunk_size=9, n_epochs=7, clip=1.0, target_loss=1.5):
    """TBPTT training loop"""
    model.to(device)
    model.train()
    
    losses = []
    perplexities = []
    
    for epoch in range(n_epochs):
        total_loss = 0
        total_chunks = 0
        
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            seq_length = inputs.size(1)
            n_chunks = max(1, seq_length // chunk_size)
            total_chunks += n_chunks
            
            state = model.init_state(batch_size, device)
            
            for i in range(n_chunks):
                model.train()
                
                if i < n_chunks - 1:
                    chunk_input = inputs[:, i*chunk_size:(i+1)*chunk_size].to(device)
                    chunk_target = targets[:, i*chunk_size:(i+1)*chunk_size].to(device)
                else:
                    chunk_input = inputs[:, i*chunk_size:].to(device)
                    chunk_target = targets[:, i*chunk_size:].to(device)
                
                if i > 0:
                    state = tuple(s.detach() for s in state)
                
                output, state = model(chunk_input, state)
                loss = criterion(output.transpose(1, 2), chunk_target)
                
                optimizer.zero_grad()
                loss.backward()
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
                
                total_loss += loss.item()
        
        avg_loss = total_loss / total_chunks
        perplexity = np.exp(avg_loss)
        losses.append(avg_loss)
        perplexities.append(perplexity)
        
        print(f'Epoch {epoch}/{n_epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}')
        
        model.eval()
        print("\nSample generation:")
        print(generate_text(model, "the president announces", word_to_int, int_to_word, device))
        print()
        model.train()

        # Early stopping when target loss is reached
        if avg_loss < target_loss:
            print(f"\nReached target loss of {target_loss} at epoch {epoch}")
            break
    
    return losses, perplexities




if __name__ == "__main__":
    '''
    Data
    '''
    # Question 1
    ds = load_dataset("heegyu/news-category-dataset")
    print("\nDataset structure:")
    print(ds['train'].features)
    
    # Question 2
    politics_data = ds['train'].filter(lambda x: x['category'] == 'POLITICS')
    print(f"\nNumber of politics articles: {len(politics_data)}")
    
    # Question 3
    tokenized_headlines = [headline.lower().split() + [EOS] for headline in politics_data['headline']]
    print("\nExample tokenized headline:")
    print(tokenized_headlines[0])
    
    # Question 4
    # Get unique words
    word_set = set()
    for headline in tokenized_headlines:
        word_set.update(word for word in headline if word != EOS)
    # Create vocabulary with special tokens
    vocab = [EOS] + sorted(list(word_set)) + [PAD]
    word_to_int = {word: idx for idx, word in enumerate(vocab)}
    int_to_word = {idx: word for word, idx in word_to_int.items()}
    # Count word frequencies
    word_counts = collections.Counter(word for headline in tokenized_headlines 
                                    for word in headline if word != EOS)
    print("\nMost common words:")
    for word, count in word_counts.most_common(5):
        print(f"'{word}': {count}")
    
    # Question 5-6
    dataset = NewsDataset(tokenized_headlines, word_to_int)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                          collate_fn=collate_fn, shuffle=True)
    
    '''
    Model & Training
    '''
    model = NewsLSTM(len(word_to_int))
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    # Evaluate the model by generating sentences
    prompt = "the president wants"
    print("The prompt is:", prompt) 
    print("\nEvaluating the model before training:")
    generate_multiple_sentences(model.to(device), prompt, word_to_int, int_to_word, device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int[PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining with standard backpropagation:")
    losses_std, perplexities_std = train_standard(
    model, dataloader, criterion, optimizer, device, target_loss=LOSS_VALID_THRESHOLD)
    
    # Create new model for TBPTT
    model_tbptt = NewsLSTM(len(word_to_int), hidden_size=2048)  # Larger model as suggested
    optimizer_tbptt = torch.optim.Adam(model_tbptt.parameters(), lr=0.001)
    
    print("\nTraining with TBPTT:")
    losses_tbptt, perplexities_tbptt = train_tbptt(
    model_tbptt, dataloader, criterion, optimizer_tbptt, device, target_loss=LOSS_VALID_THRESHOLD)
    
    
    '''
    Visualization
    '''
    # Plot training results
    plt.figure(figsize=(15, 5))
    
    # Loss plots
    plt.subplot(1, 2, 1)
    plt.plot(losses_std, label='Standard Training')
    plt.plot(losses_tbptt, label='TBPTT')
    plt.axhline(y=LOSS_VALID_THRESHOLD, color='r', linestyle='--', 
                label='Target Threshold')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Perplexity plots
    plt.subplot(1, 2, 2)
    plt.plot(perplexities_std, label='Standard Training')
    plt.plot(perplexities_tbptt, label='TBPTT')
    plt.title('Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    '''
    Final Evaluation
    '''
    print("\nFinal Evaluation:")
    
    prompt = "the president wants"
    print("First prompt is:", prompt) 
    print("\nStandard Training Model:")
    generate_multiple_sentences(model, prompt, word_to_int, int_to_word, device)
    print("\nTBPTT Model:")
    generate_multiple_sentences(model_tbptt, prompt, word_to_int, int_to_word, device)
    
    
    '''
    Bonus Question
    '''
    def get_word_embedding(model, word, word_to_int):
        """Get the embedding vector for a word with normalization."""
        if word not in word_to_int:
            return None
        word_idx = word_to_int[word]
        # Get and normalize the embedding vector
        vector = model.embedding.weight[word_idx].detach().cpu().numpy()
        return vector / np.linalg.norm(vector)  # L2 normalization

    def find_analogies(model, word_to_int, int_to_word, word1, word2, word3, n=10):
        """Find word analogies using normalized embeddings."""
        # Get embeddings
        v1 = get_word_embedding(model, word1, word_to_int)
        v2 = get_word_embedding(model, word2, word_to_int)
        v3 = get_word_embedding(model, word3, word_to_int)
        
        if v1 is None or v2 is None or v3 is None:
            return []
        
        # Calculate target vector
        target = v1 - v2 + v3
        # L2 normalize
        target = target / np.linalg.norm(target)
        
        # Calculate similarities with all words
        similarities = []
        for idx, word in int_to_word.items():
            if word in {word1, word2, word3, EOS, PAD}:  # Skip input words and special tokens
                continue
            
            vector = model.embedding.weight[idx].detach().cpu().numpy()
            vector = vector / np.linalg.norm(vector)
            
            similarity = np.dot(target, vector)
            similarities.append((word, similarity))
        
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

    # Test multiple analogies
    def test_analogies(model, word_to_int, int_to_word):
        """Test multiple word analogies to evaluate embedding quality."""
        analogy_pairs = [
            ('king', 'man', 'woman'),  # Original pair
            ('president', 'man', 'woman'),  # Political context
            ('senator', 'man', 'woman'),
            ('trump', 'republican', 'democrat'),
            ('obama', 'democrat', 'republican')
        ]
        
        print("\nTesting multiple word analogies:")
        for w1, w2, w3 in analogy_pairs:
            if all(w in word_to_int for w in (w1, w2, w3)):
                print(f"\n{w1} - {w2} + {w3} = ?")
                results = find_analogies(model, word_to_int, int_to_word, w1, w2, w3)
                for word, sim in results[:5]:
                    print(f"{word}: {sim:.4f}")
            else:
                missing = [w for w in (w1, w2, w3) if w not in word_to_int]
                print(f"\nCannot test {w1}-{w2}+{w3}. Missing words: {missing}")

    # Add to the evaluation section:
    print("\nAnalyzing embeddings with standard model:")
    test_analogies(model, word_to_int, int_to_word)
    print("\nAnalyzing embeddings with TBPTT model:")
    test_analogies(model_tbptt, word_to_int, int_to_word)