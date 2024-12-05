'''
Assignment 3
Automated Solution
'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Constants
EOS = "<EOS>"
PAD = "<PAD>"
BATCH_SIZE = 64
LOSS_VALID_THRESHOLD_BPTT = 1.5
LOSS_VALID_THRESHOLD_TBPTT = 1.0

# Set the seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_index_map(sequence, map_dict: dict, default=None):
    return [map_dict.get(key, default) for key in sequence]

class NewsDataset(Dataset):
    def __init__(self, tokenised_sequences, word_to_int):
        self.data_tuples = []
        choice = np.random.choice(list(word_to_int))
        for sequences in tokenised_sequences:
            self.data_tuples.append(create_index_map(sequences, word_to_int, choice))

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, index):
        item = self.data_tuples[index]
        x = item[:-1]
        y = item[1:]
        return torch.tensor(x), torch.tensor(y)

def collate_fn(batch):
    data, targets = zip(*batch)
    pad_value = word_to_int[PAD]
    padded_data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=pad_value)
    padded_targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=pad_value)
    return padded_data, padded_targets

class NewsLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, emb_dim=150, n_layers=1, dropout_p=0.2):
        super(NewsLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=emb_dim, 
            padding_idx=word_to_int[PAD]
        )

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True
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
        # Initialize both hidden state and cell state for LSTM
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        if device:
            h = h.to(device)
            c = c.to(device)
        return (h, c)
    
def random_sample_next(model, x, prev_state, device):
    x = x.to(device)
    prev_state = (prev_state[0].to(device), prev_state[1].to(device))
    output, state = model(x, prev_state)
    last_output = output[0, -1, :]
    p = F.softmax(last_output.detach().cpu(), dim=-1).numpy()
    return np.random.choice(len(p), p=p), state


def sample_argmax(model, x, prev_state, device):
    x = x.to(device)
    prev_state = (prev_state[0].to(device), prev_state[1].to(device))
    output, state = model(x, prev_state)
    last_output = output[0, -1, :] 
    return torch.argmax(last_output).item(), state

def generate_text(model, seed_text, word_to_int, int_to_word, device, max_len=20, sampling_func=sample_argmax):
    model.eval()
    words = seed_text.lower().split()
    state = model.init_state(1, device=device)

    for word in words[:-1]:
        x = torch.tensor([[word_to_int[word]]]).to(device)
        _, state = model(x, state)
    
    result = words.copy()
    curr_word = words[-1]
    
    for _ in range(max_len):
        x = torch.tensor([[word_to_int[curr_word]]]).to(device)
        idx, state = sampling_func(model, x, state, device)
        curr_word = int_to_word[idx]
        result.append(curr_word)
        if curr_word == EOS:
            break
    
    return ' '.join(result)

def train_model(model, train_loader, num_epochs, criterion, optimizer, device, 
                word_to_int, int_to_word, clip=1.0, print_every=1):
    model.train()  # Ensure model is in training mode
    model = model.to(device)
    losses = []
    perplexities = []
    
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()  # Ensure model is in training mode at start of each epoch
        
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            state = model.init_state(inputs.size(0), device)  # Now passing device correctly
            
            output, _ = model(inputs, state)
            loss = criterion(output.transpose(1, 2), targets)
            
            loss.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        perplexity = np.exp(avg_loss)
        losses.append(avg_loss)
        perplexities.append(perplexity)
        
        if epoch % print_every == 0:
            print(f'Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}')
            print("Sample generation:")
            print(generate_text(model, "The president announces", word_to_int, int_to_word, device))
    
    return losses, perplexities

def train_with_TBBTT(max_epochs, model, dataloader, criterion, word_to_int,
                     int_to_word, optimizer, chunk_size, device, 
                     sampler_func=sample_argmax,
                     seed="Make America", clip=None):
    model.to(device)
    model.train()
    losses = []
    perplexities = []
    running_loss = 0
    
    for epoch in range(max_epochs):
        print(f"In epoch {epoch}")
        total_chunks = 0
        for input, output in dataloader:
            n_chunks = max(1, input.shape[1] // chunk_size)
            total_chunks += n_chunks
            
            for j in range(n_chunks):
                if j < n_chunks - 1:
                    input_chunk = input[:, j*chunk_size:(j+1)*chunk_size].to(device)
                    output_chunk = output[:, j*chunk_size:(j+1)*chunk_size].to(device)
                else:
                    input_chunk = input[:, j*chunk_size:].to(device)
                    output_chunk = output[:, j*chunk_size:].to(device)

                if j == 0:
                    h, c = model.init_state(input_chunk.shape[0], device=device)
                    h, c = h.to(device), c.to(device)
                else:
                    h, c = h.detach().to(device), c.detach().to(device)

                out, state = model(input_chunk, (h, c))
                h, c = state

                loss = criterion(out.transpose(1, 2), output_chunk)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                
                if clip:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    
                optimizer.step()

        print(f"Epoch: {epoch}/{max_epochs}, Loss: {running_loss/float(total_chunks)}")
        losses.append(running_loss/float(total_chunks))
        perplexity = np.exp(running_loss/float(total_chunks))
        print(f"Perplexity: {perplexity}")
        perplexities.append(perplexity)
        running_loss = 0
        
        print(f"Sample generation:")
        model.eval()
        seed_ind = create_index_map(seed.lower().split(), word_to_int, None)
        sampled_list = generate_text(model, seed_ind, device, func=sampler_func)
        print(f"{seed} => {' '.join(create_index_map(sampled_list, int_to_word, '<?>'))}") 

    return model, losses, perplexities

if __name__ == "__main__":
    '''
    Data
    '''
    # Question 1.1.1
    print("Question 1.1.1: Loading and filtering politics data")
    ds = load_dataset("heegyu/news-category-dataset")
    politics_data = ds['train'].filter(lambda x: x['category'] == 'POLITICS')
    print("First 3 politics entries:")
    for i in range(3):
        print(politics_data[i])
    
    # Question 1.1.2
    print("\nQuestion 1.1.2: Tokenizing headlines")
    tokenised_headlines = [headline.lower().split() + [EOS] for headline in politics_data['headline']]
    # Save the sequences
    with open("./tokenised_seq.pickle", "wb") as f:
        pickle.dump(tokenised_headlines, f)
    print("First 3 tokenized headlines:")
    for i in range(3):
        print(tokenised_headlines[i])
    
    # Question 1.1.3
    print("\nQuestion 1.1.3: Creating vocabulary and word mappings")
    all_words_set = set()
    for i in tokenised_headlines:
        for j in i:
            if j != EOS:
                all_words_set.add(j)
    all_words = list(all_words_set)
    all_words.insert(0, EOS)
    all_words.append(PAD)
    
    word_to_int = {word: i for i, word in enumerate(all_words)}
    int_to_word = {i: word for word, i in word_to_int.items()}
    
    # Count word frequencies
    count_dict = {word: 0 for word in all_words}
    for i in tokenised_headlines:
        for j in i:
            if j != EOS and j != PAD:
                count_dict[j] += 1
    
    count_dict = {k: v for k, v in sorted(
        count_dict.items(), key=lambda item: item[1], reverse=True)}
    
    print("5 most common words:")
    for i, kv in enumerate(count_dict.items()):
        print(f"Word '{kv[0]}' occurred {kv[1]} times")
        if i >= 4:
            break
    
    # Question 1.1.4 & 1.1.5
    print("\nQuestion 1.1.4 & 1.1.5: Creating dataset and dataloader")
    dataset = NewsDataset(tokenised_headlines, word_to_int)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                          collate_fn=collate_fn, shuffle=True)
    
    '''
    Model & Training
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Standard BPTT Training
    print("\nTraining with standard BPTT:")
    model = NewsLSTM(len(word_to_int), hidden_size=1024, emb_dim=150, n_layers=1, dropout_p=0.2)
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_int[PAD])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses_bptt, perplexities_bptt = train_model(
        model, dataloader, num_epochs=14,
        criterion=criterion, optimizer=optimizer,
        device=device, word_to_int=word_to_int,
        int_to_word=int_to_word, clip=1
    )
    
    # TBPTT Training
    print("\nTraining with TBPTT:")
    model2 = NewsLSTM(len(word_to_int), hidden_size=2048, emb_dim=150, n_layers=1, dropout_p=0.2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    model2, losses_tbptt, perplexities_tbptt = train_with_TBBTT(
        7, model2, dataloader, criterion, word_to_int,
        int_to_word, optimizer2, chunk_size=9, device=device, clip=1
    )
    
    '''
    Visualization
    '''
    # Plot BPTT results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses_bptt, label="Loss")
    plt.axhline(y=LOSS_VALID_THRESHOLD_BPTT, color='r', linestyle='--', label='Target Threshold')
    plt.title('BPTT Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(perplexities_bptt)
    plt.title('BPTT Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig('bptt_training.png')
    plt.show()
    
    # Plot TBPTT results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses_tbptt, label="Loss")
    plt.axhline(y=LOSS_VALID_THRESHOLD_TBPTT, color='r', linestyle='--', label='Target Threshold')
    plt.title('TBPTT Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(perplexities_tbptt)
    plt.title('TBPTT Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.tight_layout()
    plt.savefig('tbptt_training.png')
    plt.show()
    
    '''
    Final Evaluation
    '''
    print("\nFinal Evaluation:")
    
    def create_sentences_after_training(model, prompt, word_to_int, int_to_word, func=sample_argmax, num=3):
        for _ in range(num):
            print(generate_text(model, prompt, word_to_int, int_to_word, device, sampling_func=func))
    
    test_prompts = [
        "Now is the",
        "This way is",
        "He likes to"
    ]
    
    print("\nBPTT Model:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("Greedy sampling:")
        create_sentences_after_training(model, prompt, word_to_int, int_to_word)
        print("Random sampling:")
        create_sentences_after_training(model, prompt, word_to_int, int_to_word, func=random_sample_next)
    
    print("\nTBPTT Model:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("Greedy sampling:")
        create_sentences_after_training(model2, prompt, word_to_int, int_to_word)
        print("Random sampling:")
        create_sentences_after_training(model2, prompt, word_to_int, int_to_word, func=random_sample_next)