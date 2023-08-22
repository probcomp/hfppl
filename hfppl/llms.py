import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TokenSequence:
    def __init__(self, lm, seq=None):
        self.lm = lm
        if seq is None:
            self.seq = [lm.tokenizer.bos_token_id]
        elif isinstance(seq, str):
            self.seq = self.lm.tokenizer.encode(seq)
        else:
            self.seq = seq
    
    def __str__(self):
        return self.lm.tokenizer.decode(self.seq)
    
    def __iadd__(self, other):
        if isinstance(other, Token):
            assert other.lm is self.lm
            self.seq.append(other.token_id)
        elif isinstance(other, TokenSequence):
            assert other.lm is self.lm
            self.seq.extend(other.seq)
        elif isinstance(other, str):
            self.seq.extend(self.lm.tokenizer.encode(other, add_special_tokens=False))
        elif isinstance(other, int):
            self.seq.append(other)
        else:
            raise RuntimeError(f"Addition not supported on {type(other)}")
        return self
    
    def __radd__(self, other):
        if isinstance(other, Token):
            assert other.lm is self.lm
            return TokenSequence(self.lm, [other.token_id, *self.seq])
        elif isinstance(other, TokenSequence):
            assert other.lm is self.lm
            return TokenSequence(self.lm, other.seq + self.seq)
        elif isinstance(other, str):
            return TokenSequence(self.lm, self.lm.tokenizer.encode(other, add_special_tokens=False) + self.seq)
        elif isinstance(other, int):
            return TokenSequence(self.lm, [other, *self.seq])
        else:
            raise RuntimeError(f"Addition not supported on {type(other)}")
    
    def __add__(self, other):
        s = TokenSequence(self.lm, self.seq)
        s += other
        return s

class Token:
    def __init__(self, lm, token_id, token_str):
        self.lm        = lm
        self.token_id  = token_id
        self.token_str = token_str
    
    # Adding tokens
    def __add__(self, other):
        s = TokenSequence(self.lm, [self.token_id])
        s += other
        return s

    def __radd__(self, other):
        s = TokenSequence(self.lm, [self.token_id])
        return other + s
    
    # Support checking for EOS
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.lm is other.lm and self.token_id == other.token_id
        elif isinstance(other, int):
            return self.token_id == other
        else:
            return self.token_str == other

    def __str__(self):
        return self.token_str
    
    def __repr__(self):
        return f"<{self.token_str}|{self.token_id}>"


class TokenTrie:
    # Trie of tokens.
    # For now, we just store the next-token distribution for each token, 
    # if it has been evaluated.

    def __init__(self, parent=None, logprobs=None):                     
        self.children = {} # maps token ID to child
        self.logprobs = logprobs  # for next token
    
    def has_token(self, token_id):
        return token_id in self.children
    
    def get_token(self, token_id):
        return self.children[token_id]
    
    def add_token(self, token_id, logprobs=None):
        self.children[token_id] = TokenTrie(self, logprobs)
        return self.children[token_id]
    

class CachedCausalLM:
    
    @classmethod
    def from_pretrained(cls, model_id, auth_token=False, load_in_8bit=True):
        
        with torch.no_grad():
            tok = AutoTokenizer.from_pretrained(model_id, use_auth_token=auth_token)
            mod = AutoModelForCausalLM.from_pretrained(model_id, do_sample=True, use_auth_token=auth_token, device_map="auto", load_in_8bit=load_in_8bit)
        
        return CachedCausalLM(mod, tok)
    
    @torch.no_grad()
    def __init__(self, hf_model, hf_tokenizer):
        self.model = hf_model
        self.tokenizer = hf_tokenizer
        self.device = hf_model.device
        
        # TODO: remove required BOS token
        if self.tokenizer.bos_token_id is None:
            raise RuntimeError("Causal LM has no BOS token, distribution of first word unclear")
        
        # Evaluate BOS token
        logits   = self.model(torch.tensor([[self.tokenizer.bos_token_id]]).to(self.model.device)).loss['logits'][0][0]
        logprobs = torch.log_softmax(logits, 0)
        
        self.cache = TokenTrie(None, logprobs.cpu().numpy())
    
    def __deepcopy__(self, memo):
        return self
    
    # Walks the cache, adds to it if necessary
    @torch.no_grad()
    def next_token_logprobs(self, token_ids):
        
        # Ensure that token list begins with BOS
        assert token_ids[0] == self.tokenizer.bos_token_id
        
        
        # Walk while tokens can be found
        node             = self.cache
        next_token_index = 1

        while next_token_index < len(token_ids):
            if node.has_token(token_ids[next_token_index]):
                node = node.get_token(token_ids[next_token_index])
                next_token_index += 1
            else:
                break
        
        # If we processed all tokens, then we're done.
        if next_token_index == len(token_ids):
            return node.logprobs
        
        # Otherwise, run the model...
        prompt = torch.tensor([token_ids]).to(self.device)
        logits = self.model(prompt).loss['logits'][0]
        
        # Create new nodes
        for j in range(next_token_index, len(token_ids)):
            token_id     = token_ids[j]
            token_logits = logits[j]
            token_logprobs  = torch.log_softmax(token_logits, 0)
            
            node = node.add_token(token_id, token_logprobs.cpu().numpy())
            
        return node.logprobs