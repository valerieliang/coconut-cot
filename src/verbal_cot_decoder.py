"""
Verbal CoT Decoder: Utilities for decoding representations from verbal Chain-of-Thought.
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.probe_utils import extract_derived_concept


class VerbalCoTDecoder:
    """Decoder for analyzing representations in verbal CoT model."""
    
    def __init__(self, model, tokenizer, device="cuda", layer=-1):
        """
        Initialize verbal CoT decoder.
        
        Args:
            model: Verbal CoT model
            tokenizer: Tokenizer for decoding
            device: Device to run on
            layer: Layer index to extract hidden states from
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.layer = layer
    
    def get_top_k_candidates(self, embedding_vector, k=10, exclude_special=True):
        """
        Project an embedding vector back to vocabulary space and return top-k tokens.
        
        Args:
            embedding_vector: Vector in embedding space (d_model,)
            k: Number of top candidates to return
            exclude_special: Whether to exclude special tokens
        
        Returns:
            Dictionary with 'cosine' and 'lm_head' candidates
        """
        # Get the embedding matrix and LM head
        embed_matrix = self.model.transformer.wte.weight
        lm_head = self.model.lm_head
        
        # Convert embedding to tensor
        if isinstance(embedding_vector, np.ndarray):
            emb = torch.from_numpy(embedding_vector).float()
        else:
            emb = embedding_vector.float()
        
        # Move to same device as model
        emb = emb.to(embed_matrix.device)
        
        # Cosine similarity to all embeddings
        import torch.nn.functional as F
        emb_norm = F.normalize(emb.unsqueeze(0), p=2, dim=-1)
        embed_norm = F.normalize(embed_matrix, p=2, dim=-1)
        cos_sim = torch.mm(emb_norm, embed_norm.t()).squeeze(0)
        
        # Project through LM head
        logits = lm_head(emb.unsqueeze(0)).squeeze(0)
        
        # Get top-k by cosine similarity
        top_k_cos_vals, top_k_cos_idx = torch.topk(cos_sim, k=k * 2)
        
        # Get top-k by LM head logits
        top_k_logit_vals, top_k_logit_idx = torch.topk(logits, k=k * 2)
        
        # Filter special tokens
        special_ids = self._get_special_token_ids() if exclude_special else set()
        
        # Process candidates
        cos_candidates = self._process_cosine_candidates(
            top_k_cos_vals, top_k_cos_idx, special_ids, k
        )
        logit_candidates = self._process_logit_candidates(
            top_k_logit_vals, top_k_logit_idx, special_ids, k
        )
        
        return {
            'cosine': cos_candidates,
            'lm_head': logit_candidates
        }
    
    def _get_special_token_ids(self):
        """Get set of special token IDs to exclude."""
        special_tokens = [
            '<|endoftext|>', '<latent>', '<start_latent>', '<end_latent>',
            '<pad>', '<s>', '</s>', '<unk>'
        ]
        special_ids = set()
        for token in special_tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            special_ids.update(ids)
        
        # Also filter newline and space tokens
        special_ids.update(self.tokenizer.encode('\n', add_special_tokens=False))
        special_ids.update(self.tokenizer.encode(' ', add_special_tokens=False))
        special_ids.update(self.tokenizer.encode('  ', add_special_tokens=False))
        
        return special_ids
    
    def _process_cosine_candidates(self, values, indices, special_ids, k):
        """Process cosine similarity candidates."""
        candidates = []
        for val, idx in zip(values, indices):
            idx_int = idx.item()
            if idx_int in special_ids:
                continue
            token = self.tokenizer.decode([idx_int]).strip()
            if token and not token.startswith('<') and not token.startswith('Ġ'):
                token = token.replace('Ġ', ' ').strip()
                if token:
                    candidates.append((token, val.item()))
            if len(candidates) >= k:
                break
        return candidates
    
    def _process_logit_candidates(self, values, indices, special_ids, k):
        """Process LM head logit candidates."""
        import torch.nn.functional as F
        probs = F.softmax(values, dim=0)
        candidates = []
        for val, prob, idx in zip(values, probs, indices):
            idx_int = idx.item()
            if idx_int in special_ids:
                continue
            token = self.tokenizer.decode([idx_int]).strip()
            if token and not token.startswith('<') and not token.startswith('Ġ'):
                token = token.replace('Ġ', ' ').strip()
                if token:
                    candidates.append((token, val.item(), prob.item()))
            if len(candidates) >= k:
                break
        return candidates
    
    def extract_step_representations(self, problem_row):
        """
        Extract hidden state representations for each reasoning step.
        
        Args:
            problem_row: DataFrame row with problem data
        
        Returns:
            List of step vectors
        """
        steps = problem_row["steps"]
        question = problem_row["question"]
        
        step_vectors = []
        for i, step in enumerate(steps):
            # Build text up to this step
            text_parts = [question] + steps[:i+1]
            text = "\n".join(text_parts)
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            captured = {}
            def hook_fn(module, input, output):
                captured["hidden"] = output[0].detach().cpu()
            
            handle = self.model.transformer.h[self.layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                self.model(**inputs)
            handle.remove()
            
            # Take the last position's hidden state
            hidden = captured["hidden"][0]
            step_vectors.append(hidden[-1, :])
        
        return step_vectors
    
    def analyze_steps(self, problem_row, k=10):
        """
        Analyze representations for each step of a problem.
        
        Args:
            problem_row: DataFrame row with problem data
            k: Number of top candidates to return
        
        Returns:
            Dictionary with analysis results
        """
        steps = problem_row["steps"]
        n_hops = problem_row["n_hops"]
        ground_truth_concepts = [extract_derived_concept(step) for step in steps]
        
        # Extract step vectors
        step_vectors = self.extract_step_representations(problem_row)
        
        # Decode each step's representation
        steps_analysis = []
        for i, vec in enumerate(step_vectors):
            candidates = self.get_top_k_candidates(vec, k=k)
            
            gt_concept = ground_truth_concepts[i] if i < len(ground_truth_concepts) else None
            gt_in_cos_top_k, gt_in_logit_top_k = self._check_gt_in_candidates(
                gt_concept, candidates
            )
            
            entropy_cos = self._compute_entropy(candidates['cosine'], method='cosine')
            entropy_logit = self._compute_entropy(candidates['lm_head'], method='lm_head')
            
            cos_top_k = [(t, round(v, 3)) for t, v in candidates['cosine'][:5]]
            logit_top_k = [(t, round(v, 3), round(p, 3)) for t, v, p in candidates['lm_head'][:5]]
            
            steps_analysis.append({
                'step': i,
                'gt_concept': gt_concept,
                'cosine_top_k': cos_top_k,
                'logit_top_k': logit_top_k,
                'gt_in_cos_top_k': gt_in_cos_top_k,
                'gt_in_logit_top_k': gt_in_logit_top_k,
                'entropy_cos': float(entropy_cos),
                'entropy_logit': float(entropy_logit),
            })
        
        # Compute adjacent similarities
        adjacent_similarities = self._compute_adjacent_similarities(step_vectors)
        
        return {
            'problem_id': int(problem_row.name) if hasattr(problem_row, 'name') else str(problem_row.name),
            'n_hops': n_hops,
            'ground_truth_concepts': ground_truth_concepts,
            'steps': steps_analysis,
            'adjacent_similarities': adjacent_similarities,
            'mean_adjacent_similarity': float(np.mean(adjacent_similarities)) if adjacent_similarities else None,
        }
    
    def _check_gt_in_candidates(self, gt_concept, candidates):
        """Check if ground truth concept appears in candidates."""
        gt_in_cos = False
        gt_in_logit = False
        
        if gt_concept:
            for token, _ in candidates['cosine']:
                if gt_concept.lower() in token.lower():
                    gt_in_cos = True
                    break
            for item in candidates['lm_head']:
                token = item[0]
                if gt_concept.lower() in token.lower():
                    gt_in_logit = True
                    break
        
        return gt_in_cos, gt_in_logit
    
    @staticmethod
    def _compute_entropy(candidates, method='lm_head'):
        """Compute entropy of candidate distribution."""
        if method == 'lm_head':
            if not candidates or len(candidates[0]) < 3:
                return 0.0
            probs = [c[2] for c in candidates if c[2] is not None]
        else:
            if not candidates:
                return 0.0
            vals = [c[1] for c in candidates]
            vals = np.array(vals)
            probs = np.exp(vals) / np.sum(np.exp(vals))
        
        if len(probs) == 0:
            return 0.0
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    @staticmethod
    def _compute_adjacent_similarities(vectors):
        """Compute cosine similarities between adjacent vectors."""
        similarities = []
        for i in range(len(vectors) - 1):
            if vectors[i] is not None and vectors[i+1] is not None:
                sim = cosine_similarity(
                    vectors[i].numpy().reshape(1, -1),
                    vectors[i+1].numpy().reshape(1, -1)
                )[0][0]
                similarities.append(float(sim))
        return similarities