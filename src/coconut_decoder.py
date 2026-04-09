"""
Coconut Decoder: Utilities for decoding continuous thoughts from Coconut model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.probe_utils import extract_derived_concept


class CoconutDecoder:
    """Decoder for analyzing continuous thoughts in Coconut model."""
    
    def __init__(self, model, tokenizer, device="cuda"):
        """
        Initialize Coconut decoder.
        
        Args:
            model: Coconut model
            tokenizer: Tokenizer for decoding
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Get special token IDs
        self.latent_id = tokenizer.convert_tokens_to_ids("<latent>")
        self.start_id = tokenizer.convert_tokens_to_ids("<start_latent>")
        self.end_id = tokenizer.convert_tokens_to_ids("<end_latent>")
    
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
        if hasattr(self.model, 'base_causallm'):
            embed_matrix = self.model.base_causallm.transformer.wte.weight
            lm_head = self.model.base_causallm.lm_head
        else:
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
        
        # Process cosine similarity candidates
        cos_candidates = self._process_cosine_candidates(
            top_k_cos_vals, top_k_cos_idx, special_ids, k
        )
        
        # Process LM head candidates
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
    
    def extract_continuous_thoughts(self, problem_row):
        """
        Extract continuous thought vectors for a problem.
        
        Args:
            problem_row: DataFrame row with problem data
        
        Returns:
            List of thought vectors
        """
        n_hops = problem_row["n_hops"]
        
        # Build input with latent tokens
        question_ids = self.tokenizer.encode(
            problem_row["question"] + "\n", add_special_tokens=True
        )
        input_ids = question_ids + [self.start_id] + [self.latent_id] * n_hops + [self.end_id]
        input_tensor = torch.tensor(input_ids).unsqueeze(0).to(self.device)
        attn_mask = torch.ones_like(input_tensor)
        
        with torch.no_grad():
            _, filled_embeds = self.model.generate(
                input_ids=input_tensor,
                attention_mask=attn_mask,
                max_new_tokens=20,
                output_embedding=True,
            )
        
        # Extract each continuous thought
        latent_start_pos = len(question_ids) + 1
        thought_vectors = []
        for pos in range(latent_start_pos, latent_start_pos + n_hops):
            if pos < filled_embeds.shape[1]:
                vec = filled_embeds[0, pos, :].detach().cpu()
                thought_vectors.append(vec)
        
        return thought_vectors
    
    def analyze_thoughts(self, problem_row, k=10):
        """
        Analyze continuous thoughts for a single problem.
        
        Args:
            problem_row: DataFrame row with problem data
            k: Number of top candidates to return
        
        Returns:
            Dictionary with analysis results
        """
        n_hops = problem_row["n_hops"]
        steps = problem_row["steps"]
        ground_truth_concepts = [extract_derived_concept(step) for step in steps]
        
        # Extract thought vectors
        thought_vectors = self.extract_continuous_thoughts(problem_row)
        
        # Decode each thought
        thoughts_analysis = []
        for i, vec in enumerate(thought_vectors):
            candidates = self.get_top_k_candidates(vec, k=k)
            
            # Check if ground truth is in top-k
            gt_concept = ground_truth_concepts[i] if i < len(ground_truth_concepts) else None
            gt_in_cos_top_k, gt_in_logit_top_k = self._check_gt_in_candidates(
                gt_concept, candidates
            )
            
            # Compute entropy
            entropy_cos = self._compute_entropy(candidates['cosine'], method='cosine')
            entropy_logit = self._compute_entropy(candidates['lm_head'], method='lm_head')
            
            # Format candidates for storage
            cos_top_k = [(t, round(v, 3)) for t, v in candidates['cosine']]
            logit_top_k = [(t, round(v, 3), round(p, 3)) for t, v, p in candidates['lm_head']]
            
            thoughts_analysis.append({
                'step': i,
                'gt_concept': gt_concept,
                'cosine_top_k': cos_top_k,
                'logit_top_k': logit_top_k,
                'gt_in_cos_top_k': gt_in_cos_top_k,
                'gt_in_logit_top_k': gt_in_logit_top_k,
                'entropy_cos': float(entropy_cos),
                'entropy_logit': float(entropy_logit),
                'vector_norm': float(torch.norm(vec)),
            })
        
        # Compute pairwise cosine similarities between adjacent thoughts
        adjacent_similarities = self._compute_adjacent_similarities(thought_vectors)
        
        return {
            'problem_id': int(problem_row.name) if hasattr(problem_row, 'name') else str(problem_row.name),
            'n_hops': n_hops,
            'ground_truth_concepts': ground_truth_concepts,
            'thoughts': thoughts_analysis,
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
            sim = cosine_similarity(
                vectors[i].numpy().reshape(1, -1),
                vectors[i+1].numpy().reshape(1, -1)
            )[0][0]
            similarities.append(float(sim))
        return similarities