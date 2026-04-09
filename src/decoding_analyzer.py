"""
Decoding Analyzer: Core analysis logic for comparing Coconut and Verbal CoT.
"""

import numpy as np
from scipy import stats
from collections import defaultdict


class DecodingAnalyzer:
    """Analyzer for comparing decoding results from Coconut and Verbal CoT."""
    
    def __init__(self):
        """Initialize analyzer."""
        self.coconut_results = []
        self.verbal_results = []
    
    def add_results(self, coconut_results, verbal_results):
        """Add results from both models."""
        self.coconut_results = coconut_results
        self.verbal_results = verbal_results
    
    def analyze_candidate_overlap(self, candidates_list, top_n=5):
        """
        Analyze overlap between candidate sets across steps.
        Returns Jaccard similarity between consecutive steps.
        """
        overlaps = []
        for i in range(len(candidates_list) - 1):
            set_i = set([item[0] for item in candidates_list[i]['cosine_top_k'][:top_n]])
            set_j = set([item[0] for item in candidates_list[i+1]['cosine_top_k'][:top_n]])
            
            if set_i and set_j:
                jaccard = len(set_i & set_j) / len(set_i | set_j)
                overlaps.append(jaccard)
        
        return overlaps
    
    def compute_gt_in_topk_rate(self):
        """Compute ground truth in top-k rates."""
        coco_gt_in_cos = []
        coco_gt_in_logit = []
        verb_gt_in_cos = []
        verb_gt_in_logit = []
        
        for res in self.coconut_results:
            for thought in res['thoughts']:
                if thought['gt_concept']:
                    coco_gt_in_cos.append(thought['gt_in_cos_top_k'])
                    coco_gt_in_logit.append(thought['gt_in_logit_top_k'])
        
        for res in self.verbal_results:
            for step in res['steps']:
                if step['gt_concept']:
                    verb_gt_in_cos.append(step['gt_in_cos_top_k'])
                    verb_gt_in_logit.append(step['gt_in_logit_top_k'])
        
        return {
            'coconut': {
                'cosine': (np.mean(coco_gt_in_cos) if coco_gt_in_cos else None, 
                          sum(coco_gt_in_cos), len(coco_gt_in_cos)),
                'logit': (np.mean(coco_gt_in_logit) if coco_gt_in_logit else None,
                         sum(coco_gt_in_logit), len(coco_gt_in_logit))
            },
            'verbal': {
                'cosine': (np.mean(verb_gt_in_cos) if verb_gt_in_cos else None,
                          sum(verb_gt_in_cos), len(verb_gt_in_cos)),
                'logit': (np.mean(verb_gt_in_logit) if verb_gt_in_logit else None,
                         sum(verb_gt_in_logit), len(verb_gt_in_logit))
            }
        }
    
    def compute_entropy_stats(self):
        """Compute entropy statistics."""
        coco_entropy_cos = []
        coco_entropy_logit = []
        verb_entropy_cos = []
        verb_entropy_logit = []
        
        for res in self.coconut_results:
            for thought in res['thoughts']:
                coco_entropy_cos.append(thought['entropy_cos'])
                coco_entropy_logit.append(thought['entropy_logit'])
        
        for res in self.verbal_results:
            for step in res['steps']:
                verb_entropy_cos.append(step['entropy_cos'])
                verb_entropy_logit.append(step['entropy_logit'])
        
        stats_dict = {
            'coconut': {
                'cosine': {'mean': np.mean(coco_entropy_cos) if coco_entropy_cos else None,
                          'std': np.std(coco_entropy_cos) if coco_entropy_cos else None,
                          'n': len(coco_entropy_cos)},
                'logit': {'mean': np.mean(coco_entropy_logit) if coco_entropy_logit else None,
                         'std': np.std(coco_entropy_logit) if coco_entropy_logit else None,
                         'n': len(coco_entropy_logit)}
            },
            'verbal': {
                'cosine': {'mean': np.mean(verb_entropy_cos) if verb_entropy_cos else None,
                          'std': np.std(verb_entropy_cos) if verb_entropy_cos else None,
                          'n': len(verb_entropy_cos)},
                'logit': {'mean': np.mean(verb_entropy_logit) if verb_entropy_logit else None,
                         'std': np.std(verb_entropy_logit) if verb_entropy_logit else None,
                         'n': len(verb_entropy_logit)}
            }
        }
        
        # Perform t-test if both have data
        t_test_result = None
        if coco_entropy_logit and verb_entropy_logit:
            t_stat, p_val = stats.ttest_ind(coco_entropy_logit, verb_entropy_logit)
            t_test_result = {'t_stat': t_stat, 'p_val': p_val}
        
        return stats_dict, t_test_result
    
    def compute_adjacent_similarities(self):
        """Compute adjacent step similarity statistics."""
        coco_adj_sims = []
        verb_adj_sims = []
        
        for res in self.coconut_results:
            coco_adj_sims.extend(res['adjacent_similarities'])
        
        for res in self.verbal_results:
            verb_adj_sims.extend(res['adjacent_similarities'])
        
        stats_dict = {
            'coconut': {
                'mean': np.mean(coco_adj_sims) if coco_adj_sims else None,
                'std': np.std(coco_adj_sims) if coco_adj_sims else None,
                'n': len(coco_adj_sims)
            },
            'verbal': {
                'mean': np.mean(verb_adj_sims) if verb_adj_sims else None,
                'std': np.std(verb_adj_sims) if verb_adj_sims else None,
                'n': len(verb_adj_sims)
            }
        }
        
        # Perform t-test if both have data
        t_test_result = None
        if coco_adj_sims and verb_adj_sims:
            t_stat, p_val = stats.ttest_ind(coco_adj_sims, verb_adj_sims)
            t_test_result = {'t_stat': t_stat, 'p_val': p_val}
        
        return stats_dict, t_test_result
    
    def compute_unique_candidates(self):
        """Compute number of unique candidates per step."""
        coco_unique_per_step = []
        verb_unique_per_step = []
        
        for res in self.coconut_results:
            for thought in res['thoughts']:
                unique_tokens = set([item[0] for item in thought['cosine_top_k']])
                coco_unique_per_step.append(len(unique_tokens))
        
        for res in self.verbal_results:
            for step in res['steps']:
                unique_tokens = set([item[0] for item in step['cosine_top_k']])
                verb_unique_per_step.append(len(unique_tokens))
        
        return {
            'coconut': {
                'mean': np.mean(coco_unique_per_step) if coco_unique_per_step else None,
                'std': np.std(coco_unique_per_step) if coco_unique_per_step else None,
                'n': len(coco_unique_per_step)
            },
            'verbal': {
                'mean': np.mean(verb_unique_per_step) if verb_unique_per_step else None,
                'std': np.std(verb_unique_per_step) if verb_unique_per_step else None,
                'n': len(verb_unique_per_step)
            }
        }
    
    def compute_candidate_overlap(self):
        """Compute Jaccard similarity of candidates between consecutive steps."""
        coco_overlaps = []
        verb_overlaps = []
        
        for res in self.coconut_results:
            overlaps = self.analyze_candidate_overlap(res['thoughts'])
            coco_overlaps.extend(overlaps)
        
        for res in self.verbal_results:
            overlaps = self.analyze_candidate_overlap(res['steps'])
            verb_overlaps.extend(overlaps)
        
        return {
            'coconut': {
                'mean': np.mean(coco_overlaps) if coco_overlaps else None,
                'std': np.std(coco_overlaps) if coco_overlaps else None,
                'n': len(coco_overlaps)
            },
            'verbal': {
                'mean': np.mean(verb_overlaps) if verb_overlaps else None,
                'std': np.std(verb_overlaps) if verb_overlaps else None,
                'n': len(verb_overlaps)
            }
        }
    
    def generate_summary(self):
        """Generate complete summary of all analyses."""
        gt_stats = self.compute_gt_in_topk_rate()
        entropy_stats, entropy_ttest = self.compute_entropy_stats()
        adj_stats, adj_ttest = self.compute_adjacent_similarities()
        unique_stats = self.compute_unique_candidates()
        overlap_stats = self.compute_candidate_overlap()
        
        return {
            'ground_truth_in_topk': gt_stats,
            'entropy': entropy_stats,
            'entropy_ttest': entropy_ttest,
            'adjacent_similarity': adj_stats,
            'adjacent_similarity_ttest': adj_ttest,
            'unique_candidates': unique_stats,
            'candidate_overlap': overlap_stats,
            'n_problems': len(self.coconut_results),
            'n_steps_coconut': sum(len(res['thoughts']) for res in self.coconut_results),
            'n_steps_verbal': sum(len(res['steps']) for res in self.verbal_results)
        }