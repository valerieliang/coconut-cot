# A Causal Comparison of Coconut vs. Verbal Chain-of-Thought

Do language models actually use the reasoning they write out, or is that reasoning just a surface-level explanation?

This project compares two approaches:

* **Verbal Chain-of-Thought (CoT):** reasoning expressed in natural language
* **Coconut (Continuous Thought):** reasoning through latent vectors instead of text

Using activation steering, we test whether intervening on intermediate reasoning changes the model’s final answer.

---

## Key Idea

Standard CoT produces a sequence of text steps:

```
Step 1 → Step 2 → Step 3 → Answer
```

Coconut instead propagates continuous vectors:

```
c₁ → Transformer → c₂ → Transformer → c₃ → ... → Answer
```

If intermediate reasoning is causally important, then perturbing it should affect the final output. If it is not, the answer should remain stable.

---

## Hypotheses

| Hypothesis | Prediction                                         | Result        |
| ---------- | -------------------------------------------------- | ------------- |
| Bottleneck | Coconut is more sensitive to interventions         | Confirmed     |
| Separation | CoT reasoning can change without affecting answers | Confirmed     |
| Scaling    | Longer chains increase the gap                     | Not supported |

---

## Experimental Setup

### Dataset

We use **PrOntoQA**, a controlled dataset with multi-step logical structure.

### Steering Method

For each example:

1. Select an intermediate logical statement
2. Extract:

   * `v+`: representation when the statement is true
   * `v-`: representation when it is false
3. Compute a steering vector:

```
delta = v- - v+
```

We inject this during inference:

* **Coconut:** perturb the continuous thought vector
* **Verbal CoT:** perturb the hidden state at the end of a reasoning sentence

---

## Why This Matters

If model outputs are used for monitoring or safety:

* Verbal reasoning alone may not reflect the true decision process
* A model can produce coherent explanations that do not determine its answer

Coconut-style reasoning suggests a direction where intermediate states are more tightly coupled to outcomes, but introduces challenges for interpretability.