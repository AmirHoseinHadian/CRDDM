## Is CRDDM right for my data?

Use the checklist below to quickly determine whether CRDDM is appropriate for your dataset:

### ✔ Suitable for CRDDM if:

- You have **trial-level response times** for each decision.
- You have **trial-level continuous response values** (e.g., angle, position, 2D coordinate).
- Response times meaningfully reflect **decision dynamics** (i.e., evidence accumulation).
- You are interested in modeling **latent decision processes** (e.g., drift rate, threshold, non-decision time).
- Your task involves a well-defined response geometry (circular, bounded, or multi-dimensional).

### ✖ CRDDM may not be appropriate if:

- Response times were **not recorded**.
- Response times are artificially constrained (e.g., fixed response windows, delayed response prompts).
- The task was not designed to measure **decision latency**.
- You are only interested in modeling response error distributions without reference to decision time.

If you are unsure whether diffusion-based modeling is appropriate for your task,
consider whether response times are theoretically meaningful in your paradigm.
If not, alternative descriptive or static models may be more suitable.


Follow the steps below:

**1️⃣ Do you have trial-level response times (RTs) for each decision?**

- ❌ **No** → CRDDM is not appropriate.  
  Consider descriptive or static models of response distributions.

- ✔ **Yes** → Go to Step 2.

---

**2️⃣ Do you have trial-level continuous response values?**  
(e.g., angle, slider position, 2D coordinate)

- ❌ **No** → CRDDM is not appropriate.  
  Traditional discrete-choice diffusion models may be more suitable.

- ✔ **Yes** → Go to Step 3.

---

**3️⃣ Do response times meaningfully reflect decision dynamics?**  
(i.e., are they not artificially constrained, delayed, or fixed by design?)

- ❌ **No** → Diffusion-based modeling may not be theoretically justified.

- ✔ **Yes** → Go to Step 4.

---

**4️⃣ Are you interested in modeling latent decision processes?**  
(e.g., drift rate, threshold, non-decision time, collapsing boundaries)

- ❌ **No** → A descriptive model may be sufficient.

- ✔ **Yes** → ✅ **CRDDM is likely appropriate for your data.**

``` mermaid
graph 
    A[Start] --> B{Do you have trial-level<br/>response times?};
    B -- No --> B0[CRDDM not appropriate<br/>Use models that do not rely on RTs];
    B -- Yes --> C{Do you have trial-level<br/>continuous responses?};
    C -- No --> C0[CRDDM not appropriate<br/>Consider discrete-choice diffusion models]
    C -- Yes --> D{Are RTs meaningful measures<br/>of decision latency?}

    D -- No --> D0[Diffusion modeling may not be justified<br/>RTs may not reflect accumulation dynamics]
    D -- Yes --> E{Do you want to model latent<br/>decision processes?}

    E -- No --> E0[Descriptive or static models may be sufficient]
    E -- Yes --> F([CRDDM is likely appropriate])
```