# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

We already imported 20k products and 200k reviews in milestone 2. We did not match products with reviews; instead, we left approximately half of the products without any reviews, while other products could have a range of reviews assigned. We thought this reflected differences in popularity among Amazon products online. No changes to sampling were made.

### LLM Experiment

**Models Compared:**
| Model | Family | Architecture | Parameters |
|-------|--------|--------------|------------|
| `qwen/qwen3-32b` | Qwen3 (Alibaba Cloud) | Dense transformer | 32B |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 (Meta) | Mixture-of-Experts (MoE) | 17B active (109B total) with 16 experts |

We chose `meta-llama/llama-4-scout-17b-16e-instruct` as the comparison model because it comes from a different family and uses a different architecture than `qwen/qwen3-32b`. Qwen3-32B is a dense transformer where every token passes through all 32B parameters whereas Llama 4 Scout uses a Mixture-of-Experts (MoE) design, routing inputs through specialized subnetworks called experts with only 17B parameters active at one time, out of 109B total. Since both models were released in April 2025, any differences in query results should reflect upon the architecture instead of a model being newer. Both models were accessed via Groq's free API tier using identical settings for a fair comparison (temperature=0.2, max_tokens=3000).

**Prompt Used:**

System prompt:

    You are a helpful Amazon shopping assistant.
    Answer the question using ONLY the following context (real product reviews + metadata).
    Always cite the product ASIN when possible.
    Be concise and specific.

Prompt structure:

    {system_prompt}

    context:
    {retrieved context — top 5 hybrid results with title, ASIN, rating, review text}

    question:
    {query}

    Answer based on the Amazon datasets:

**Queries Used:**
1. Washing machine
2. Stainless steel coffee maker
3. Magic Bullet
4. kitchen device to heat food quickly
5. Best appliances for a small apartment

**Results:**

#### Query: Washing machine

*qwen3-32b:*

> Based on the Amazon datasets, here are two notable washing machine options:  
> 
> 1. **Formemory Portable Folding Bucket Turbo Ultrasonic Washing Machine** (ASIN: B08FMND788):  
>    - **Rating**: 5.0 (1 review, 2 helpful votes).  
>    - **Key features**: Compact, portable, and folds for storage. Ideal for small/light loads (e.g., pet clothing). Includes remote control.  
> 
> 2. **Midea 1.6 CF Portable Washing Machine** (ASIN: B00DJF6296):  
>    - **Rating**: 4.5 (8 reviews, 4 helpful votes).  
>    - **Key features**: Handles larger loads (e.g., queen-sized sheets, jeans). Users note it requires a 12-foot water hose extension for drainage.  
> 
> For a budget-friendly option with no reviews yet, consider the **Pink Mini USB Portable Washing Machine** (ASIN: B09L1LHFP8, $17.89).

*llama-4-scout:*

> If you're looking for a compact washing machine, consider the Midea 1.6 CF Portable Washing Machine Washer (ASIN: B00DJF6296) with a 4.5-star rating. It's a great option for handling larger loads, but note that it's not very portable without additional modifications, such as adding wheels.
> 
> For smaller, lightweight clothes, the Mini Washing Machine by Formemory (ASIN: B08FMND788) is a 5.0-star rated, portable, and foldable option that's perfect for travel or small loads, such as washing dog clothes.
> 
> The price for the Mini Ultrasonic Washing Machine (ASIN: B09L1LHFP8) is $17.89, but it has no reviews.
> 
> Let me know if you'd like more information!

---

#### Query: Stainless steel coffee maker

*qwen3-32b:*

> For stainless steel coffee makers, consider the following options based on the dataset:
> 
> 1. **Aeropress Coffee Maker** (ASIN: B00WT80GUU)  
>    - **Rating**: 5.0 (1 review)  
>    - **Key Features**: Reduces coffee acidity (ideal for acid reflux), portable, and preferred over an espresso maker by a reviewer. Originally purchased for camping but now used daily.  
> 
> 2. **Stainless Steel Coffee Filter for Espresso Machines** (ASIN: B08TW7J1JQ)  
>    - **Rating**: 5.0 (1 review)  
>    - **Key Features**: Double-layer pressurized filter basket compatible with portafilters; works "as good as the original." Priced at $8.79.  
> 
> 3. **Pour Over Coffee Dripper** (ASIN: B07X7MH6BG)  
>    - **Price**: $15.99  
>    - **Key Features**: Paperless, reusable, single-cup stainless steel dripper with a coffee spoon. No reviews yet.  
> 
> For a larger pour-over option, the **Coffee Filter Cone with Cup Stand** (ASIN: B08FBFCSRQ) (4.0 rating, 1 review) is noted for its two-stage filtration and 6-cup capacity.  
> 
> The **Travel Glass Pour Over Set** (ASIN: B0BHTY25D2) ($24.99) is another stainless steel option but has no reviews.  
> 
> **Top Recommendation**: The Aeropress (B00WT80GUU) is highly rated for versatility and health benefits, while the espresso filter (B08TW7J1JQ) is ideal for machine compatibility.

*llama-4-scout:*

> For a stainless steel coffee maker, consider the Aeropress Coffee Maker with 2 Domestic Corner Stainless Steel Filters, ASIN: B00WT80GUU. It's a 5.0 rated product that's been praised for reducing coffee acidity and is versatile enough for both daily use and camping. 
> 
> Alternatively, you may also look into:
> - Pour Over Coffee Dripper Stainless Steel Paperless Reusable Coffee Filter Single Cup Coffee Maker with Coffee Spoon, ASIN: B07X7MH6BG, priced at $15.99.
> - Coffee Filter Stainless Steel, Reusable Pour Over Coffee Filter Cone Coffee Dripper with Cup Stand (Fixed base/6 cup), ASIN: B08FBFCSRQ, with a 4.0 rating.
> - Travel Glass Pour Over Coffee maker Set 4pcs (Gray) | 18/2 Stainless Steel Filter Dripper and Lid | BPA Free Borosilicate Glass Mug | Paperless Drip Coffee Maker Pour Over For Coffee Lovers, ASIN: B0BHTY25D2, priced at $24.99.
> - Stainless Steel Coffee Filter, 51mm Double Layer Pressurized Filter Basket Espresso Filter Basket for Portafilter Coffee Machine(Double Cup), ASIN: B08TW7J1JQ, with a 5.0 rating and priced at $8.79.

---

#### Query: Magic Bullet

*qwen3-32b:*

> The only relevant product related to "Magic Bullet" in the provided context is **Magic Bullet 2 pcs Short Cups Original Replacement Parts by Homeland Housewares** (ASIN: B01C855CYS). This product has no reviews, ratings, or price data available in the dataset.

*llama-4-scout:*

> For Magic Bullet replacement parts, I found a product titled "Magic Bullet 2 pcs Short Cups Original Replacement Parts" with ASIN: **B01C855CYS** by Homeland Housewares. Unfortunately, there are no reviews available for this product. 
> 
> Would you like to explore more options or details about this product?

---

#### Query: kitchen device to heat food quickly

*qwen3-32b:*

> The **QTYANCY Electric Cooktop (ASIN: B0C4L98ZFV)** is a strong option for heating food quickly. Users report it heats water and cooks seafood stock pots rapidly, with 6800W power and even heat distribution. One reviewer called it their "favorite Amazon purchase" for its speed and performance. For specialized quick heating of vegetables, the **Rapid Corn & Potato Cooker (ASIN: B07TKDHFGP)** also works well for microwave cooking, though it’s limited to specific items like corn and potatoes.

*llama-4-scout:*

> For quickly heating food, consider the QTYANCY Electric Cooktop (ASIN: B0C4L98ZFV). A customer review mentions that it "heats quickly and evenly" and can bring a stock pot of water to a boil rapidly. This 30-inch electric smoothtop has 4 elements, 9 heating levels, and a power output of 6800W. It's priced at $249.00.

---

#### Query: Best appliances for a small apartment

*qwen3-32b:*

> Based on the provided data, the best appliances for small apartments are:  
> 
> 1. **GE JGAS02SENSS 24" Stainless Steel Gas Open Burner Range (ASIN: B001C3W83Q)**  
>    - Rated 5.0 (1 review). Praised for reliability, aesthetics, and fit in compact kitchens.  
> 
> 2. **Qzexun Portable 2 in 1 Mini Laundry Washer and Spin Dryer (ASIN: B0B1ZK2JF1)**  
>    - Rated 5.0 (1 review). Ideal for small spaces, saving money on laundry for families.  
> 
> 3. **COYCYQI Mini Fridge (ASIN: B0987DK532)**  
>    - Rated 4.3 (6 reviews). Compact, USB-powered, and dual-function (cool/warm), though some note a short lifespan (3 months reported).  
> 
> The **Premium Mini Fridge (B01128A5LW)** and **LemoHome Dishwasher (B0B4FLTFY6)** lack reviews/data, making them less reliable choices. Prioritize appliances with proven performance in constrained spaces.

*llama-4-scout:*

> For a small apartment, consider the following appliances:
> 
> 1. **Mini Fridge**: 
>    - **COYCYQI Mini Fridge for Skincare (ASIN: B0987DK532)**: A compact, 4L/6 can mini fridge that cools and warms, suitable for small spaces like bedrooms or home offices. It's USB-powered, making it portable for use in cars or homes. Priced at $38.99.
> 
> 2. **Portable Dishwasher**: 
>    - **LemoHome Portable Countertop Dishwasher (ASIN: B0B4FLTFY6)**: Offers 4 washing programs and an air-dry function, ideal for small apartments, dorms, or RVs.
> 
> 3. **Mini Laundry Washer**: 
>    - **Qzexun Portable 2 in 1 Mini Laundry Washer and Spin Dryer Combo (ASIN: B0B1ZK2JF1)**: A 26lb washer and spin dryer combo that's suitable for small apartments, especially for those with limited laundry facilities. It's a 2-in-1 solution that can help save money on laundry services.
> 
> 4. **Compact Stove**: 
>    - **GE JGAS02SENSS 24" Stainless Steel Gas Open Burner Range (ASIN: B001C3W83Q)**: A compact, 24-inch gas stove that's perfect for small kitchens in apartments. It's aesthetically pleasing and functional, with a flawless performance record.
> 
> These appliances are highly rated and suitable for small apartments, offering compact designs and efficient performance.

---

**Key Observations:**

Overall both models performed comparably across the 5 queries. Both produced readable, grounded answers without hallucinating any products outside of the provided context. As noted in the Milestone 2 discussion, when the hybrid retriever returns poor retrieved results, answer quality in both models also drops. For example, in the "Magic Bullet" query where there was only 1 related result (which was a replacement part) in the dataset, both models had no choice but to return this single product.

Of note, Qwen3-32B was more likely to flag products that had no reviews, as seen in queries "Stainless steel coffee maker" and "Best appliances for a small apartment". Interestingly, Llama 4 Scout was more likely to add followup prompts such as "Let me know if you'd like more information!" and "Would you like to explore more options or details about this product?" which gives it a more conversational feel.

Since both models performed similarly overall, the choice of model would come down to preference. We liked how personable Llama 4 Scout felt but Qwen3-32B's behaviour that flags missing reviews from products would be more useful for a product search tool. Since Qwen3-32B is already implemented in the current pipeline, we kept it as is with no changes made.

## Step 2: Additional Feature

### Quantitative Evaluation

We used a custom script-based approach rather than a RADAS workflow because the goal here was to evaluate the information retrieval component of the RAG system without introducing an LLM into the labelling pipeline itself. Codex was used to help brainstorm the purpose of this script and draft it, but was reviewed and appraised.

#### Building Retrieval Labels

The queries used to evaluate the app in Milestone 2 were exported to `retrieval_queries.json` and reused as candidate queries in our quantitative evaluation. `build_retrieval_labels.py` loads the evaluation queries from the JSON file, runs BM25, semantic, and hybrid retrieval for each query, and pools the returned candidates into a single CSV for review. "Pooling" here means taking the union of products returned by the different retrievers for the same query. Each retriever returns its top 10 products. For each `parent_asin` that appears in any of the result lists, the script creates a row if that product has not been seen yet for the query, or updates the existing row if it has. The row tracks which retrievers returned the product, stores each retriever's rank and score in separate fields such as `bm25_rank`, `semantic_rank`, `bm25_score`, and `semantic_score`, and records `best_rank` as the highest placement the product achieved across the retrievers, meaning the smallest rank number it received.

The output file, `retrieval_labels.csv`, is intended for human validation of relevance. To streamline this process, given the large number of results to be evaluated, ChatGPT-5 was first used to provide an initial screen, filling in the `relevance` column to categorize relevance to the query (`0` for not relevant, `1` for relevant). However, this process was only semi-supervised, as a spot check was manually performed using human judgment on the `relevance` column to assess quality and consistency. Relevance decisions were based on what a user would reasonably expect to retrieve for a query, not on whether a product was an exact lexical match. Some queries are broad, so relevance judgments were applied somewhat more leniently in those cases.

### Evaluation Retreival

Codex was used to brainstorm and draft `evaluate_retrieval.py`. This script evaluates the retrieval component of the system using labeled relevance judgments from `retrieval_labels.csv`. It does not compare BM25, semantic, and hybrid systems by raw scores, since those scores are not necessarily on the same scale. Instead, it evaluates the ranked lists returned by each retriever against the human relevance labels. For each labeled query, the script retrieves the top-k (5) results from BM25, semantic, and hybrid retrieval, then computes precision@k, recall@k, and reciprocal rank.

Mean precision@k and mean recall@k summarize each retriever's performance across the full query set at the chosen cutoff. MRR summarizes, on average, how early the first relevant item appears in the ranking. The output is written as a JSON file containing both summary metrics and per-query details, allowing results to be inspected directly. In the terminal, the summary across queries is also presented.

| Retriever | Queries | Mean Precision@5 | Mean Recall@5 | MRR |
|-----------|---------|------------------|---------------|-----|
| BM25 | 10 | 0.5400 | 0.3324 | 0.7250 |
| Semantic | 10 | 0.6800 | 0.3892 | 0.8500 |
| Hybrid | 10 | 0.5800 | 0.3594 | 0.8500 |

These results suggest that the semantic retriever performs best overall, compared with BM25 and even Hybrid. It achieved the highest mean precision of 0.68, meaning that about 3.4 of the top 5 retrieved products were relevant. It also achieved the highest mean recall of 0.3892, recovering a larger share of the judged relevant products within the top 5 than the other retrieval methods. For MRR, its score was tied with Hybrid search at 0.85, indicating that relevant items usually appear early in the ranking, often in the first position.

It is surprising that Semantic outperformed Hybrid, given clear weaknesses in both unimodal approaches. For example, for "Washingmachine," which contains a typo, semantic retrieval performs better than BM25, whereas for the exact serial number of a product, BM25 performs better. This may depend on the choice of search queries, as a few were deliberately designed to target semantic interpretation, for which exact matching would perform poorly. Also, for some items such as "Stainless Steel Coffee Maker," there were many matching keywords in product titles, but correctly identifying the best result likely still required semantic interpretation.

Overall, the quantitative assessment shows modest performance

An improvement to our analysis would be to add a filter to our searches so that either the top 10 results are returned or only results that are at least 80% of the top result's score are returned. Through the quantitative analysis, it became apparent that `Magic Bullet` has only one truly relevant product, yet nine other products are returned even though their scores are much worse. This may add noise to the interpretation of the quantitative measurements across retrieval modes.

## Step 3: Improve Documentation and Code Quality

### Documentation Update

Summary of `README` improvements:

- Updated the mermaid file structure based on the additional feature
- Updated the makefile to include controls for the new feature
- Added document in the custom Quantitative Evaluation feature

### Code Quality Changes

- Summary of cleanups

## Step 4: Cloud Deployment Plan

(See Step 4 above for required subsections)
