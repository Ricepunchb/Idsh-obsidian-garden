  
# JD
 **About Mistral**
At Mistral AI, we believe in the power of AI to simplify tasks, save time, and enhance learning and creativity. Our technology is designed to integrate seamlessly into daily working life.

We democratize AI through high-performance, optimized, open-source and cutting-edge models, products and solutions. Our comprehensive AI platform is designed to meet enterprise needs, whether on-premises or in cloud environments. Our offerings include le Chat, the AI assistant for life and work.

We are a dynamic, collaborative team passionate about AI and its potential to transform society.

Our diverse workforce thrives in competitive environments and is committed to driving innovation. Our teams are distributed between France, USA, UK, Germany and Singapore. We are creative, low-ego and team-spirited.

Join us to be part of a pioneering company shaping the future of AI. Together, we can make a meaningful impact. See more about our culture on [https://mistral.ai/careers](https://mistral.ai/careers).

**About The Job**

Mistral AI is seeking Applied Scientists and Research Engineers to drive innovative research and collaborate with clients on complex research projects.

You will develop SOTA models across different modalities such as text, image, and speech. By developing novel methods and research ideas you will apply these models across a diverse set of use cases and domains. Working cross-functionally with both external and internal science, engineering, and product teams you will deliver high-impact AI solutions that turn the needle.

**What you will do**

• Run pre-training, post-training and deploy state of the art models on clusters with thousands of GPUs. You don’t panic when you see OOM errors or when NCCL feels like not wanting to talk.

• Generate and curate data for pre-training and post-training, working on evaluations and making sure the model’s performance beats expectations.

• Develop the necessary tools and frameworks to facilitate data generation, model training, evaluation and deployment.

• Collaborate with cross-functional teams to tackle complex use cases using agents and RAG pipelines.

• Manage research projects and communications with client research teams.

**About you**

• You are fluent in English and Korean, and have excellent communication skills. You are at ease explaining complex technical concepts to both technical and non-technical audiences.

• You’re an expert with PyTorch or JAX.

• You’re not afraid of contributing to a big codebase and can find yourself around independently with little guidance.

• You write clean, readable, high-performance, fault-tolerant Python code.

• You don’t need roadmaps: you just do. You don’t need a manager: you just ship.

• Low-ego, collaborative and eager to learn.

• You have a track record of success through personal projects, professional projects or in academia.

**It would be great if you** 

• Hold a PhD / master in a relevant field (e.g., Mathematics, Physics, Machine Learning), but if you’re an exceptional candidate from a different background, you should apply.

• Can bring a variety of research experience (agents, multi-modality, robotics, diffusion, time-series).

• Have contributed to a large codebase used by many (open source or in the industry).

• Have a track record of publications in top academic journals or conferences.

• Love improving existing code by fixing typing issues, adding tests and improving CI pipelines.

# 질문들
### 1. Do you have experience in model pre-training?
My experience isn't in training large foundation models from scratch on raw corpora. Instead, I've worked one layer below that: redesigning how numbers are represented before a model ever sees them.

In my current research (the Adele Embeddings project), I use algebraic number theory — adele spaces — to build number representations that preserve additive and multiplicative structure. I started with a training-free module, validated on algebraic combinatorics benchmarks, and I'm now benchmarking a trainable version from scratch against methods like FoNE and BitTokens on transformer backbones (Llama, nanoGPT).

The next step is integrating this embedding into existing pre-trained LLMs without full retraining. That part isn't finished — it's not a plug-and-play swap yet — but the architecture is built with that path in mind. It's a narrower slice of pre-training than training a full model, but it required the same depth of understanding of how models encode information, just built from the ground up rather than at scale.
### 2. Do you have experience in model post-training?
Yes, I have extensive and diverse experience in model post-training, specifically focusing on Supervised Fine-Tuning (SFT), data-centric alignment, and hardware-efficient optimization. 

My practical technical stack includes executing robust SFT pipelines, implementing curriculum learning and knowledge distillation strategies, and leveraging advanced optimization frameworks such as QLoRA, FlashAttention, and next-generation optimizers like MUON to systematically control, align, and enhance model capabilities after the initial pre-training phase.
### 3. If yes to the above, describe the nature of your work, and the techniques used.
My post-training work spans three areas: data curation, memory-efficient fine-tuning, and architectural modification.

For the AI Grand Challenge, I built data pipelines using the GPT API to generate and refine domain-specific training data, then fine-tuned BigBird using weighted sampling to handle multi-answer distributions.

For the Samsung SCPC AI Challenge, I applied QLoRA and a curriculum learning strategy to Flan-T5 and KOSMOS-2, optimizing for on-device inference under a single-GPU constraint.

In my current research, I work directly with nanoGPT and Llama architectures — writing custom loss functions to enforce mathematical constraints on embeddings, and using the MUON optimizer for training stability. I also use vLLM across these projects to speed up inference during evaluation.
### 4. What is the largest scale (wrt. GPU nodes) that you have operated at?
The largest cluster I've worked with is my lab's local setup — three A100s (40GB) and five A6000s (48GB). It's a small cluster compared to an industrial data center, but I manage it directly: allocating GPU indices manually, isolating experiments with Docker, and running distributed jobs without dedicated infrastructure support. I'm used to squeezing performance out of constrained hardware rather than having automated tooling handle it for me.
### 5. Do you have an understanding of deployment: MLOps and Software Development, in an industry setting?
Most of my hands-on work has been on the modeling side rather than deployment infrastructure itself. That said, at SK Magic I worked closely with the engineers who took my models into production — they wrapped my Python-based analytics into the company's Java-based enterprise systems. That handoff process taught me what "production-ready" actually means in practice: clean interfaces, predictable failure modes, and code that someone else can integrate without needing me in the room.
### 6. Are you comfortable working in customer-facing roles and presenting to senior stakeholders?
Absolutely. As a Data Scientist Intern at SK Magic, a major part of my role involved scouting innovative startups and setting up technical alignments. I initiated and led numerous cross-functional meetings with C-level executives and senior stakeholders from global wellness and smart device enterprises, including Garmin Korea. This experience has made me highly comfortable and articulate when presenting abstract technical concepts, architectural choices, and technological synergies to both technical teams and non-technical executive audiences.
### 7. Do you have experience building transformer models from scratch?
Yes, particularly around integrating a custom embedding layer into transformer internals at the code level. Beyond high-level wrapper APIs, I've worked directly inside PyTorch transformer codebases — using nanoGPT and Llama-style architectures as a base — to wire a deterministic, Adele-space embedding layer into the model's forward pass, along with the custom loss functions needed to train it and a decoding head that reconstructs values from the learned representation.

I've also configured modern architectural components — RoPE, QK-normalization, the Muon optimizer — to build stable backbones for from-scratch benchmarking.

I haven't rewritten the attention mechanism itself, but I've had to understand it well enough to know exactly where a new representation needs to interface with it — and to debug issues, like normalization-induced distortion or gradient signal loss across channels, that only surface at that level.
### 8. Where do you see yourself in the next 2 years?
In two years, I'd like to be someone who can take an idea from a mathematical observation to a working improvement in a real model — end to end. Right now that's what I'm doing at a small scale: identifying a structural gap in how models represent information, then building and testing a fix for it.

At Mistral, I'd want to bring that same approach to problems the team is already working on — starting with post-training and architecture work, where I have the most direct experience with custom loss functions, optimizers like MUON, and inference tooling like vLLM. Over time, I'd like to take on harder, more open-ended problems, particularly around numerical and logical reasoning, where I think there's real room for architectural — not just scale-driven — improvements.