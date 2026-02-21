Multimodal Fake Profile Detection

This repository contains the code and architecture for a multimodal fake profile detection system, designed to identify sophisticated bot accounts by fusing multiple distinct data streams.
The Problem

Detecting fake profiles on modern social networks is highly complex because malicious actors have evolved. Early bots relied on simple, repetitive scripts, making them easy to catch. Today, coordinated botnets and LLM-generated content can perfectly mimic human typing patterns.

Relying on a single modality—such as text analysis alone—is no longer sufficient. A bot might post highly realistic text, but exhibit unnatural following/follower ratios, or use artificially generated profile images. To build a robust detection system, we must analyze the entire profile ecosystem simultaneously.
The Architecture

To capture the full scope of a profile's behavior, this system moves beyond single-stream analysis and implements a multimodal fusion approach. It utilizes three distinct neural networks, each specialized for a specific data type, before fusing their outputs into a single, unified decision.

    Text Processing Network: Extracts semantic meaning, sentiment, and linguistic patterns from the user's posts and bio descriptions.

    Metadata/Numerical Network: Processes account statistics like follower counts, account age, and posting frequency to identify unnatural spikes or ratios.

    Feature Fusion Layer: The hidden states and feature embeddings from all independent networks are concatenated and passed through dense classification layers.

This late-fusion technique ensures that a profile trying to mask its identity in one modality (e.g., posting normal text) will still be flagged by anomalies in another (e.g., network behavior).
The Dataset: TwiBot-20

The system is built and evaluated using the TwiBot-20 dataset.

About the Dataset:
TwiBot-20 is a comprehensive, large-scale Twitter bot detection benchmark designed to represent the current generation of the real-world Twittersphere. Unlike older datasets that focus on simple rule-based bots, TwiBot-20 covers diverse domains (politics, business, entertainment, and sports) and captures three crucial modalities of user information:

    Semantics: The textual content of a user's recent tweets.

    Properties: User profile metadata and numerical statistics.

    Neighborhood: The topological graph of follower and following relationships.

Because it includes both highly sophisticated bots and genuine users, it provides a highly complex, heavily imbalanced environment that accurately reflects the challenges of modern social media moderation.

🔗 Dataset Link: TwiBot-20 Official GitHub Repository
Performance Metrics

The following metrics represent our Phase 2 baseline results on the imbalanced TwiBot-20 benchmark:

    Accuracy: 69.0%

    Precision: 74.0%

    F1-Score: 69.8%

Achieving a high precision score is critical in this domain to minimize false positives and ensure genuine users are not incorrectly penalized.
