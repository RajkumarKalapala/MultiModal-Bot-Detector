Multimodal Fake Profile Detection

Live Interactive Demo: https://huggingface.co/spaces/RajkumarSpace/MultiModal-Bot-Detector

This repository contains the architecture and implementation of a multimodal neural network designed to detect sophisticated bot accounts and fake profiles across social media platforms.
The Engineering Problem

Detecting fake profiles is a complex, adversarial challenge. Modern malicious actors no longer rely on simple scripts; they utilize coordinated botnets and LLM-generated content that easily bypass standard text-matching filters.

Relying on a single modality (e.g., NLP on tweets alone) results in high failure rates. A sophisticated bot can generate perfectly coherent, human-like text but will often exhibit anomalous numerical metadata (unnatural following/follower ratios) or highly irregular network graph structures. To build a robust, production-ready detection system, we must process the entire ecosystem of a profile simultaneously and look for contradictions across different data streams.
The Architecture

To capture this full spectrum of behavior, our system utilizes a late-fusion multimodal architecture. It routes different data types through specialized, independent neural networks before converging them into a final decision matrix.

    Text Processing Network: Ingests user posts and bio descriptions to extract semantic meaning, sentiment, and linguistic inconsistencies.

    Metadata/Numerical Network: Processes raw account statistics (follower counts, account age, posting frequency, variance in activity) to identify unnatural spikes or automated ratios.

    Feature Fusion Layer: The high-dimensional feature embeddings and hidden states from the independent networks are concatenated. This combined vector is then passed through dense classification layers to output a final probability score.

By fusing these modalities, the system cross-references behavior: an account attempting to mask its nature by posting realistic text will still be flagged by anomalies in its underlying metadata.
Dataset: TwiBot-20

This model was trained and evaluated using the TwiBot-20 dataset.

TwiBot-20 is a large-scale, highly complex benchmark designed to represent the modern social media landscape. It goes beyond simple bots to include sophisticated profiles across diverse domains (politics, business, entertainment). Crucially, it provides a heavily imbalanced, real-world distribution of genuine users versus malicious actors.

The dataset captures semantic text, profile properties, and neighborhood topology, making it the ideal baseline for multimodal evaluation.

Dataset Source & Documentation: https://github.com/BunsenFeng/TwiBot-20
Performance Metrics

The following metrics represent our Phase 2 baseline results evaluated against the imbalanced TwiBot-20 benchmark:

    Accuracy: 69.0%

    Precision: 74.0%

    F1-Score: 69.8%

Note: In automated moderation environments, achieving a high Precision score (74.0%) is critical to minimizing false positives, ensuring that genuine users are not mistakenly banned or shadowbanned.
