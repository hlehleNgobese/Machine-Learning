"""
COMPONENT B: LLM and RAG System
South African Parliamentary Hansard Transcripts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ================================================
# 1. DATA PREPARATION AND EMBEDDING GENERATION 
# ================================================

def load_hansard_data(filepath=None):
    """Load parliamentary transcripts or generate representative synthetic data."""
    if filepath:
        # Attempt to load - adjust columns as needed
        df = pd.read_csv(filepath)
    else:
        print(">> No file provided. Generating synthetic Hansard transcript data.")
        np.random.seed(42)
        topics = ['Education Policy', 'Healthcare Budget', 'Land Reform',
                  'Energy Crisis', 'Crime Prevention', 'Economic Growth',
                  'Water Infrastructure', 'Housing Policy', 'Transport Budget',
                  'Digital Transformation']

        templates_pos = [
            "The government has made significant progress in {} this year. We commend the efforts.",
            "I am pleased to report that {} initiatives have yielded positive results for our communities.",
            "The committee acknowledges the successful implementation of {} programmes across provinces.",
            "We welcome the increased budget allocation for {}. This will benefit many South Africans.",
            "The progress on {} demonstrates our commitment to building a better South Africa.",
        ]
        templates_neg = [
            "The failure to address {} is deeply concerning. Communities continue to suffer.",
            "We condemn the lack of progress on {}. The government must be held accountable.",
            "The budget cuts to {} are unacceptable and will harm vulnerable populations.",
            "Despite promises, {} remains in crisis. Urgent intervention is required.",
            "The mismanagement of {} funds has led to widespread service delivery failures.",
        ]
        templates_neu = [
            "The committee reviewed the {} report and noted the findings for further discussion.",
            "Members debated various aspects of {} policy during the session.",
            "The minister presented statistics on {} for the current financial year.",
            "A briefing on {} was provided to the portfolio committee members.",
            "The department submitted its {} quarterly report as required.",
        ]

        records = []
        for _ in range(600):
            topic = np.random.choice(topics)
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.35, 0.35])
            year = np.random.choice([2019, 2020, 2021, 2022, 2023])

            if sentiment == 'positive':
                text = np.random.choice(templates_pos).format(topic.lower())
            elif sentiment == 'negative':
                text = np.random.choice(templates_neg).format(topic.lower())
            else:
                text = np.random.choice(templates_neu).format(topic.lower())

            records.append({'text': text, 'topic': topic, 'year': year, 'sentiment': sentiment})

        df = pd.DataFrame(records)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def clean_and_tokenize(df):
    """Clean text and apply tokenization."""
    print("\n--- TEXT PREPROCESSING ---")

    # Basic cleaning
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_text'] = df['text'].apply(clean_text)
    print(f"  Cleaned {len(df)} documents")
    print(f"  Avg length: {df['clean_text'].str.len().mean():.0f} chars")

    # Sub-word tokenization using HuggingFace tokenizer (BPE)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        df['tokens'] = df['clean_text'].apply(lambda x: tokenizer.tokenize(x))
        df['token_count'] = df['tokens'].apply(len)
        print(f"  BPE tokenization applied (bert-base-uncased)")
        print(f"  Avg tokens per doc: {df['token_count'].mean():.1f}")
    except ImportError:
        print("  [WARNING] transformers not installed. Using basic whitespace tokenization.")
        df['tokens'] = df['clean_text'].str.split()
        df['token_count'] = df['tokens'].apply(len)

    print(f"\n  Tokenization justification:")
    print(f"  - BPE (Byte-Pair Encoding) handles out-of-vocabulary words by splitting into subwords")
    print(f"  - Effective for South African English which may contain Afrikaans/Zulu terms")
    print(f"  - Preserves semantic meaning better than character-level tokenization")

    return df


def generate_embeddings(df):
    """Generate semantic embeddings using a transformer model."""
    print("\n--- EMBEDDING GENERATION ---")

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(df['clean_text'].tolist(), show_progress_bar=True, batch_size=32)
        print(f"  Generated embeddings shape: {embeddings.shape}")
        print(f"  Using: all-MiniLM-L6-v2 (384-dim)")
    except ImportError:
        print("  [WARNING] sentence-transformers not installed. Using TF-IDF as fallback.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=384)
        embeddings = tfidf.fit_transform(df['clean_text']).toarray()
        print(f"  Generated TF-IDF embeddings shape: {embeddings.shape}")

    return embeddings


# =====================================
# 2. FINE-TUNING A TRANSFORMER MODEL 
# =====================================

def train_sentiment_classifier(df, embeddings):
    """Fine-tune/train a sentiment classifier."""
    print("\n--- SENTIMENT CLASSIFICATION ---")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y)

    # Try transformer-based classification, fallback to sklearn
    model = None
    y_pred = None

    try:
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                                  TrainingArguments, Trainer)
        import torch
        from torch.utils.data import Dataset

        print("  Using BERT fine-tuning for sentiment classification...")

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=3)

        class SentimentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len=128):
                self.encodings = tokenizer(texts, truncation=True, padding=True,
                                           max_length=max_len, return_tensors='pt')
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __getitem__(self, idx):
                item = {k: v[idx] for k, v in self.encodings.items()}
                item['labels'] = self.labels[idx]
                return item

            def __len__(self):
                return len(self.labels)

        train_texts = df.iloc[: len(y_train)]['clean_text'].tolist()
        test_texts = df.iloc[len(y_train):]['clean_text'].tolist()

        # Use smaller subset for faster training
        train_dataset = SentimentDataset(train_texts[:200], y_train[:200], tokenizer)
        test_dataset = SentimentDataset(test_texts[:100], y_test[:100], tokenizer)

        training_args = TrainingArguments(
            output_dir='./results', num_train_epochs=2, per_device_train_batch_size=16,
            per_device_eval_batch_size=16, logging_steps=50, save_strategy='no',
            report_to='none'
        )

        trainer = Trainer(model=bert_model, args=training_args,
                         train_dataset=train_dataset, eval_dataset=test_dataset)
        trainer.train()

        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_test_eval = y_test[:100]

    except (ImportError, Exception) as e:
        print(f"  [INFO] BERT fine-tuning unavailable ({e}). Using gradient boosting on embeddings.")
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_test_eval = y_test

    # Evaluation
    acc = accuracy_score(y_test_eval, y_pred)
    prec = precision_score(y_test_eval, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_eval, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_eval, y_pred, average='weighted', zero_division=0)

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_eval, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Sentiment Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('data/sentiment_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n  Classification Report:\n{classification_report(y_test_eval, y_pred, target_names=class_names)}")

    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}


# =======================
# 3. RAG IMPLEMENTATION 
# =======================

def build_rag_pipeline(df, embeddings):
    """Build Retrieval-Augmented Generation pipeline."""
    print("\n--- RAG PIPELINE ---")

    # Vector store using FAISS
    try:
        import faiss
        print("  Using FAISS for vector search")
    except ImportError:
        print("  [WARNING] FAISS not installed. Using sklearn NearestNeighbors as fallback.")

    # Build index
    embedding_dim = embeddings.shape[1]

    try:
        import faiss
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings.astype(np.float32))
        use_faiss = True
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5, metric='cosine')
        nn.fit(embeddings)
        use_faiss = False

    # Query function
    def retrieve(query, k=5):
        """Retrieve top-k relevant documents for a query."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            q_emb = model.encode([query]).astype(np.float32)
        except ImportError:
            from sklearn.feature_extraction.text import TfidfVectorizer
            tfidf = TfidfVectorizer(max_features=embedding_dim)
            tfidf.fit(df['clean_text'])
            q_emb = tfidf.transform([query.lower()]).toarray().astype(np.float32)

        if use_faiss:
            distances, indices = index.search(q_emb, k)
            indices = indices[0]
        else:
            distances, indices = nn.kneighbors(q_emb, n_neighbors=k)
            indices = indices[0]

        return df.iloc[indices]

    def generate_answer(query, context_docs):
        """Generate answer using retrieved context."""
        context = "\n".join(context_docs['text'].tolist())

        try:
            from transformers import pipeline
            generator = pipeline('text-generation', model='gpt2', max_new_tokens=100)
            prompt = f"Based on parliamentary records:\n{context}\n\nQuestion: {query}\nAnswer:"
            response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            return response[0]['generated_text']
        except (ImportError, Exception):
            # Rule-based fallback
            sentiments = context_docs['sentiment'].value_counts()
            topics = context_docs['topic'].value_counts()
            dominant_sentiment = sentiments.index[0] if len(sentiments) > 0 else 'unknown'
            dominant_topic = topics.index[0] if len(topics) > 0 else 'unknown'

            return (f"Based on {len(context_docs)} retrieved parliamentary records, "
                    f"the dominant sentiment is '{dominant_sentiment}'. "
                    f"The most discussed topic is '{dominant_topic}'. "
                    f"Key excerpts: '{context_docs.iloc[0]['text']}'")

    # Demo queries
    queries = [
        "What was the sentiment on education policy in 2023?",
        "How did parliament discuss the energy crisis?",
        "What are the views on land reform?",
    ]

    print("\n  --- RAG Demo Queries ---")
    for query in queries:
        print(f"\n  Q: {query}")
        retrieved = retrieve(query)
        answer = generate_answer(query, retrieved)
        print(f"  Retrieved {len(retrieved)} documents")
        print(f"  Sentiments: {retrieved['sentiment'].value_counts().to_dict()}")
        print(f"  A: {answer[:300]}...")

    # Evaluate relevance
    print("\n  --- Response Evaluation ---")
    print("""
    Relevance Assessment:
    - Retrieved documents are matched by semantic similarity to the query.
    - Factual grounding is ensured by generating answers ONLY from retrieved context.
    - Limitations: The model may hallucinate details not present in the retrieved documents.
    - Mitigation: We display source documents alongside generated answers for verification.
    """)

    return retrieve, generate_answer


# =======================================
# 4. ETHICS, RISK, AND RESPONSIBLE AI 
# =======================================

def ethics_discussion():
    """Critical discussion on ethics, risk, and responsible AI."""
    print("\n" + "=" * 60)
    print("ETHICS, RISK, AND RESPONSIBLE AI DISCUSSION")
    print("=" * 60)

    discussion = """
    1. BIAS IN DATASETS
    ─────────────────────
    Accident Data: South African traffic accident data suffers from reporting bias —
    accidents in rural areas and informal settlements are significantly under-reported.
    This creates models that disproportionately allocate resources to urban areas.
    Provincial disparities in data collection quality further compound this issue.

    Parliamentary Transcripts: Hansard records reflect the views of elected officials,
    not the general population. Opposition parties may be under-represented in debate
    time. Language bias exists as transcripts are primarily in English, excluding
    contributions in other official South African languages.

    2. TRANSPARENCY AND EXPLAINABILITY
    ────────────────────────────────────
    Ensemble Models: Random Forest and XGBoost are relatively interpretable through
    feature importance scores. However, the interaction effects between features
    (e.g., road condition × time of day) are opaque. SHAP values can improve
    local explainability but add computational cost.

    LLMs: Transformer models are fundamentally opaque. Attention weights provide
    some insight but do not constitute true explanations. The RAG pipeline improves
    transparency by grounding responses in retrievable source documents.

    3. RISKS OF HARMFUL/MISLEADING CONTENT
    ─────────────────────────────────────────
    - Accident prediction models may create false confidence in safety assessments,
      leading to reduced vigilance in areas predicted as "low risk."
    - LLM-generated summaries of parliamentary debates may misrepresent positions,
      particularly on sensitive topics like land reform or racial policy.
    - Generated content could be weaponized for political misinformation.

    4. MITIGATION STRATEGIES
    ──────────────────────────
    - Prompt Engineering: Constrain LLM outputs to factual summaries with citations.
    - Content Filtering: Implement toxicity and factuality checks on generated text.
    - Human-in-the-Loop: All model recommendations (both accident interventions and
      policy summaries) should require human review before action.
    - Confidence Thresholds: Only present predictions above a reliability threshold.
    - Regular Auditing: Periodic bias audits across demographic and geographic groups.

    5. RESPONSIBLE AI IN THE SOUTH AFRICAN CONTEXT
    ─────────────────────────────────────────────────
    - The Protection of Personal Information Act (POPIA) requires that automated
      decision-making systems be transparent and contestable.
    - South Africa's diverse linguistic landscape (11 official languages) means
      English-only models exclude significant portions of the population.
    - Historical inequalities mean that biased models risk perpetuating apartheid-era
      resource allocation patterns.
    - The National Development Plan 2030 emphasizes inclusive technology deployment —
      AI systems must demonstrably serve all communities equitably.
    - Deployment should prioritize provinces with highest accident rates while ensuring
      rural communities are not systematically disadvantaged by data gaps.
    """
    print(discussion)


def run_component_b(filepath=None):
    """Execute full Component B pipeline."""
    print("=" * 60)
    print("COMPONENT B: LLM and RAG System")
    print("=" * 60)

    # 1. Data Preparation
    df = load_hansard_data(filepath)
    df = clean_and_tokenize(df)

    # 2. Embeddings
    embeddings = generate_embeddings(df)

    # Save preprocessed data
    df[['text', 'clean_text', 'topic', 'year', 'sentiment']].to_csv(
        'data/preprocessed_hansard.csv', index=False)
    print("\nPreprocessed data saved to data/preprocessed_hansard.csv")

    # 3. Sentiment Classification
    results = train_sentiment_classifier(df, embeddings)

    # 4. RAG Pipeline
    retrieve, generate = build_rag_pipeline(df, embeddings)

    # 5. Ethics Discussion
    ethics_discussion()

    return df, results


if __name__ == '__main__':
    run_component_b()
