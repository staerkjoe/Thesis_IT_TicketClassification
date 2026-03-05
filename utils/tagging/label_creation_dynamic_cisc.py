"""
Dynamic Auto Tagging Service for ServiceNow Tickets (CISC Enabled)
Uses RAG-based retrieval (TF-IDF) for dynamic few-shot learning,
combined with Confidence-Informed Self-Consistency (N samples, majority voting).
"""

import asyncio
import hashlib
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Union

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicAutoTagger:
    """
    Dynamic auto tagging with RAG-based example retrieval + CISC Aggregation
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        example_db_path: Optional[str] = None,
        use_async: bool = True,
    ) -> None:
        """
        Initialize the Dynamic AutoTagger
        """
        load_dotenv()

        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_key = os.getenv("API_KEY")
        self.api_version = "2025-01-01-preview"
        self.deployment_name = "gpt-5"

        self.use_async = use_async
        self.cache_enabled = True
        self._prediction_cache: Dict[str, Union[str, List[str]]] = {}

        # Load config
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent
                / "config"
                / "config_labels_dynamic.yaml"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.hierarchical_categories = [
            tuple(cat) for cat in config["hierarchical_categories"]
        ]
        self.scenarios = config["scenarios"]
        self.prompt_template = config["prompt_template"]

        # Retrieval settings
        self.num_examples = config["retrieval"]["num_examples"]
        self.min_similarity = config["retrieval"]["min_similarity"]
        self.retrieval_method = config["retrieval"]["method"]

        # Load example database
        if example_db_path is None:
            example_db_path = str(
                Path(__file__).parent.parent.parent / config["example_database"]["path"]
            )

        self.text_column = config["example_database"]["text_column"]
        self.label_column = config["example_database"]["label_column"]

        logger.info(f"Loading example database from {example_db_path}")
        self.example_df = pd.read_csv(example_db_path)

        # FIXED LEGACY BUG: Drop NA instead of taking every 3rd row. Keeps 1-row-per-ticket intact.
        self.example_df = self.example_df.dropna(
            subset=[self.text_column, self.label_column]
        ).reset_index(drop=True)
        logger.info(f"Loaded {len(self.example_df)} labeled examples for RAG database.")

        # Initialize TF-IDF vectorizer
        self._initialize_retriever()

        # Initialize Azure OpenAI clients
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        if self.use_async:
            self.async_client = AsyncAzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.api_key,
                api_version=self.api_version,
            )

        # Cost tracking metrics (February 2026 GPT-5 Rates)
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.price_per_1M_input = 1.25
        self.price_per_1M_output = 10.00

    def _initialize_retriever(self) -> None:
        """Initialize TF-IDF vectorizer and fit on example database"""
        logger.info("Initializing TF-IDF retriever...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        self.example_vectors = self.vectorizer.fit_transform(
            self.example_df[self.text_column].fillna("")
        )
        logger.info("✓ TF-IDF retriever initialized")

    def retrieve_examples(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve most similar examples for a query"""
        query_vector = self.vectorizer.transform([query_text])
        similarities = cosine_similarity(query_vector, self.example_vectors)[0]
        top_indices = similarities.argsort()[-self.num_examples :][::-1]

        retrieved = []
        for idx in top_indices:
            if similarities[idx] >= self.min_similarity:
                retrieved.append(
                    {
                        "text": self.example_df.iloc[idx][self.text_column],
                        "label": self.example_df.iloc[idx][self.label_column],
                        "similarity": float(similarities[idx]),
                    }
                )
        return retrieved

    def _format_examples(self, examples: List[Dict[str, Any]]) -> str:
        """Format retrieved examples to perfectly mirror the static run's 4-header format"""
        if not examples:
            return "No similar examples found."

        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(
                f"Description: {ex['text']}\n"
                f"Tag: {ex['label']}\n"
                f"Reasoning: Based on historical data, this matches previous tickets.\n"
                f"Reasoning_Confidence: 1.00\n"  # Synchronized with static
                f"Label_Confidence: 1.00\n"  # Synchronized with static
            )
        return "\n".join(formatted)

    def _get_cache_key(
        self, description: str, examples_text: str, n: int, temperature: float
    ) -> str:
        """Generate cache key including decoding parameters"""
        combined = f"{description}::{examples_text}::{n}::{temperature}"
        return hashlib.md5(combined.encode()).hexdigest()

    def test_connection(self) -> bool:
        """Test Azure OpenAI connection"""
        try:
            self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
            )
            logger.info("✓ Connection successful!")
            return True
        except Exception as e:
            logger.error(f"✗ Connection failed: {e}")
            return False

    # ------------------------------------------------------------------------
    # ASYNC LLM CALLS
    # ------------------------------------------------------------------------

    async def predict_samples_async(
        self,
        description: str,
        examples_text: str,
        n: int = 10,
        temperature: float = 1.0,
        max_retries: int = 3,
    ) -> List[str]:
        """Async: Fetch N samples using dynamically generated examples"""

        if self.cache_enabled:
            cache_key = self._get_cache_key(description, examples_text, n, temperature)
            if cache_key in self._prediction_cache:
                cached_val = self._prediction_cache[cache_key]
                if isinstance(cached_val, list):
                    return cached_val
                return [str(cached_val)]

        prompt = self.prompt_template.format(
            hierarchical_categories=self.hierarchical_categories,
            scenarios=self.scenarios,
            examples_text=examples_text,
            description=description,
        )

        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    n=n,
                    timeout=45.0 if n > 1 else 30.0,
                )

                results = [
                    choice.message.content.strip()
                    for choice in response.choices
                    if choice.message.content
                ]

                if self.cache_enabled:
                    self._prediction_cache[cache_key] = results

                return results

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(f"All retries failed for {description[:30]}...")
                    return ["UNKNOWN"] * n
        return ["UNKNOWN"] * n

    # ------------------------------------------------------------------------
    # CISC PARSING AND AGGREGATION (MIRRORED FROM STATIC)
    # ------------------------------------------------------------------------

    def parse_cisc_single_response(self, text: str) -> Dict[str, Any]:
        """
        Unified Parser: Robust against squished text.
        Correctly handles: (Category, Sub), Scenario Reasoning: ... Confidence: ...
        """
        result = {
            "label": "UNKNOWN",
            "reasoning": text,
            "confidence_score": 0.0,
            "raw_text": text,
        }

        # 1. Extract Tag/Label (Robust against missing newlines)
        # Pattern: (Something), 3 Something. Stops before 'Reasoning'
        label_match = re.search(
            r"(\([0-9][^)]+\),\s*3\s+.*?)(?=\s*Reasoning|\s*Confidence|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if label_match:
            result["label"] = label_match.group(1).strip()

        # 2. Extract Confidence (Looks for the final score)
        # Pattern: Finds any number following 'Confidence' or 'score'
        conf_matches = re.findall(
            r"(?:confidence|score)[:\s]*([0-9]*\.?[0-9]+)", text, re.IGNORECASE
        )
        if conf_matches:
            score = float(conf_matches[-1])
            result["confidence_score"] = score if score <= 1.0 else score / 100

        # 3. Extract Reasoning (Grabs text between 'Reasoning:' and 'Confidence:')
        reasoning_match = re.search(
            r"Reasoning[:\s]*(.*?)(?=Confidence|Reasoning_Confidence|Label_Confidence|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if reasoning_match:
            clean_reasoning = reasoning_match.group(1).strip()
            if len(clean_reasoning) > 10:
                result["reasoning"] = clean_reasoning

        return result

    def aggregate_cisc_results(self, raw_samples: List[str]) -> Dict[str, Any]:
        """
        Performs Confidence-Informed Self-Consistency Aggregation.
        """
        parsed_samples = [self.parse_cisc_single_response(s) for s in raw_samples]

        valid_samples = [s for s in parsed_samples if s["label"] != "UNKNOWN"]
        if not valid_samples:
            return {
                "label": "UNKNOWN",
                "reasoning": "All samples failed parsing",
                "scientific_confidence": 0.0,
            }

        # Typed variables to fix Mypy errors
        vote_scores: DefaultDict[str, float] = defaultdict(float)
        label_counts: Counter[str] = Counter()
        label_confs: DefaultDict[str, List[float]] = defaultdict(list)
        label_samples: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

        for sample in valid_samples:
            lbl = str(sample["label"])
            conf = float(sample["confidence_score"])

            vote_scores[lbl] += conf
            label_counts[lbl] += 1
            label_confs[lbl].append(conf)
            label_samples[lbl].append(sample)

        winning_label = max(vote_scores, key=lambda k: vote_scores[k])

        n_total = len(raw_samples)
        consistency = label_counts[winning_label] / n_total

        winner_confs = label_confs[winning_label]
        mean_conf = sum(winner_confs) / len(winner_confs) if winner_confs else 0.0

        scientific_confidence = (0.7 * consistency) + (0.3 * mean_conf)

        best_sample = max(
            label_samples[winning_label], key=lambda x: x["confidence_score"]
        )

        return {
            "label": winning_label,
            "reasoning": best_sample["reasoning"],
            "scientific_confidence": round(scientific_confidence, 4),
            "reasoning_confidence": best_sample["confidence_score"],
            "consistency_score": round(consistency, 4),
            "vote_count": label_counts[winning_label],
            "total_samples": n_total,
        }

    # ------------------------------------------------------------------------
    # MAIN BATCH PROCESSING
    # ------------------------------------------------------------------------

    async def predict_and_create_csv_async(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        n_samples: int = 10,
        semaphore_limit: int = 10,
    ) -> pd.DataFrame:
        """Async batch processing integrating RAG and CISC."""
        results: List[Dict[str, Any]] = []
        semaphore = asyncio.Semaphore(semaphore_limit)

        async def process_row(row: pd.Series) -> Dict[str, Any]:
            async with semaphore:
                number = row[number_column]
                text = row[text_column]

                # 1. RAG Step: Retrieve examples & format
                retrieved_examples = self.retrieve_examples(text)
                examples_text = self._format_examples(retrieved_examples)

                # 2. Call LLM Step: Request N Samples
                raw_samples = await self.predict_samples_async(
                    description=text,
                    examples_text=examples_text,
                    n=n_samples,
                    temperature=1.0,
                )

                # 3. Aggregate Step: CISC Logic
                agg_result = self.aggregate_cisc_results(raw_samples)

                # 4. Format Output Row
                result_row = {
                    "number": number,
                    "text": text,
                    "label": agg_result["label"],
                    "reasoning": agg_result["reasoning"],
                    "scientific_confidence": agg_result.get(
                        "scientific_confidence", 0.0
                    ),
                    "reasoning_confidence": agg_result.get("reasoning_confidence", 0.0),
                    "consistency": agg_result.get("consistency_score", 0.0),
                }

                # Attach the dynamically retrieved examples used for this prediction
                for i, ex in enumerate(retrieved_examples, 1):
                    result_row[f"shot{i}"] = f"{ex['text']} → {ex['label']}"

                # Fill remaining empty shots up to self.num_examples
                for i in range(len(retrieved_examples) + 1, self.num_examples + 1):
                    result_row[f"shot{i}"] = ""

                return result_row

        tasks = [process_row(row) for _, row in df.iterrows()]

        desc = f"Processing (Dynamic CISC, N={n_samples})"
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            row_data = await coro
            results.append(row_data)

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")

        return result_df

    def predict_and_create_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        n_samples: int = 10,
        semaphore_limit: int = 10,
    ) -> pd.DataFrame:
        """Main entry point wrapper for async execution"""
        return asyncio.run(
            self.predict_and_create_csv_async(
                df, output_path, number_column, text_column, n_samples, semaphore_limit
            )
        )

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics for notebook analysis"""
        return {
            "num_examples_in_db": len(self.example_df),
            "retrieval_method": self.retrieval_method,
            "num_examples_per_query": self.num_examples,
            "min_similarity_threshold": self.min_similarity,
        }

    def clear_cache(self) -> None:
        self._prediction_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._prediction_cache),
            "cache_enabled": self.cache_enabled,
        }
