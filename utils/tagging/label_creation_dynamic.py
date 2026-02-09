"""
Dynamic Auto Tagging Service for ServiceNow Tickets
Uses RAG-based retrieval for dynamic few-shot learning
"""

import hashlib
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicAutoTagger:
    """
    Dynamic auto tagging with RAG-based example retrieval
    Uses TF-IDF for similarity-based example selection
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        example_db_path: Optional[str] = None,
        max_workers: int = 5,
    ) -> None:
        """
        Initialize the Dynamic AutoTagger

        Args:
            config_path: Path to config_labels_dynamic.yaml
            example_db_path: Path to labeled examples CSV (overrides config)
            max_workers: Number of parallel workers
        """
        # Load environment variables
        load_dotenv()

        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_key = os.getenv("API_KEY")
        self.api_version = "2025-01-01-preview"
        self.deployment_name = "gpt-5"

        # Performance settings
        self.max_workers = max_workers
        self.cache_enabled = True
        self._prediction_cache: Dict[str, str] = {}

        # Load config
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent / "config" / "config_labels_dynamic.yaml"
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
            example_db_path = config["example_database"]["path"]

        self.text_column = config["example_database"]["text_column"]
        self.label_column = config["example_database"]["label_column"]

        logger.info(f"Loading example database from {example_db_path}")
        self.example_df = pd.read_csv(example_db_path)

        # Filter to get only top1 predictions (assumes 3 rows per ticket)
        # Take every 3rd row starting from 0 (top1 prediction)
        self.example_df = self.example_df.iloc[::3].reset_index(drop=True)

        logger.info(f"Loaded {len(self.example_df)} labeled examples")

        # Initialize TF-IDF vectorizer
        self._initialize_retriever()

        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
        )

    def _initialize_retriever(self) -> None:
        """Initialize TF-IDF vectorizer and fit on example database"""
        logger.info("Initializing TF-IDF retriever...")

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )

        # Fit on example descriptions
        self.example_vectors = self.vectorizer.fit_transform(
            self.example_df[self.text_column].fillna("")
        )

        logger.info("✓ TF-IDF retriever initialized")

    def retrieve_examples(self, query_text: str) -> List[Dict[str, Any]]:
        """
        Retrieve most similar examples for a query

        Args:
            query_text: Input text to find similar examples for

        Returns:
            List of dicts with 'text' and 'label' keys
        """
        # Vectorize query
        query_vector = self.vectorizer.transform([query_text])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vector, self.example_vectors)[0]

        # Get top N indices
        top_indices = similarities.argsort()[-self.num_examples :][::-1]

        # Filter by minimum similarity
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
        """Format retrieved examples for prompt"""
        if not examples:
            return "No similar examples found."

        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Description: {ex['text']}\nTag: {ex['label']}\n")

        return "\n".join(formatted)

    def _get_cache_key(self, description: str, examples_text: str) -> str:
        """Generate cache key"""
        combined = f"{description}::{examples_text}"
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

    def predict_tags_with_llm(
        self, description: str, max_retries: int = 3
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Predict tags with dynamic few-shot examples

        Args:
            description: Ticket description
            max_retries: Number of retry attempts

        Returns:
            Tuple of (predicted_tag, retrieved_examples)
        """
        # Retrieve similar examples
        retrieved_examples = self.retrieve_examples(description)

        # Format examples for prompt
        examples_text = self._format_examples(retrieved_examples)

        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(description, examples_text)
            if cache_key in self._prediction_cache:
                logger.debug(f"Cache hit for description: {description[:50]}...")
                return self._prediction_cache[cache_key], retrieved_examples

        # Build prompt
        prompt = self.prompt_template.format(
            hierarchical_categories=self.hierarchical_categories,
            scenarios=self.scenarios,
            examples_text=examples_text,
            description=description,
        )

        # Call LLM with retries
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=30.0,
                )
                predicted_tag = response.choices[0].message.content.strip()

                # Cache result
                if self.cache_enabled:
                    self._prediction_cache[cache_key] = predicted_tag

                return predicted_tag, retrieved_examples

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"All retries failed for description: {description[:50]}..."
                    )
                    return "UNKNOWN", retrieved_examples
        return "UNKNOWN", retrieved_examples

    def parse_llm_response(self, llm_response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract top 3 predictions"""
        predictions: List[Dict[str, Any]] = []
        lines = llm_response.split("\n")

        current_pred: Dict[str, Any] = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for label patterns
            label_match = re.search(r"\([^)]+\)[^,]*,\s*3\s+[^\n]+", line)
            if label_match:
                if current_pred:
                    predictions.append(current_pred)
                    current_pred = {}
                current_pred["label"] = label_match.group(0).strip()

            # Look for confidence score
            conf_match = re.search(
                r"(?:confidence|score)[:\s]*([0-9]*\.?[0-9]+)", line, re.IGNORECASE
            )
            if conf_match and "confidence_score" not in current_pred:
                score = float(conf_match.group(1))
                current_pred["confidence_score"] = (
                    score if score <= 1.0 else score / 100
                )

            # Look for reasoning
            reason_keywords = [
                "reason",
                "explanation",
                "because",
                "indicates",
                "suggests",
            ]
            if (
                any(kw in line.lower() for kw in reason_keywords)
                and "reasoning" not in current_pred
            ):
                reasoning = re.sub(r"^[0-9\.\)\-\s]*", "", line)
                reasoning = re.sub(
                    r"(?:Reason|Reasoning|Explanation)[:\s]*",
                    "",
                    reasoning,
                    flags=re.IGNORECASE,
                )
                if reasoning and len(reasoning) > 10:
                    current_pred["reasoning"] = reasoning

        # Add last prediction
        if current_pred:
            predictions.append(current_pred)

        # Ensure 3 predictions
        while len(predictions) < 3:
            predictions.append(
                {
                    "label": "(1 Misc incidents, 2 Other), 3 Other",
                    "reasoning": "Unable to determine from description",
                    "confidence_score": 0.0,
                }
            )

        # Fill missing fields
        for i, pred in enumerate(predictions[:3]):
            if "label" not in pred:
                pred["label"] = "(1 Misc incidents, 2 Other), 3 Other"
            if "reasoning" not in pred:
                pred["reasoning"] = f"Classification {i+1}"
            if "confidence_score" not in pred:
                pred["confidence_score"] = 0.0

        return predictions[:3]

    def predict_and_create_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
    ) -> pd.DataFrame:
        """
        Predict tags with dynamic examples and create CSV

        Args:
            df: Input DataFrame
            output_path: Path to save output CSV
            number_column: Column with ticket IDs
            text_column: Column with text descriptions

        Returns:
            DataFrame with predictions and retrieved examples
        """

        def process_row(row: pd.Series) -> List[Dict[str, Any]]:
            """Process a single row"""
            number = row[number_column]
            text = row[text_column]

            # Get prediction with retrieved examples
            llm_response, retrieved_examples = self.predict_tags_with_llm(text)

            # Parse response
            predictions = self.parse_llm_response(llm_response)

            # Create results for this row (3 rows for 3 predictions)
            row_results: List[Dict[str, Any]] = []
            for pred in predictions:
                result = {
                    "number": number,
                    "text": text,
                    "label": pred["label"],
                    "reasoning": pred["reasoning"],
                    "confidence_score": pred["confidence_score"],
                }

                # Add retrieved examples (same for all 3 predictions)
                for i, ex in enumerate(retrieved_examples, 1):
                    result[f"shot{i}"] = f"{ex['text']} → {ex['label']}"

                # Fill remaining shots if fewer than 3 retrieved
                for i in range(len(retrieved_examples) + 1, 4):
                    result[f"shot{i}"] = ""

                row_results.append(result)

            return row_results

        results: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_row, row) for _, row in df.iterrows()]

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing tickets"
            ):
                try:
                    row_results = future.result()
                    results.extend(row_results)
                except Exception as e:
                    logger.error(f"Error processing row: {e}")

        # Create DataFrame
        result_df = pd.DataFrame(results)

        # Save to CSV
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")
        logger.info(f"✓ Total predictions: {len(result_df)}")
        logger.info(f"✓ Unique tickets: {result_df['number'].nunique()}")

        return result_df

    def clear_cache(self) -> None:
        """Clear prediction cache"""
        self._prediction_cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._prediction_cache),
            "cache_enabled": self.cache_enabled,
        }

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics"""
        return {
            "num_examples_in_db": len(self.example_df),
            "retrieval_method": self.retrieval_method,
            "num_examples_per_query": self.num_examples,
            "min_similarity_threshold": self.min_similarity,
        }
