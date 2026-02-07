"""
Auto Tagging Service for ServiceNow Tickets - OPTIMIZED VERSION
Improvements:
- Async/await for concurrent API calls
- Thread pool executor for parallel processing
- Better error handling and retry logic
- Progress tracking with tqdm
- Caching to avoid redundant API calls
- Batch API calls where possible
"""

import asyncio
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
from openai import AsyncAzureOpenAI, AzureOpenAI
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoTaggerServiceNow:
    """
    Auto tagging service for ServiceNow tickets using Azure OpenAI
    Optimized for parallel processing and performance
    """

    def __init__(self, config_path=None, use_async=True, max_workers=5):
        """
        Initialize the AutoTagger with configuration

        Args:
            config_path: Path to config_labels.yaml file
            use_async: Whether to use async processing (recommended)
            max_workers: Maximum number of parallel workers
        """
        # Load environment variables
        load_dotenv()

        # Azure OpenAI Configuration
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_key = os.getenv("API_KEY")
        self.api_version = "2025-01-01-preview"
        self.deployment_name = "gpt-5"

        # Performance settings
        self.use_async = use_async
        self.max_workers = max_workers
        self.cache_enabled = True
        self._prediction_cache = {}

        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config_labels.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.hierarchical_categories = [
            tuple(cat) for cat in config["hierarchical_categories"]
        ]
        self.scenarios = config["scenarios"]
        self.few_shot_examples = config["few_shot_examples"]
        self.prompt_template = config["prompt_template"]

        # Initialize clients
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

    def _get_cache_key(self, description: str, examples_text: str) -> str:
        """Generate cache key for a description"""
        combined = f"{description}::{examples_text}"
        return hashlib.md5(combined.encode()).hexdigest()

    def test_connection(self):
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

    async def predict_tags_with_llm_async(
        self,
        description: str,
        examples_text: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Async version: Predict tags using few-shot learning with Azure OpenAI

        Args:
            description: Ticket description to classify
            examples_text: Few-shot examples (uses default if None)
            max_retries: Number of retry attempts

        Returns:
            Predicted tag string
        """
        if examples_text is None:
            examples_text = self.few_shot_examples

        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(description, examples_text)
            if cache_key in self._prediction_cache:
                logger.debug(f"Cache hit for description: {description[:50]}...")
                return self._prediction_cache[cache_key]

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
                    timeout=30.0,  # Add timeout
                )
                predicted_tag = response.choices[0].message.content.strip()

                # Cache result
                if self.cache_enabled:
                    self._prediction_cache[cache_key] = predicted_tag

                return predicted_tag

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"All retries failed for description: {description[:50]}..."
                    )
                    return "UNKNOWN"
        return "UNKNOWN"

    def predict_tags_with_llm(
        self,
        description: str,
        examples_text: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Synchronous version: Predict tags using few-shot learning

        Args:
            description: Ticket description to classify
            examples_text: Few-shot examples (uses default if None)
            max_retries: Number of retry attempts

        Returns:
            Predicted tag string
        """
        if examples_text is None:
            examples_text = self.few_shot_examples

        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(description, examples_text)
            if cache_key in self._prediction_cache:
                logger.debug(f"Cache hit for description: {description[:50]}...")
                return self._prediction_cache[cache_key]

        prompt = self.prompt_template.format(
            hierarchical_categories=self.hierarchical_categories,
            scenarios=self.scenarios,
            examples_text=examples_text,
            description=description,
        )

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

                return predicted_tag

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    logger.error(
                        f"All retries failed for description: {description[:50]}..."
                    )
                    return "UNKNOWN"
        return "UNKNOWN"

    def parse_llm_response(self, llm_response: str) -> List[Dict]:
        """
        Parse LLM response to extract top 3 predictions with labels, reasoning, and confidence scores

        Args:
            llm_response: Raw LLM response string

        Returns:
            List of dicts with keys: label, reasoning, confidence_score
        """
        predictions = []

        # Split response into sections
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
                    # current_pred: Dict[str, Any] = {}
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

        # Ensure we have 3 predictions
        while len(predictions) < 3:
            predictions.append(
                {
                    "label": "(1 Misc incidents, 2 Other), 3 Other",
                    "reasoning": "Unable to determine from description",
                    "confidence_score": 0.0,
                }
            )

        # Fill in missing fields
        for i, pred in enumerate(predictions[:3]):
            if "label" not in pred:
                pred["label"] = "(1 Misc incidents, 2 Other), 3 Other"
            if "reasoning" not in pred:
                pred["reasoning"] = f"Classification {i+1}"
            if "confidence_score" not in pred:
                pred["confidence_score"] = 0.0

        return predictions[:3]

    async def predict_tags_batch_async(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        examples_text: Optional[str] = None,
        semaphore_limit: int = 10,
    ) -> List[str]:
        """
        Async batch prediction with controlled concurrency

        Args:
            df: DataFrame with text column
            text_column: Name of the column containing text
            examples_text: Few-shot examples
            semaphore_limit: Maximum concurrent API calls

        Returns:
            List of predicted tags
        """
        if examples_text is None:
            examples_text = self.few_shot_examples

        semaphore = asyncio.Semaphore(semaphore_limit)

        async def predict_with_semaphore(text: str) -> str:
            async with semaphore:
                return await self.predict_tags_with_llm_async(text, examples_text)

        # Create tasks for all rows
        tasks = [predict_with_semaphore(row[text_column]) for _, row in df.iterrows()]

        # Execute with progress bar
        predictions: List[str] = []
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Predicting tags"
        ):
            result = await coro
            predictions.append(result)

        return predictions

    def predict_tags_batch_parallel(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        examples_text: Optional[str] = None,
    ) -> List[str]:
        """
        Parallel batch prediction using ThreadPoolExecutor

        Args:
            df: DataFrame with text column
            text_column: Name of the column containing text
            examples_text: Few-shot examples

        Returns:
            List of predicted tags
        """
        if examples_text is None:
            examples_text = self.few_shot_examples

        def predict_single(text: str) -> str:
            return self.predict_tags_with_llm(text, examples_text)

        predictions: List[Optional[str]] = [None] * len(df)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(predict_single, row[text_column]): idx
                for idx, row in df.iterrows()
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Predicting tags",
            ):
                idx = future_to_idx[future]
                try:
                    predictions[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    predictions[idx] = "UNKNOWN"

        # Convert to List[str] by filtering out None values (shouldn't happen but for type safety)
        return [p if p is not None else "UNKNOWN" for p in predictions]

    async def predict_and_create_csv_async(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        examples_text: Optional[str] = None,
        semaphore_limit: int = 10,
    ) -> pd.DataFrame:
        """
        Async version: Predict tags and create structured CSV

        Args:
            df: DataFrame with number and text columns
            output_path: Path to save the output CSV
            number_column: Column containing ticket number/ID
            text_column: Column containing text to classify
            examples_text: Few-shot examples
            semaphore_limit: Maximum concurrent API calls

        Returns:
            DataFrame with predictions
        """
        if examples_text is None:
            examples_text = self.few_shot_examples

        results = []
        semaphore = asyncio.Semaphore(semaphore_limit)

        async def process_row(row: pd.Series) -> List[Dict]:
            async with semaphore:
                number = row[number_column]
                text = row[text_column]

                # Get LLM prediction
                llm_response = await self.predict_tags_with_llm_async(
                    text, examples_text
                )

                # Parse response
                predictions = self.parse_llm_response(llm_response)

                # Create results for this row
                row_results: List[Dict] = []
                for pred in predictions:
                    row_results.append(
                        {
                            "number": number,
                            "text": text,
                            "label": pred["label"],
                            "reasoning": pred["reasoning"],
                            "confidence_score": pred["confidence_score"],
                        }
                    )
                return row_results

        # Create tasks
        tasks = [process_row(row) for _, row in df.iterrows()]

        # Execute with progress bar
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing tickets"
        ):
            row_results = await coro
            results.extend(row_results)

        # Create DataFrame
        result_df = pd.DataFrame(results)

        # Save to CSV
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")

        return result_df

    def predict_and_create_csv_parallel(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        examples_text: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parallel version: Predict tags and create structured CSV

        Args:
            df: DataFrame with number and text columns
            output_path: Path to save output CSV
            number_column: Column containing ticket number/ID
            text_column: Column containing text to classify
            examples_text: Few-shot examples

        Returns:
            DataFrame with predictions
        """
        if examples_text is None:
            examples_text = self.few_shot_examples

        def process_row(row: pd.Series) -> List[Dict]:
            number = row[number_column]
            text = row[text_column]

            # Get LLM prediction
            llm_response = self.predict_tags_with_llm(text, examples_text)

            # Parse response
            predictions = self.parse_llm_response(llm_response)

            # Create results for this row
            row_results: List[Dict] = []
            for pred in predictions:
                row_results.append(
                    {
                        "number": number,
                        "text": text,
                        "label": pred["label"],
                        "reasoning": pred["reasoning"],
                        "confidence_score": pred["confidence_score"],
                    }
                )
            return row_results

        results = []

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

        return result_df

    def predict_and_create_csv(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        examples_text: Optional[str] = None,
        use_async: Optional[bool] = None,
        semaphore_limit: int = 10,
    ) -> pd.DataFrame:
        """
        Main entry point: Automatically chooses async or parallel processing

        Args:
            df: DataFrame with number and text columns
            output_path: Path to save output CSV
            number_column: Column containing ticket number/ID
            text_column: Column containing text to classify
            examples_text: Few-shot examples
            use_async: Force async (True) or parallel (False), None = use default
            semaphore_limit: Max concurrent calls for async

        Returns:
            DataFrame with predictions
        """
        if use_async is None:
            use_async = self.use_async

        if use_async:
            logger.info("Using async processing...")
            return asyncio.run(
                self.predict_and_create_csv_async(
                    df,
                    output_path,
                    number_column,
                    text_column,
                    examples_text,
                    semaphore_limit,
                )
            )
        else:
            logger.info("Using parallel processing...")
            return self.predict_and_create_csv_parallel(
                df, output_path, number_column, text_column, examples_text
            )

    def clear_cache(self) -> None:
        """Clear prediction cache"""
        self._prediction_cache.clear()
        logger.info("Cache cleared")
        return None

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self._prediction_cache),
            "cache_enabled": self.cache_enabled,
        }

    def set_few_shot_examples(self, examples_text: str) -> None:
        """Set custom few-shot examples"""
        self.few_shot_examples = examples_text
        # Clear cache when examples change
        if self.cache_enabled:
            self.clear_cache()
        return None
