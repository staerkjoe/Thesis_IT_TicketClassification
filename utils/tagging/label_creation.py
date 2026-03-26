"""
Auto Tagging Service for ServiceNow Tickets
Single label prediction with reasoning and confidence score.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoTaggerServiceNow:
    """
    Auto tagging service for ServiceNow tickets using Azure OpenAI.
    Returns a single label, reasoning, and confidence score per ticket.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        use_async: bool = True,
        max_workers: int = 5,
    ) -> None:
        load_dotenv()

        self.azure_endpoint = os.getenv("AZURE_ENDPOINT")
        self.api_key = os.getenv("API_KEY")
        self.api_version = "2025-01-01-preview"
        self.deployment_name = "gpt-5"

        self.use_async = use_async
        self.max_workers = max_workers
        self.cache_enabled = True
        self._prediction_cache: Dict[str, str] = {}

        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent.parent
                / "config"
                / "prompt_template_teacher.yaml"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.hierarchical_categories = [
            tuple(cat) for cat in config["hierarchical_categories"]
        ]
        self.scenarios = config["scenarios"]
        self.few_shot_examples = config["few_shot_examples"]
        self.prompt_template = config["prompt_template"]

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
        combined = f"{description}::{examples_text}"
        return hashlib.md5(combined.encode()).hexdigest()

    def test_connection(self) -> bool:
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

    def parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract a single label, reasoning, and confidence score.

        Returns:
            Dict with keys: label, reasoning, confidence_score
        """
        prediction: Dict[str, Any] = {
            "label": "(1 Misc incidents, 2 Other), 3 Other",
            "reasoning": "Unable to determine from description",
            "confidence_score": 0.0,
        }

        # Extract label — pattern: (Category, Subcategory), 3 Scenario
        label_match = re.search(r"\([^)]+\)[^,]*,\s*3\s+[^\n]+", llm_response)
        if label_match:
            prediction["label"] = label_match.group(0).strip()

        # Extract confidence score
        conf_match = re.search(
            r"(?:confidence|score|label_confidence)[:\s]*([0-9]*\.?[0-9]+)",
            llm_response,
            re.IGNORECASE,
        )
        if conf_match:
            score = float(conf_match.group(1))
            prediction["confidence_score"] = score if score <= 1.0 else score / 100

        # Extract reasoning
        reason_match = re.search(
            r"(?:Reasoning|Reason|Explanation)[:\s]*(.+?)(?:\n|Label_Confidence|$)",
            llm_response,
            re.IGNORECASE | re.DOTALL,
        )
        if reason_match:
            reasoning = reason_match.group(1).strip()
            if len(reasoning) > 10:
                prediction["reasoning"] = reasoning

        return prediction

    def predict_tags_with_llm(
        self,
        description: str,
        examples_text: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        if examples_text is None:
            examples_text = self.few_shot_examples

        if self.cache_enabled:
            cache_key = self._get_cache_key(description, examples_text)
            if cache_key in self._prediction_cache:
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

                if self.cache_enabled:
                    self._prediction_cache[cache_key] = predicted_tag

                return predicted_tag

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                else:
                    logger.error(f"All retries failed for: {description[:50]}...")
                    return "UNKNOWN"
        return "UNKNOWN"

    async def predict_tags_with_llm_async(
        self,
        description: str,
        examples_text: Optional[str] = None,
        max_retries: int = 3,
    ) -> str:
        if examples_text is None:
            examples_text = self.few_shot_examples

        if self.cache_enabled:
            cache_key = self._get_cache_key(description, examples_text)
            if cache_key in self._prediction_cache:
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
                    timeout=30.0,
                )
                predicted_tag = response.choices[0].message.content.strip()

                if self.cache_enabled:
                    self._prediction_cache[cache_key] = predicted_tag

                return predicted_tag

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(f"All retries failed for: {description[:50]}...")
                    return "UNKNOWN"
        return "UNKNOWN"

    def predict_and_create_csv_parallel(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        examples_text: Optional[str] = None,
    ) -> pd.DataFrame:
        if examples_text is None:
            examples_text = self.few_shot_examples

        def process_row(row: pd.Series) -> Dict[str, Any]:
            llm_response = self.predict_tags_with_llm(row[text_column], examples_text)
            prediction = self.parse_llm_response(llm_response)
            return {
                "number": row[number_column],
                "text": row[text_column],
                "label": prediction["label"],
                "reasoning": prediction["reasoning"],
                "confidence_score": prediction["confidence_score"],
            }

        results: List[Dict[str, Any]] = [None] * len(df)  # type: ignore

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(process_row, row): idx for idx, row in df.iterrows()
            }
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(future_to_idx),
                desc="Processing tickets",
            ):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error processing row {idx}: {e}")
                    results[idx] = {
                        "number": df.loc[idx, number_column],
                        "text": df.loc[idx, text_column],
                        "label": "(1 Misc incidents, 2 Other), 3 Other",
                        "reasoning": "Processing error",
                        "confidence_score": 0.0,
                    }

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        logger.info(f"✓ Results saved to {output_path}")
        return result_df

    async def predict_and_create_csv_async(
        self,
        df: pd.DataFrame,
        output_path: str,
        number_column: str = "number",
        text_column: str = "text",
        examples_text: Optional[str] = None,
        semaphore_limit: int = 10,
    ) -> pd.DataFrame:
        if examples_text is None:
            examples_text = self.few_shot_examples

        semaphore = asyncio.Semaphore(semaphore_limit)

        async def process_row(row: pd.Series) -> Dict[str, Any]:
            async with semaphore:
                llm_response = await self.predict_tags_with_llm_async(
                    row[text_column], examples_text
                )
                prediction = self.parse_llm_response(llm_response)
                return {
                    "number": row[number_column],
                    "text": row[text_column],
                    "label": prediction["label"],
                    "reasoning": prediction["reasoning"],
                    "confidence_score": prediction["confidence_score"],
                }

        tasks = [process_row(row) for _, row in df.iterrows()]
        results: List[Dict[str, Any]] = []

        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing tickets"
        ):
            results.append(await coro)

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
        examples_text: Optional[str] = None,
        use_async: Optional[bool] = None,
        semaphore_limit: int = 10,
    ) -> pd.DataFrame:
        """
        Main entry point. Runs async or parallel processing based on config.

        Output CSV columns: number, text, label, reasoning, confidence_score
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
        self._prediction_cache.clear()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self._prediction_cache),
            "cache_enabled": self.cache_enabled,
        }

    def set_few_shot_examples(self, examples_text: str) -> None:
        self.few_shot_examples = examples_text
        if self.cache_enabled:
            self.clear_cache()
