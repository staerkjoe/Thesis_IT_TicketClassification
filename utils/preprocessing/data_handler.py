import re
from pathlib import Path
from typing import List, Optional, Union

import nltk
import pandas as pd
from nltk.corpus import stopwords


class DataHandler:
    """Class for handling data operations including loading, combining, and preprocessing."""

    def __init__(self):
        pass

    def load_and_combine_excels(self, folder_path: Union[str, Path]) -> pd.DataFrame:
        """
        Loop through a folder and combine/concatenate all Excel files.

        Args:
            folder_path: Path to the folder containing Excel files

        Returns:
            Combined pandas DataFrame from all Excel files
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise ValueError(f"Folder path does not exist: {folder_path}")

        # Find all Excel files in the folder
        excel_files = list(folder_path.glob("*.xlsx")) + list(folder_path.glob("*.xls"))

        if not excel_files:
            raise ValueError(f"No Excel files found in {folder_path}")

        print(f"Found {len(excel_files)} Excel file(s) to combine")

        # Load each Excel file into a DataFrame and store in a list
        dataframes = []
        for excel_file in excel_files:
            print(f"Loading {excel_file.name}...")
            df = pd.read_excel(excel_file)
            dataframes.append(df)

        # Concatenate all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")

        return combined_df

    def preprocess_for_eda(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for EDA:
        - Rename columns: lowercase and replace spaces with underscores
        - Keep only specified columns: number, opened, title, description,
          assignment_group, priority, urgency

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()

        # Rename columns: lowercase and replace spaces with underscores
        df_processed.columns = df_processed.columns.str.lower().str.replace(" ", "_")

        # Define columns to keep
        columns_to_keep = [
            "number",
            "opened",
            "title",
            "description",
            "assignment_group",
            "priority",
            "urgency",
        ]

        # Check which columns are missing
        missing_cols = [
            col for col in columns_to_keep if col not in df_processed.columns
        ]
        if missing_cols:
            print(f"Warning: The following columns are missing: {missing_cols}")

        # Keep only the specified columns that exist
        existing_cols = [col for col in columns_to_keep if col in df_processed.columns]
        df_processed = df_processed[existing_cols]

        print(f"EDA preprocessing complete. Shape: {df_processed.shape}")
        print(f"Columns kept: {list(df_processed.columns)}")

        return df_processed

    def preprocess_for_modeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for modeling:
        - Rename columns: lowercase and replace spaces with underscores
        - Keep only specified columns: number, title, description, assignment_group

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying the original
        df_processed = df.copy()

        # Rename columns: lowercase and replace spaces with underscores
        df_processed.columns = df_processed.columns.str.lower().str.replace(" ", "_")

        # Define columns to keep
        columns_to_keep = ["number", "title", "description", "assignment_group"]

        # Check which columns are missing
        missing_cols = [
            col for col in columns_to_keep if col not in df_processed.columns
        ]
        if missing_cols:
            print(f"Warning: The following columns are missing: {missing_cols}")

        # Keep only the specified columns that exist
        existing_cols = [col for col in columns_to_keep if col in df_processed.columns]
        df_processed = df_processed[existing_cols]

        print(f"Modeling preprocessing complete. Shape: {df_processed.shape}")
        print(f"Columns kept: {list(df_processed.columns)}")

        return df_processed

    @staticmethod
    def clean_text_series(
        text_series: pd.Series, extra_stopwords: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Clean a pandas Series of text:
        - Convert to lowercase
        - Remove punctuation and special characters
        - Remove stopwords (English stopwords + any custom ones)
        - Remove extra whitespace

        Args:
            text_series: pandas Series containing text data
            extra_stopwords: Optional list of additional stopwords to remove

        Returns:
            Cleaned pandas Series
        """
        # Download stopwords if not already available
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        # Get English stopwords
        stop_words = set(stopwords.words("english"))

        # Add extra stopwords if provided
        if extra_stopwords:
            stop_words.update([word.lower() for word in extra_stopwords])

        def clean_text(text):
            """Clean individual text entry"""
            if pd.isna(text) or text == "":
                return ""

            # Convert to string and lowercase
            text = str(text).lower()

            # Remove URLs
            text = re.sub(r"http\S+|www\S+", "", text)

            # Remove email addresses
            text = re.sub(r"\S+@\S+", "", text)

            # Remove punctuation and special characters (keep spaces)
            text = re.sub(r"[^a-z\s]", " ", text)

            # Remove stopwords
            words = [
                word
                for word in text.split()
                if word not in stop_words and len(word) > 2
            ]

            # Join words back together
            cleaned = " ".join(words)

            return cleaned

        # Apply cleaning function to the series
        cleaned_series = text_series.apply(clean_text)

        print(
            f"Text cleaning complete. Non-empty entries: {(cleaned_series != '').sum()}"
        )

        return cleaned_series

    @staticmethod
    def extract_named_entities(
        text_series: pd.Series,
        model: str = "en_core_web_sm",
        exclude_numeric: bool = True,
        batch_size: int = 1000,
        n_process: int = 1,
    ) -> pd.Series:
        """
        Extract named entities from text using spaCy NER with batch processing for efficiency.

        Args:
            text_series: pandas Series containing text data
            model: spaCy model to use (default: 'en_core_web_sm')
            exclude_numeric: If True, exclude pure numeric entities
            batch_size: Number of texts to process in each batch (default: 1000)
            n_process: Number of parallel processes (-1 for all cores, 1 for single process)

        Returns:
            pandas Series with extracted entities as lists
        """
        try:
            import spacy
        except ImportError:
            raise ImportError("spaCy is required. Install with: pip install spacy")

        # Load spaCy model
        try:
            nlp = spacy.load(model)
        except OSError:
            print(f"Model '{model}' not found. Downloading...")
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", model])
            nlp = spacy.load(model)

        print(f"Extracting named entities using {model} with batch processing...")
        print(f"Batch size: {batch_size}, Processes: {n_process}")

        # Prepare texts - convert to strings and handle NaN
        texts = [
            str(text) if pd.notna(text) and text != "" else "" for text in text_series
        ]

        # Process texts in batches using nlp.pipe for efficiency
        all_entities: List[List[str]] = []

        # Use nlp.pipe for parallel batch processing
        for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            if doc.text == "":
                all_entities.append([])
            else:
                entities = [ent.text.lower() for ent in doc.ents]

                # Optionally exclude pure numeric entities
                if exclude_numeric:
                    entities = [
                        ent for ent in entities if not re.fullmatch(r"\d+", ent)
                    ]

                all_entities.append(entities)

        # Convert to pandas Series
        entities_series = pd.Series(all_entities, index=text_series.index)

        # Count total entities found
        total_entities = sum(len(ents) for ents in all_entities)
        non_empty = sum(1 for ents in all_entities if len(ents) > 0)
        print("NER extraction complete.")
        print(f"Total entities found: {total_entities}")
        print(f"Texts with entities: {non_empty}/{len(text_series)}")

        return entities_series

    @staticmethod
    def add_sentiment_analysis(
        df: pd.DataFrame, text_column: str, new_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add sentiment analysis scores to a DataFrame using TextBlob.

        Args:
            df: Input DataFrame
            text_column: Name of the column containing text to analyze
            new_column: Name for the new sentiment column (default: {text_column}_sentiment)

        Returns:
            DataFrame with added sentiment column
        """
        try:
            from textblob import TextBlob
        except ImportError:
            raise ImportError(
                "TextBlob is required. Install with: pip install textblob"
            )

        if new_column is None:
            new_column = f"{text_column}_sentiment"

        def get_sentiment(text):
            """Calculate sentiment polarity for text"""
            if pd.isna(text) or text == "":
                return 0.0
            return TextBlob(str(text)).sentiment.polarity

        print(f"Analyzing sentiment for column '{text_column}'...")
        df_copy = df.copy()
        df_copy[new_column] = df_copy[text_column].apply(get_sentiment)

        # Show basic statistics
        sentiment_stats = df_copy[new_column].describe()
        print("Sentiment analysis complete.")
        print(f"Mean sentiment: {sentiment_stats['mean']:.3f}")
        print(f"Range: [{sentiment_stats['min']:.3f}, {sentiment_stats['max']:.3f}]")

        return df_copy
