"""
PII Detection Solution for ServiceNow Tickets
Detects: Email, Phone, Danish CPR, Bank Details (IBAN)
Uses Microsoft Presidio Analyzer
"""

from typing import Dict, List

import pandas as pd
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider


class DanishPIIDetector:
    """
    Comprehensive PII detector for Danish data using Presidio
    """

    def __init__(self):
        """Initialize the analyzer with custom Danish recognizers"""
        # Create NLP engine configuration
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }

        # Create NLP engine
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # Initialize analyzer
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        # Add custom Danish recognizers
        self._add_danish_cpr_recognizer()
        self._add_danish_phone_recognizer()
        self._add_iban_recognizer()

    def _add_danish_cpr_recognizer(self):
        """
        Add Danish CPR (personnummer) recognizer
        Format: DDMMYY-XXXX or DDMMYYXXXX
        """
        # Pattern for Danish CPR: 6 digits (DDMMYY) + optional hyphen + 4 digits
        cpr_pattern = Pattern(
            name="danish_cpr_pattern", regex=r"\b\d{6}[-\s]?\d{4}\b", score=0.85
        )

        cpr_recognizer = PatternRecognizer(
            supported_entity="DK_CPR",
            patterns=[cpr_pattern],
            context=[
                "cpr",
                "personnummer",
                "social security",
                "ssn",
                "personal number",
            ],
        )

        self.analyzer.registry.add_recognizer(cpr_recognizer)

    def _add_danish_phone_recognizer(self):
        """
        Add Danish phone number recognizer
        Formats: +45 12345678, 0045 12345678, 12345678, +45-12-34-56-78, etc.
        """
        phone_patterns = [
            Pattern(
                name="danish_phone_international",
                regex=r"\+45[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}",
                score=0.9,
            ),
            Pattern(
                name="danish_phone_double_zero",
                regex=r"0045[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}[\s-]?\d{2}",
                score=0.9,
            ),
            Pattern(
                name="danish_phone_8digit",
                regex=r"\b\d{8}\b",
                score=0.6,  # Lower score as 8 digits could be other things
            ),
            Pattern(
                name="danish_phone_formatted",
                regex=r"\b\d{2}[\s-]\d{2}[\s-]\d{2}[\s-]\d{2}\b",
                score=0.75,
            ),
        ]

        phone_recognizer = PatternRecognizer(
            supported_entity="DK_PHONE_NUMBER",
            patterns=phone_patterns,
            context=["phone", "tel", "mobile", "mobil", "telefon", "tlf", "call"],
        )

        self.analyzer.registry.add_recognizer(phone_recognizer)

    def _add_iban_recognizer(self):
        """
        Add IBAN recognizer for bank account numbers
        Danish IBAN: DK + 2 check digits + 14 digits
        """
        iban_pattern = Pattern(
            name="iban_pattern",
            regex=r"\b[A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{4}[\s]?[\dA-Z]{0,4}\b",
            score=0.9,
        )

        # Also detect Danish registration number + account number format
        dk_account_pattern = Pattern(
            name="danish_account_pattern",
            regex=r"\b\d{4}[-\s]?\d{10}\b",  # Reg number (4 digits) + account (10 digits)
            score=0.75,
        )

        iban_recognizer = PatternRecognizer(
            supported_entity="BANK_ACCOUNT",
            patterns=[iban_pattern, dk_account_pattern],
            context=["iban", "account", "bank", "konto", "kontonummer", "reg"],
        )

        self.analyzer.registry.add_recognizer(iban_recognizer)

    def analyze_text(self, text: str, language: str = "en") -> List[Dict]:
        """
        Analyze text for PII entities

        Args:
            text: Text to analyze
            language: Language code (default: "en")

        Returns:
            List of dictionaries with entity information
        """
        if pd.isna(text) or text == "":
            return []

        # Run analysis
        results = self.analyzer.analyze(
            text=str(text),
            language=language,
            entities=[
                "EMAIL_ADDRESS",  # Built-in
                "PHONE_NUMBER",  # Built-in
                "DK_CPR",  # Custom
                "DK_PHONE_NUMBER",  # Custom
                "BANK_ACCOUNT",  # Custom
                "IBAN_CODE",  # Built-in
            ],
        )

        # Convert to dictionary format
        findings = []
        for result in results:
            findings.append(
                {
                    "entity_type": result.entity_type,
                    "text": text[result.start : result.end],
                    "start": result.start,
                    "end": result.end,
                    "score": result.score,
                }
            )

        return findings

    def scan_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "description",
        id_column: str = "number",
    ) -> pd.DataFrame:
        """
        Scan entire dataframe for PII

        Args:
            df: DataFrame to scan
            text_column: Column containing text to analyze
            id_column: Column containing ticket ID

        Returns:
            DataFrame with columns: ticket_id, description_preview, entity_type,
                                   detected_value, confidence_score
        """
        results = []

        for idx, row in df.iterrows():
            ticket_id = row[id_column]
            text = row[text_column]

            # Analyze the text
            findings = self.analyze_text(text)

            # Store each finding
            for finding in findings:
                results.append(
                    {
                        "ticket_id": ticket_id,
                        "description_preview": (
                            str(text)[:100] + "..."
                            if len(str(text)) > 100
                            else str(text)
                        ),
                        "entity_type": finding["entity_type"],
                        "detected_value": finding["text"],
                        "confidence_score": round(finding["score"], 3),
                        "position": f"{finding['start']}-{finding['end']}",
                    }
                )

        # Create results dataframe
        results_df = pd.DataFrame(results)

        # Sort by ticket_id and entity_type
        if not results_df.empty:
            results_df = results_df.sort_values(["ticket_id", "entity_type"])

        return results_df

    def get_summary_stats(self, results_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of PII findings

        Args:
            results_df: Results dataframe from scan_dataframe()

        Returns:
            Dictionary with summary statistics
        """
        if results_df.empty:
            return {
                "total_findings": 0,
                "unique_tickets_with_pii": 0,
                "findings_by_type": {},
            }

        stats = {
            "total_findings": len(results_df),
            "unique_tickets_with_pii": results_df["ticket_id"].nunique(),
            "findings_by_type": results_df["entity_type"].value_counts().to_dict(),
            "tickets_by_entity_type": results_df.groupby("entity_type")["ticket_id"]
            .nunique()
            .to_dict(),
        }

        return stats

    def mask_pii_in_text(self, text: str) -> str:
        """
        Mask PII entities in text with asterisks

        Args:
            text: Text to mask

        Returns:
            Masked text
        """
        if pd.isna(text) or text == "":
            return text

        findings = self.analyze_text(str(text))

        if not findings:
            return text

        # Sort findings by start position in reverse order
        # This allows us to replace from right to left, avoiding offset issues
        sorted_findings = sorted(findings, key=lambda x: x["start"], reverse=True)

        masked_text = str(text)

        for finding in sorted_findings:
            start = finding["start"]
            end = finding["end"]
            length = end - start
            masked_value = "*" * length

            masked_text = masked_text[:start] + masked_value + masked_text[end:]

        return masked_text

    def mask_pii_in_dataframe(
        self, df: pd.DataFrame, columns_to_mask: List[str]
    ) -> pd.DataFrame:
        """
        Mask PII in specified DataFrame columns

        Args:
            df: DataFrame to process
            columns_to_mask: List of column names to mask PII in

        Returns:
            DataFrame with PII masked in specified columns
        """
        # Create a copy to avoid modifying the original
        df_masked = df.copy()

        for column in columns_to_mask:
            if column not in df_masked.columns:
                print(f"Warning: Column '{column}' not found in DataFrame. Skipping.")
                continue

            # Apply masking to each cell in the column
            print(f"Masking PII in column: {column}")
            df_masked[column] = df_masked[column].apply(self.mask_pii_in_text)

        return df_masked
