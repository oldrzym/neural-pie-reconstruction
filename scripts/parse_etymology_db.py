"""
Parser for etymology-db dataset to extract PIE proto-forms and their descendants.

Download the dataset first from:
https://1drv.ms/u/s!AtpEocFNRNBWhAe7co0JFvac-OfA?e=wnJe4r (CSV.gz)
or
https://1drv.ms/u/s!AtpEocFNRNBWhhP6w5D9XfdtPH9I?e=jWRwnI (Parquet)

Place it in: etymology-db/etymology.csv.gz or etymology.parquet
"""

import pandas as pd
import logging
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# Create logs directory FIRST
Path('logs').mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/parse_etymology_db.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EtymologyDBParser:
    """
    Extracts PIE → modern language cognate pairs from etymology-db.

    Schema:
    - term, lang: the word and its language
    - reltype: inherited_from, derived_from, root, etc.
    - related_term, related_lang: the etymological source
    """

    def __init__(self, data_path: str):
        """Load the etymology dataset."""
        self.data_path = Path(data_path)
        logger.info(f"Initializing EtymologyDBParser with {self.data_path}")

        if self.data_path.suffix == '.gz':
            logger.info("Loading gzipped CSV...")
            self.df = pd.read_csv(self.data_path, compression='gzip')
        elif self.data_path.suffix == '.parquet':
            logger.info("Loading Parquet...")
            self.df = pd.read_parquet(self.data_path)
        else:
            logger.info("Loading CSV...")
            self.df = pd.read_csv(self.data_path)

        logger.info(f"Loaded {len(self.df):,} etymological relationships")
        logger.info(f"Columns: {self.df.columns.tolist()}")
        logger.debug(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    def extract_pie_direct_descendants(self) -> pd.DataFrame:
        """
        Extract direct PIE → modern language relationships.

        Returns DataFrame with columns:
        - pie_form: the PIE proto-form
        - descendant: the modern word
        - language: the language of the descendant
        - reltype: the relationship type
        """
        logger.info("Extracting direct PIE → modern descendants...")

        # Filter for relationships where the SOURCE is PIE
        logger.debug("Filtering for related_lang == 'Proto-Indo-European'")
        pie_mask = self.df['related_lang'] == 'Proto-Indo-European'

        # Keep only inheritance/derivation relationships
        inheritance_types = ['inherited_from', 'derived_from']
        logger.debug(f"Filtering for reltype in {inheritance_types}")
        reltype_mask = self.df['reltype'].isin(inheritance_types)

        # Combine filters
        pie_descendants = self.df[pie_mask & reltype_mask].copy()
        logger.debug(f"After filtering: {len(pie_descendants):,} rows")

        # Rename columns for clarity
        pie_descendants = pie_descendants[[
            'related_term', 'term', 'lang', 'reltype'
        ]].rename(columns={
            'related_term': 'pie_form',
            'term': 'descendant',
            'lang': 'language'
        })

        logger.info(f"Found {len(pie_descendants):,} direct PIE → modern descendant pairs")
        logger.info(f"Unique PIE forms: {pie_descendants['pie_form'].nunique():,}")
        logger.info(f"Languages: {pie_descendants['language'].nunique()}")

        return pie_descendants

    def extract_pie_via_proto_branches(self) -> pd.DataFrame:
        """
        Extract PIE → Proto-X → modern language chains.

        This walks the inheritance chains:
        PIE *root → Proto-Germanic *form → English word
        """
        logger.info("Extracting PIE → Proto-X → modern chains...")

        # Step 1: Find PIE → Proto-branch relationships
        logger.debug("Step 1: Finding PIE → Proto-branch relationships")
        pie_to_proto = self.df[
            (self.df['related_lang'] == 'Proto-Indo-European') &
            (self.df['reltype'].isin(['inherited_from', 'derived_from']))
        ][['term', 'lang', 'related_term']].rename(columns={
            'term': 'proto_form',
            'lang': 'proto_lang',
            'related_term': 'pie_form'
        })
        logger.debug(f"Found {len(pie_to_proto):,} PIE → Proto-X relationships")

        # Step 2: Find Proto-branch → modern language relationships
        logger.debug("Step 2: Finding Proto-branch → modern relationships")
        proto_langs = pie_to_proto['proto_lang'].unique()
        logger.debug(f"Proto-languages: {len(proto_langs)}")
        proto_to_modern = self.df[
            (self.df['related_lang'].isin(proto_langs)) &
            (self.df['reltype'].isin(['inherited_from', 'derived_from']))
        ][['term', 'lang', 'related_term', 'related_lang']].rename(columns={
            'term': 'descendant',
            'lang': 'language',
            'related_term': 'proto_form',
            'related_lang': 'proto_lang'
        })
        logger.debug(f"Found {len(proto_to_modern):,} Proto-X → modern relationships")

        # Step 3: Join the chains
        logger.debug("Step 3: Joining chains")
        chains = proto_to_modern.merge(
            pie_to_proto,
            on=['proto_form', 'proto_lang'],
            how='inner'
        )[['pie_form', 'proto_lang', 'descendant', 'language']]

        logger.info(f"Found {len(chains):,} PIE → Proto-X → modern chains")
        logger.info(f"Unique PIE forms: {chains['pie_form'].nunique():,}")
        logger.info(f"Via proto-branches: {chains['proto_lang'].nunique()}")
        logger.info(f"To languages: {chains['language'].nunique()}")

        return chains

    def extract_pie_roots(self) -> pd.DataFrame:
        """
        Extract relationships where reltype='root' and related_lang='Proto-Indo-European'.

        This catches entries like:
        English "dog" has root "*ḱwṓ" in PIE
        """
        logger.info("Extracting PIE root annotations...")
        roots = self.df[
            (self.df['reltype'] == 'root') &
            (self.df['related_lang'] == 'Proto-Indo-European')
        ][['related_term', 'term', 'lang']].rename(columns={
            'related_term': 'pie_form',
            'term': 'descendant',
            'lang': 'language'
        })

        logger.info(f"Found {len(roots):,} words with PIE root annotations")
        logger.info(f"Unique PIE roots: {roots['pie_form'].nunique():,}")

        return roots

    def build_cognate_sets(self, descendants_df: pd.DataFrame) -> dict:
        """
        Group descendants by PIE form to create cognate sets.

        Returns dict: {pie_form: [(language, word), ...]}
        """
        logger.info("Building cognate sets...")
        cognate_sets = defaultdict(list)

        for _, row in tqdm(descendants_df.iterrows(), total=len(descendants_df), desc="Grouping cognates"):
            cognate_sets[row['pie_form']].append(
                (row['language'], row['descendant'])
            )

        logger.info(f"Built {len(cognate_sets):,} cognate sets")
        return dict(cognate_sets)

    def export_to_training_format(
        self,
        cognate_sets: dict,
        output_name: str,
        min_cognates: int = 2
    ):
        """
        Export cognate sets to training format (both TXT and CSV).

        TXT format: [lang1] word1 [lang2] word2 ... -> *PIE-form
        CSV format: input,output,meaning,num_cognates,cognate_set_id

        Args:
            cognate_sets: Output from build_cognate_sets()
            output_name: Base name (e.g., 'pie_direct')
            min_cognates: Minimum number of cognates per set
        """
        # Prepare output paths
        txt_path = Path(f"dataset/raw/etymology_db/{output_name}.txt")
        csv_path = Path(f"dataset/processed/etymology_db_{output_name}.csv")

        txt_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare CSV rows
        logger.info(f"Preparing data for export (min_cognates={min_cognates})...")
        csv_rows = []
        txt_lines = []
        cognate_set_id = 0

        for pie_form, cognates in tqdm(cognate_sets.items(), desc="Formatting output"):
            if len(cognates) < min_cognates:
                continue

            # Format: [lang1] word1 [lang2] word2 ...
            cognate_str = ' '.join(
                f"[{lang}] {word}" for lang, word in cognates
            )

            # TXT line
            txt_lines.append(f"{cognate_str} -> {pie_form}\n")

            # CSV row
            csv_rows.append({
                'input': cognate_str,
                'output': pie_form,
                'meaning': '',  # etymology-db doesn't have meanings
                'num_cognates': len(cognates),
                'cognate_set_id': cognate_set_id
            })
            cognate_set_id += 1

        # Write TXT
        logger.info(f"Writing TXT to {txt_path}...")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.writelines(txt_lines)
        logger.debug(f"Wrote {len(txt_lines):,} lines to TXT")

        # Write CSV
        logger.info(f"Writing CSV to {csv_path}...")
        df_out = pd.DataFrame(csv_rows)
        df_out.to_csv(csv_path, index=False, encoding='utf-8')
        logger.debug(f"CSV shape: {df_out.shape}")

        logger.info(f"✓ Exported {len(csv_rows):,} cognate sets")
        logger.info(f"  → TXT (raw): {txt_path}")
        logger.info(f"  → CSV (processed): {csv_path}")

        return df_out


def main():
    """Example usage."""
    logger.info("="*60)
    logger.info("Etymology-db Parser for PIE Reconstruction")
    logger.info("="*60)

    # Path to the downloaded dataset
    DATA_PATH = "etymology-db/etymology.csv.gz"
    # Or use Parquet: "etymology-db/etymology.parquet"

    # Initialize parser
    parser = EtymologyDBParser(DATA_PATH)

    # Strategy 1: Direct PIE → modern descendants
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 1: Direct PIE → modern language pairs")
    logger.info("="*60)
    direct = parser.extract_pie_direct_descendants()
    cognate_sets_1 = parser.build_cognate_sets(direct)
    df1 = parser.export_to_training_format(
        cognate_sets_1,
        "pie_direct",
        min_cognates=2
    )

    # Strategy 2: PIE → Proto-branch → modern chains
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 2: PIE → Proto-X → modern chains")
    logger.info("="*60)
    chains = parser.extract_pie_via_proto_branches()
    cognate_sets_2 = parser.build_cognate_sets(chains)
    df2 = parser.export_to_training_format(
        cognate_sets_2,
        "pie_via_proto",
        min_cognates=2
    )

    # Strategy 3: PIE root annotations
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 3: PIE root annotations")
    logger.info("="*60)
    roots = parser.extract_pie_roots()
    cognate_sets_3 = parser.build_cognate_sets(roots)
    df3 = parser.export_to_training_format(
        cognate_sets_3,
        "pie_roots",
        min_cognates=2
    )

    # Statistics
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Strategy 1 (Direct):      {len(df1):,} examples from {len(cognate_sets_1):,} PIE forms")
    logger.info(f"Strategy 2 (Via proto):   {len(df2):,} examples from {len(cognate_sets_2):,} PIE forms")
    logger.info(f"Strategy 3 (Root annot):  {len(df3):,} examples from {len(cognate_sets_3):,} PIE forms")
    logger.info(f"Total unique PIE forms:   {len(set(cognate_sets_1.keys()) | set(cognate_sets_2.keys()) | set(cognate_sets_3.keys())):,}")

    # Show example
    logger.info("\nExample cognate set (Strategy 2):")
    for pie_form, cognates in list(cognate_sets_2.items())[:1]:
        logger.info(f"PIE form: {pie_form}")
        for lang, word in cognates[:10]:
            logger.info(f"  {lang:30s} {word}")

    logger.info("\n" + "="*60)
    logger.info("✓ Done! Check logs/parse_etymology_db.log for details")
    logger.info("="*60)


if __name__ == "__main__":
    main()
