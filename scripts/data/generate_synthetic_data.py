"""
Synthetic Data Generation for ASD Prediction System

Generates realistic synthetic ASD screening data for development and testing.
Based on behavioral markers commonly used in early ASD screening tools
like M-CHAT-R/F adapted for the Zimbabwean context.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_STATE = 42


class ASDDataGenerator:
    """
    Generator for synthetic ASD screening data.

    Creates realistic data based on known ASD behavioral patterns
    while maintaining statistical properties suitable for ML training.
    """

    # Zimbabwe provinces for geographic distribution
    PROVINCES = [
        'Harare', 'Bulawayo', 'Manicaland', 'Mashonaland_Central',
        'Mashonaland_East', 'Mashonaland_West', 'Masvingo',
        'Matabeleland_North', 'Matabeleland_South', 'Midlands'
    ]

    # Urban vs rural classification
    SETTING_TYPES = ['urban', 'peri-urban', 'rural']

    def __init__(self, random_state: int = RANDOM_STATE):
        """
        Initialize the data generator.

        Args:
            random_state: Seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)

    def _generate_behavioral_features(
        self,
        n_samples: int,
        asd_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate behavioral screening features.

        ASD-positive cases have higher probability of showing
        atypical behavioral patterns.

        Args:
            n_samples: Number of samples
            asd_labels: Array of ASD diagnosis labels (0/1)

        Returns:
            DataFrame with behavioral features
        """
        # Probability of atypical response (1) for each feature
        # Format: (prob_if_no_asd, prob_if_asd)
        feature_probs = {
            'eye_contact': (0.15, 0.75),           # Reduced eye contact
            'response_to_name': (0.10, 0.65),      # Doesn't respond to name
            'pointing': (0.12, 0.70),              # Lack of pointing
            'social_smile': (0.08, 0.55),          # Reduced social smiling
            'repetitive_behaviors': (0.10, 0.72),  # Repetitive movements
            'joint_attention': (0.12, 0.68),       # Joint attention deficits
            'pretend_play': (0.15, 0.65),          # Limited pretend play
            'unusual_interests': (0.08, 0.60),     # Unusual sensory interests
            'hand_flapping': (0.05, 0.45),         # Hand flapping/stimming
            'toe_walking': (0.08, 0.35),           # Toe walking
            'lines_up_toys': (0.10, 0.55),         # Lines up objects
            'upset_by_change': (0.15, 0.62),       # Distress with routine changes
        }

        data = {}
        for feature, (prob_no_asd, prob_asd) in feature_probs.items():
            probs = np.where(asd_labels == 1, prob_asd, prob_no_asd)
            # Add some noise to make it more realistic
            probs = np.clip(probs + np.random.normal(0, 0.05, n_samples), 0.01, 0.99)
            data[feature] = np.random.binomial(1, probs)

        return pd.DataFrame(data)

    def _generate_communication_features(
        self,
        n_samples: int,
        asd_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate communication-related features.

        Args:
            n_samples: Number of samples
            asd_labels: Array of ASD diagnosis labels

        Returns:
            DataFrame with communication features
        """
        data = {}

        # Word count (vocabulary size) - continuous
        # Typical: 50-300 words at 18-36 months
        # ASD: often delayed, 0-100 words
        typical_words = np.random.normal(150, 50, n_samples)
        asd_words = np.random.normal(40, 30, n_samples)
        data['word_count'] = np.where(
            asd_labels == 1,
            np.clip(asd_words, 0, 200),
            np.clip(typical_words, 20, 350)
        ).astype(int)

        # Two-word phrases (0 = no, 1 = yes)
        data['two_word_phrases'] = np.where(
            asd_labels == 1,
            np.random.binomial(1, 0.25, n_samples),
            np.random.binomial(1, 0.85, n_samples)
        )

        # Echolalia (repeating words/phrases)
        data['echolalia'] = np.where(
            asd_labels == 1,
            np.random.binomial(1, 0.55, n_samples),
            np.random.binomial(1, 0.10, n_samples)
        )

        # Loss of language skills
        data['language_regression'] = np.where(
            asd_labels == 1,
            np.random.binomial(1, 0.30, n_samples),
            np.random.binomial(1, 0.02, n_samples)
        )

        return pd.DataFrame(data)

    def _generate_demographic_features(
        self,
        n_samples: int,
        asd_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate demographic features.

        Args:
            n_samples: Number of samples
            asd_labels: Array of ASD diagnosis labels

        Returns:
            DataFrame with demographic features
        """
        data = {}

        # Age in months (18-36 months target range)
        data['age_months'] = np.random.randint(18, 37, n_samples)

        # Gender (ASD is ~4x more common in males)
        # Overall: slightly more males in ASD group
        gender_probs = np.where(asd_labels == 1, 0.80, 0.52)  # Prob of male
        data['gender'] = np.where(
            np.random.random(n_samples) < gender_probs, 'M', 'F'
        )

        # Geographic location
        # Weight towards urban areas (more access to screening)
        province_weights = np.array([0.25, 0.15, 0.10, 0.08, 0.08, 0.08, 0.08, 0.06, 0.06, 0.06])
        data['geographic_location'] = np.random.choice(
            self.PROVINCES,
            n_samples,
            p=province_weights
        )

        # Setting type
        setting_weights = [0.45, 0.30, 0.25]
        data['setting_type'] = np.random.choice(
            self.SETTING_TYPES,
            n_samples,
            p=setting_weights
        )

        # Family history of ASD (higher in ASD cases)
        data['family_history_asd'] = np.where(
            asd_labels == 1,
            np.random.binomial(1, 0.20, n_samples),
            np.random.binomial(1, 0.05, n_samples)
        )

        # Birth order
        data['birth_order'] = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.35, 0.30, 0.20, 0.10, 0.05])

        # Gestational age at birth (weeks)
        data['gestational_weeks'] = np.clip(
            np.random.normal(39, 2, n_samples),
            28, 42
        ).astype(int)

        return pd.DataFrame(data)

    def _generate_screening_scores(
        self,
        n_samples: int,
        asd_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generate simulated screening tool scores.

        Args:
            n_samples: Number of samples
            asd_labels: Array of ASD diagnosis labels

        Returns:
            DataFrame with screening scores
        """
        data = {}

        # M-CHAT-R score (0-20 scale, higher = more risk)
        typical_scores = np.clip(np.random.normal(2, 1.5, n_samples), 0, 8)
        asd_scores = np.clip(np.random.normal(12, 3, n_samples), 5, 20)
        data['mchat_score'] = np.where(
            asd_labels == 1,
            asd_scores,
            typical_scores
        ).astype(int)

        # Social Communication Score (0-10, higher = more concerns)
        data['social_communication_score'] = np.where(
            asd_labels == 1,
            np.clip(np.random.normal(7, 1.5, n_samples), 3, 10),
            np.clip(np.random.normal(2, 1, n_samples), 0, 5)
        ).round(1)

        # Restricted/Repetitive Behavior Score (0-10)
        data['rrb_score'] = np.where(
            asd_labels == 1,
            np.clip(np.random.normal(6, 2, n_samples), 2, 10),
            np.clip(np.random.normal(1.5, 1, n_samples), 0, 4)
        ).round(1)

        return pd.DataFrame(data)

    def generate_dataset(
        self,
        n_samples: int = 1000,
        asd_prevalence: float = 0.25,
        include_missing: bool = True,
        missing_rate: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.

        Args:
            n_samples: Total number of samples to generate
            asd_prevalence: Proportion of ASD-positive cases (for training, higher than real-world)
            include_missing: Whether to introduce missing values
            missing_rate: Proportion of missing values to introduce

        Returns:
            Complete DataFrame with all features and target
        """
        logger.info(f"Generating {n_samples} synthetic samples...")

        # Generate ASD labels with specified prevalence
        asd_labels = np.random.binomial(1, asd_prevalence, n_samples)
        logger.info(f"ASD prevalence in generated data: {asd_labels.mean():.2%}")

        # Generate all feature groups
        behavioral_df = self._generate_behavioral_features(n_samples, asd_labels)
        communication_df = self._generate_communication_features(n_samples, asd_labels)
        demographic_df = self._generate_demographic_features(n_samples, asd_labels)
        screening_df = self._generate_screening_scores(n_samples, asd_labels)

        # Combine all features
        df = pd.concat([
            demographic_df,
            behavioral_df,
            communication_df,
            screening_df
        ], axis=1)

        # Add target variable
        df['asd_diagnosis'] = asd_labels

        # Add participant ID
        df.insert(0, 'participant_id', [f'P{str(i).zfill(5)}' for i in range(1, n_samples + 1)])

        # Introduce missing values if requested
        if include_missing:
            df = self._introduce_missing_values(df, missing_rate)

        logger.info(f"Generated dataset shape: {df.shape}")
        return df

    def _introduce_missing_values(
        self,
        df: pd.DataFrame,
        missing_rate: float
    ) -> pd.DataFrame:
        """
        Introduce realistic missing values.

        Args:
            df: Input DataFrame
            missing_rate: Overall proportion of missing values

        Returns:
            DataFrame with missing values
        """
        df = df.copy()

        # Columns that can have missing values
        # (exclude ID and target)
        eligible_cols = [col for col in df.columns
                        if col not in ['participant_id', 'asd_diagnosis']]

        n_missing = int(len(df) * len(eligible_cols) * missing_rate)

        # Randomly select cells to make missing
        for _ in range(n_missing):
            row_idx = np.random.randint(0, len(df))
            col = np.random.choice(eligible_cols)
            df.loc[row_idx, col] = np.nan

        total_missing = df.isnull().sum().sum()
        logger.info(f"Introduced {total_missing} missing values ({total_missing / df.size:.2%} of data)")

        return df

    def save_dataset(
        self,
        df: pd.DataFrame,
        output_path: Path,
        filename: str = 'asd_screening_data.csv'
    ) -> Path:
        """
        Save dataset to file.

        Args:
            df: DataFrame to save
            output_path: Directory to save to
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        file_path = output_path / filename
        df.to_csv(file_path, index=False)

        logger.info(f"Dataset saved to {file_path}")
        return file_path


def generate_train_test_datasets(
    output_dir: str = 'data/raw',
    train_samples: int = 800,
    test_samples: int = 200,
    random_state: int = RANDOM_STATE
) -> Tuple[Path, Path]:
    """
    Generate separate train and test datasets.

    Args:
        output_dir: Directory to save datasets
        train_samples: Number of training samples
        test_samples: Number of test samples
        random_state: Random seed

    Returns:
        Tuple of (train_path, test_path)
    """
    generator = ASDDataGenerator(random_state=random_state)
    output_path = Path(output_dir)

    # Generate training data (higher ASD prevalence for balanced learning)
    train_df = generator.generate_dataset(
        n_samples=train_samples,
        asd_prevalence=0.30,
        include_missing=True,
        missing_rate=0.05
    )
    train_path = generator.save_dataset(train_df, output_path, 'asd_train_data.csv')

    # Generate test data (more realistic prevalence)
    np.random.seed(random_state + 1)  # Different seed for test data
    test_df = generator.generate_dataset(
        n_samples=test_samples,
        asd_prevalence=0.25,
        include_missing=True,
        missing_rate=0.03
    )
    test_path = generator.save_dataset(test_df, output_path, 'asd_test_data.csv')

    return train_path, test_path


def main():
    """Main function to generate synthetic data."""
    parser = argparse.ArgumentParser(description='Generate synthetic ASD screening data')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--output', type=str, default='data/raw', help='Output directory')
    parser.add_argument('--prevalence', type=float, default=0.25, help='ASD prevalence rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--split', action='store_true', help='Generate train/test split')

    args = parser.parse_args()

    if args.split:
        train_samples = int(args.samples * 0.8)
        test_samples = args.samples - train_samples
        train_path, test_path = generate_train_test_datasets(
            output_dir=args.output,
            train_samples=train_samples,
            test_samples=test_samples,
            random_state=args.seed
        )
        print(f"Train data: {train_path}")
        print(f"Test data: {test_path}")
    else:
        generator = ASDDataGenerator(random_state=args.seed)
        df = generator.generate_dataset(
            n_samples=args.samples,
            asd_prevalence=args.prevalence
        )
        output_path = generator.save_dataset(df, Path(args.output))
        print(f"Data saved to: {output_path}")


if __name__ == '__main__':
    main()
