import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
from pathlib import Path
from typing import List, Dict, Any
import json
import nltk
from nltk.corpus import stopwords
import re

# Download stopwords if not present
try:
    nltk.data.find('corps/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from sentiment_analysis.entity.config_entity import EDAConfig
from sentiment_analysis.utils.logging_setup import logger


class SentimentEDA:
    """
    Production-ready Exploratory Data Analysis (EDA) for sentiment analysis.
    Generates comprehensive visualizations and statistics with proper text cleaning.
    """
    
    def __init__(self, config: EDAConfig):
        self.config = config
        self.report_path = Path(config.report_dir)
        
        # Text cleaning configuration
        self.stop_words = set(stopwords.words('english')) | set(STOPWORDS)
        self.min_word_len = config.min_word_len if hasattr(config, 'min_word_len') else 3
        
    def _clean_text_for_analysis(self, texts: List[str]) -> List[str]:
        """Clean text for word frequency and wordcloud analysis."""
        cleaned = []
        for text in texts:
            # Basic cleaning: lowercase, remove special chars, filter short words
            words = re.findall(r'\b[a-zA-Z]{' + str(self.min_word_len) + ',}\b', 
                             text.lower())
            words = [w for w in words if w not in self.stop_words]
            cleaned.extend(words)
        return cleaned
    
    def plot_class_distribution(self, df: pd.DataFrame, file_name: str = "class_distribution.png"):
        """Enhanced class distribution with imbalance metrics."""
        plt.figure(figsize=(8, 6))
        
        # Count plot
        ax1 = plt.gca()
        sns.countplot(x='label', data=df, ax=ax1, palette=['#ff6b6b', '#4ecdc4'])
        ax1.set_title('Sentiment Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Sentiment (0=Negative, 1=Positive)')
        ax1.set_ylabel('Count')
        
        # Add imbalance ratio as text
        class_counts = df['label'].value_counts()
        imbalance_ratio = class_counts[1] / class_counts[0] if class_counts[0] > 0 else 0
        ax1.text(0.02, 0.98, f'Imbalance Ratio (Pos/Neg): {imbalance_ratio:.2f}', 
                transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.report_path / file_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved class distribution to {file_name}")
    
    def plot_text_length_distribution(self, df: pd.DataFrame, file_name: str = "text_length_distribution.png"):
        """Text length distribution by sentiment."""
        df['text_len'] = df['text'].str.len()
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='text_len', hue='label', bins=50, 
                    palette=['#ff6b6b', '#4ecdc4'], alpha=0.7, stat='density')
        plt.title('Text Length Distribution by Sentiment', fontsize=14, fontweight='bold')
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Density')
        plt.legend(labels=['Negative', 'Positive'])
        plt.tight_layout()
        plt.savefig(self.report_path / file_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved text length distribution to {file_name}")
    
    def plot_word_frequency(self, texts: List[str], file_name: str, title: str, max_words: int = None):
        """Enhanced word frequency with relative frequency scores."""
        if max_words is None:
            max_words = self.config.max_words_to_plot
            
        # Clean and count words
        cleaned_words = self._clean_text_for_analysis(texts)
        word_counts = Counter(cleaned_words).most_common(max_words)
        total_words = sum(count for _, count in word_counts)
        
        words, counts = zip(*word_counts)
        relative_freq = [count / total_words for count in counts]
        
        df_wc = pd.DataFrame({
            'word': words, 
            'count': counts,
            'relative_freq': relative_freq
        })
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='relative_freq', y='word', data=df_wc, palette='viridis')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Relative Frequency')
        plt.ylabel('Word')
        plt.tight_layout()
        plt.savefig(self.report_path / file_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved word frequency plot to {file_name}")
        
        return df_wc
    
    def generate_word_cloud(self, texts: List[str], file_name: str, title: str):
        """Enhanced wordcloud with custom stopwords and colormap."""
        cleaned_text = ' '.join(self._clean_text_for_analysis(texts))
        
        wordcloud = WordCloud(
            width=1200, 
            height=600, 
            background_color='white',
            stopwords=self.stop_words,
            colormap='viridis',
            max_words=self.config.max_words_to_plot
        ).generate(cleaned_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.report_path / file_name, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved word cloud to {file_name}")
    
    def save_summary_stats(self, df: pd.DataFrame, positive_wc: pd.DataFrame, 
                          negative_wc: pd.DataFrame):
        """Save comprehensive EDA statistics as JSON."""
        stats = {
            'class_distribution': df['label'].value_counts(normalize=True).to_dict(),
            'text_stats': {
                'avg_length': df['text'].str.len().mean(),
                'max_length': df['text'].str.len().max(),
                'min_length': df['text'].str.len().min()
            },
            'positive_top_words': positive_wc[['word', 'relative_freq']].to_dict('records'),
            'negative_top_words': negative_wc[['word', 'relative_freq']].to_dict('records')
        }
        
        stats_path = self.report_path / 'eda_summary_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved EDA summary stats to {stats_path}")
    
    def run_full_eda(self, raw_texts: List[str], raw_labels: List[int], 
                    val_texts: List[str] = None, val_labels: List[int] = None):
        """
        Complete EDA pipeline for train/validation data.
        """
        logger.info("Starting comprehensive Sentiment EDA execution...")
        
        # Create DataFrame(s)
        df_train = pd.DataFrame({'text': raw_texts, 'label': raw_labels})
        
        # 1. Class distribution & text length
        self.plot_class_distribution(df_train)
        self.plot_text_length_distribution(df_train)
        
        # 2. Separate by sentiment
        positive_texts = df_train[df_train['label'] == 1]['text'].tolist()
        negative_texts = df_train[df_train['label'] == 0]['text'].tolist()
        
        # 3. Positive sentiment analysis
        pos_wc = self.plot_word_frequency(
            positive_texts, 
            "word_frequency_positive.png", 
            f"Top {self.config.max_words_to_plot} Positive Words (Relative Freq)"
        )
        self.generate_word_cloud(
            positive_texts, 
            "word_cloud_positive.png", 
            "Word Cloud - Positive Sentiment"
        )
        
        # 4. Negative sentiment analysis
        neg_wc = self.plot_word_frequency(
            negative_texts, 
            "word_frequency_negative.png", 
            f"Top {self.config.max_words_to_plot} Negative Words (Relative Freq)"
        )
        self.generate_word_cloud(
            negative_texts, 
            "word_cloud_negative.png", 
            "Word Cloud - Negative Sentiment"
        )
        
        # 5. Save comprehensive stats
        self.save_summary_stats(df_train, pos_wc, neg_wc)
        
        logger.info("✅ Full Sentiment EDA completed. All reports saved.")
        logger.info(f"📁 EDA reports saved to: {self.report_path}")
        
        return {
            'train_df': df_train,
            'positive_word_freq': pos_wc,
            'negative_word_freq': neg_wc,
            'stats': self.report_path / 'eda_summary_stats.json'
        }
        