"""
AutoGen-based Feedback Classifier Agent for categorizing user feedback
"""

import pandas as pd
import numpy as np
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from autogen_support import AssistantAgent, UserProxyAgent, autogen_is_ready, load_config_list

class FeedbackClassifierAgent:
    """
    AutoGen agent for classifying user feedback into categories
    """
    
    def __init__(self, model_dir: str = "models", confidence_threshold: float = 0.7):
        """
        Initialize the Feedback Classifier Agent
        
        Args:
            model_dir (str): Directory to save/load ML models
            confidence_threshold (float): Minimum confidence for classification
        """
        self.model_dir = model_dir
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize AutoGen agents
        self._setup_autogen_agents()
        
        # Classification categories
        self.categories = ['Bug', 'Feature Request', 'Praise', 'Complaint', 'Spam']
        
        # Keyword patterns for rule-based classification
        self.keyword_patterns = {
            'Bug': ['crash', 'error', 'bug', 'broken', 'not working', 'issue', 'problem', 'freeze', 'slow'],
            'Feature Request': ['feature', 'add', 'implement', 'new', 'would like', 'suggest', 'enhancement'],
            'Praise': ['love', 'great', 'amazing', 'excellent', 'perfect', 'awesome', 'fantastic', 'good'],
            'Complaint': ['bad', 'terrible', 'awful', 'hate', 'disappointed', 'frustrated', 'annoying'],
            'Spam': ['spam', 'advertisement', 'promo', 'offer', 'buy now', 'click here', 'free money']
        }
    
    def _setup_autogen_agents(self):
        """Setup AutoGen agents for classification"""
        try:
            config_list = load_config_list(base_dir=os.path.join(os.path.dirname(__file__), '..', '..'))
            if not autogen_is_ready(config_list):
                raise RuntimeError("AutoGen dependencies or configuration are unavailable")
            
            # Create classifier assistant agent
            self.classifier_agent = AssistantAgent(
                name="feedback_classifier",
                llm_config={
                    "config_list": config_list,
                    "temperature": 0.1,
                    "timeout": 120,
                },
                system_message="""You are an expert feedback classification specialist. 
                Your task is to categorize user feedback into one of these categories:
                - Bug: Technical issues, crashes, errors
                - Feature Request: Suggestions for new features
                - Praise: Positive feedback and compliments
                - Complaint: Negative feedback about user experience
                - Spam: Irrelevant or promotional content
                
                Analyze the feedback and provide:
                1. Category (one of the above)
                2. Confidence score (0-1)
                3. Brief reasoning
                
                Format your response as JSON:
                {"category": "Bug", "confidence": 0.85, "reasoning": "User reports app crashing"}
                """
            )
            
            # Create user proxy agent
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            self.logger.info("AutoGen agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up AutoGen agents: {str(e)}")
            # Fallback to rule-based classification
            self.classifier_agent = None
            self.user_proxy = None
    
    def classify_feedback(self, feedback_text: str) -> Dict:
        """
        Classify a single feedback item
        
        Args:
            feedback_text (str): Feedback text to classify
            
        Returns:
            Dict: Classification result with category, confidence, and reasoning
        """
        try:
            # Try AutoGen classification first
            if self.classifier_agent and self.user_proxy:
                return self._classify_with_autogen(feedback_text)
            else:
                return self._classify_with_rules(feedback_text)
        except Exception as e:
            self.logger.error(f"Error classifying feedback: {str(e)}")
            # Fallback to rule-based classification
            return self._classify_with_rules(feedback_text)
    
    def _classify_with_autogen(self, feedback_text: str) -> Dict:
        """
        Classify feedback using AutoGen agents
        
        Args:
            feedback_text (str): Feedback text to classify
            
        Returns:
            Dict: Classification result
        """
        try:
            message = f"Please classify this feedback: {feedback_text}"
            
            # Start the classification conversation
            chat_result = self.user_proxy.initiate_chat(
                self.classifier_agent,
                message=message,
                max_turns=2
            )
            
            # Extract the last response
            last_message = chat_result.chat_history[-1]['content']
            
            # Parse JSON response
            import json
            result = json.loads(last_message)
            
            # Validate result
            if result['category'] not in self.categories:
                raise ValueError(f"Invalid category: {result['category']}")
            
            return result
        except Exception as e:
            self.logger.error(f"AutoGen classification failed: {str(e)}")
            # Fallback to rule-based
            return self._classify_with_rules(feedback_text)
    
    def _classify_with_rules(self, feedback_text: str) -> Dict:
        """
        Classify feedback using rule-based approach
        
        Args:
            feedback_text (str): Feedback text to classify
            
        Returns:
            Dict: Classification result
        """
        try:
            text_lower = feedback_text.lower()
            scores = {}
            
            # Calculate scores for each category
            for category, keywords in self.keyword_patterns.items():
                score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1
                scores[category] = score
            
            # Get the category with highest score
            if max(scores.values()) == 0:
                # Default to 'Feature Request' if no keywords match
                best_category = 'Feature Request'
                confidence = 0.5
            else:
                best_category = max(scores, key=scores.get)
                confidence = min(scores[best_category] / 3.0, 0.9)  # Normalize confidence
            
            return {
                'category': best_category,
                'confidence': confidence,
                'reasoning': f"Rule-based classification based on keywords: {self.keyword_patterns.get(best_category, [])}"
            }
        except Exception as e:
            self.logger.error(f"Rule-based classification failed: {str(e)}")
            # Ultimate fallback
            return {
                'category': 'Feature Request',
                'confidence': 0.5,
                'reasoning': 'Default classification due to error'
            }
    
    def classify_batch(self, feedback_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify a batch of feedback items
        
        Args:
            feedback_df (pd.DataFrame): DataFrame with feedback data
            
        Returns:
            pd.DataFrame: DataFrame with classification results added
        """
        try:
            results = []
            
            for idx, row in feedback_df.iterrows():
                feedback_text = row['content']
                classification = self.classify_feedback(feedback_text)
                
                result = {
                    'id': row['id'],
                    'content': feedback_text,
                    'predicted_category': classification['category'],
                    'classification_confidence': classification['confidence'],
                    'classification_reasoning': classification['reasoning'],
                    'source_type': row.get('source_type', 'unknown'),
                    'timestamp': row.get('timestamp', '')
                }
                
                results.append(result)
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Classified {idx + 1}/{len(feedback_df)} feedback items")
            
            classified_df = pd.DataFrame(results)
            self.logger.info(f"Successfully classified {len(classified_df)} feedback items")
            return classified_df
            
        except Exception as e:
            self.logger.error(f"Error in batch classification: {str(e)}")
            raise
    
    def evaluate_classification(self, classified_df: pd.DataFrame, expected_df: pd.DataFrame) -> Dict:
        """
        Evaluate classification accuracy against expected results
        
        Args:
            classified_df (pd.DataFrame): Classified results
            expected_df (pd.DataFrame): Expected classifications
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            # Merge classified and expected data
            merged_df = pd.merge(classified_df, expected_df, on='id', suffixes=('_pred', '_exp'))
            
            # Calculate accuracy
            correct_predictions = (merged_df['predicted_category'] == merged_df['expected_category']).sum()
            total_predictions = len(merged_df)
            accuracy = correct_predictions / total_predictions
            
            # Calculate per-category metrics
            category_metrics = {}
            for category in self.categories:
                category_mask = merged_df['expected_category'] == category
                if category_mask.sum() > 0:
                    category_accuracy = (merged_df.loc[category_mask, 'predicted_category'] == 
                                       merged_df.loc[category_mask, 'expected_category']).sum() / category_mask.sum()
                    category_metrics[category] = category_accuracy
            
            evaluation = {
                'overall_accuracy': accuracy,
                'total_correct': correct_predictions,
                'total_predictions': total_predictions,
                'category_metrics': category_metrics,
                'average_confidence': classified_df['classification_confidence'].mean()
            }
            
            self.logger.info(f"Classification accuracy: {accuracy:.2%}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating classification: {str(e)}")
            return {'overall_accuracy': 0, 'error': str(e)}
