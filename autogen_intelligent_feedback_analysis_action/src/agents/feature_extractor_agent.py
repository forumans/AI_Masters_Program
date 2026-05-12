"""
AutoGen-based Feature Extractor Agent for analyzing feature requests
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

class FeatureExtractorAgent:
    """
    AutoGen agent for analyzing feature requests and extracting insights
    """
    
    def __init__(self, impact_threshold: float = 0.6):
        """
        Initialize the Feature Extractor Agent
        
        Args:
            impact_threshold (float): Threshold for feature impact assessment
        """
        self.impact_threshold = impact_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize AutoGen agents
        self._setup_autogen_agents()
        
        # Feature priority levels
        self.priority_levels = ['Critical', 'High', 'Medium', 'Low']
        
        # Feature categories
        self.feature_categories = [
            'UI/UX', 'Performance', 'Security', 'Integration', 
            'Functionality', 'Accessibility', 'Mobile', 'Analytics'
        ]
        
        # Feature request patterns
        self.feature_patterns = {
            'add': ['add', 'addition', 'new', 'include', 'implement'],
            'improve': ['improve', 'enhance', 'better', 'upgrade', 'optimize'],
            'integrate': ['integrate', 'connect', 'sync', 'link', 'api'],
            'customize': ['customize', 'personalize', 'settings', 'options', 'configur'],
            'automate': ['automate', 'automatic', 'auto', 'schedule', 'batch']
        }
        
        # Impact indicators
        self.impact_indicators = {
            'high': ['critical', 'urgent', 'important', 'essential', 'must-have'],
            'medium': ['useful', 'helpful', 'nice', 'good', 'valuable'],
            'low': ['minor', 'small', 'nice-to-have', 'optional', 'convenience']
        }
    
    def _setup_autogen_agents(self):
        """Setup AutoGen agents for feature extraction"""
        try:
            config_list = load_config_list(base_dir=os.path.join(os.path.dirname(__file__), '..', '..'))
            if not autogen_is_ready(config_list):
                raise RuntimeError("AutoGen dependencies or configuration are unavailable")
            
            # Create feature extractor assistant agent
            self.feature_extractor = AssistantAgent(
                name="feature_extractor",
                llm_config={
                    "config_list": config_list,
                    "temperature": 0.1,
                    "timeout": 120,
                },
                system_message="""You are an expert feature analysis specialist. 
                Your task is to analyze feature requests and extract:
                1. Feature category (UI/UX, Performance, Security, etc.)
                2. Feature priority (Critical, High, Medium, Low)
                3. Feature impact on users
                4. Implementation complexity
                5. Target user segment
                6. Expected benefits
                
                Analyze the feature request and provide a JSON response with:
                {
                    "category": "UI/UX",
                    "priority": "High",
                    "impact_score": 0.8,
                    "complexity": "Medium",
                    "target_users": "Power users",
                    "benefits": ["Improved productivity", "Better user experience"],
                    "confidence": 0.85,
                    "reasoning": "Feature addresses common user pain points"
                }
                """
            )
            
            # Create user proxy agent
            self.user_proxy = UserProxyAgent(
                name="user_proxy",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=1,
                code_execution_config=False,
            )
            
            self.logger.info("Feature extraction AutoGen agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up feature extraction AutoGen agents: {str(e)}")
            # Fallback to rule-based extraction
            self.feature_extractor = None
            self.user_proxy = None
    
    def extract_feature_info(self, feedback_text: str, feedback_id: str = "") -> Dict:
        """
        Extract feature information from feedback
        
        Args:
            feedback_text (str): Feature request text to analyze
            feedback_id (str): ID of the feedback item
            
        Returns:
            Dict: Feature extraction result
        """
        try:
            # Try AutoGen extraction first
            if self.feature_extractor and self.user_proxy:
                return self._extract_with_autogen(feedback_text, feedback_id)
            else:
                return self._extract_with_rules(feedback_text, feedback_id)
        except Exception as e:
            self.logger.error(f"Error extracting feature info: {str(e)}")
            # Fallback to rule-based extraction
            return self._extract_with_rules(feedback_text, feedback_id)
    
    def _extract_with_autogen(self, feedback_text: str, feedback_id: str) -> Dict:
        """
        Extract feature information using AutoGen agents
        
        Args:
            feedback_text (str): Feature request text
            feedback_id (str): Feedback ID
            
        Returns:
            Dict: Extraction result
        """
        try:
            message = f"Please analyze this feature request: {feedback_text}"
            
            # Start the analysis conversation
            chat_result = self.user_proxy.initiate_chat(
                self.feature_extractor,
                message=message,
                max_turns=2
            )
            
            # Extract the last response
            last_message = chat_result.chat_history[-1]['content']
            
            # Parse JSON response
            import json
            result = json.loads(last_message)
            
            # Validate result
            if result['category'] not in self.feature_categories:
                # Assign default category if invalid
                result['category'] = 'Functionality'
            
            if result['priority'] not in self.priority_levels:
                result['priority'] = 'Medium'
            
            result['feedback_id'] = feedback_id
            return result
            
        except Exception as e:
            self.logger.error(f"AutoGen feature extraction failed: {str(e)}")
            # Fallback to rule-based
            return self._extract_with_rules(feedback_text, feedback_id)
    
    def _extract_with_rules(self, feedback_text: str, feedback_id: str) -> Dict:
        """
        Extract feature information using rule-based approach
        
        Args:
            feedback_text (str): Feature request text
            feedback_id (str): Feedback ID
            
        Returns:
            Dict: Extraction result
        """
        try:
            text_lower = feedback_text.lower()
            
            # Determine feature category
            category = self._determine_category(text_lower)
            
            # Determine priority
            priority = self._determine_priority(text_lower)
            
            # Calculate impact score
            impact_score = self._calculate_impact_score(text_lower)
            
            # Determine complexity
            complexity = self._determine_complexity(feedback_text)
            
            # Extract target users
            target_users = self._extract_target_users(text_lower)
            
            # Extract benefits
            benefits = self._extract_benefits(text_lower)
            
            # Calculate confidence
            confidence = min(len(benefits) * 0.15 + impact_score * 0.5, 0.9)
            
            return {
                'feedback_id': feedback_id,
                'category': category,
                'priority': priority,
                'impact_score': impact_score,
                'complexity': complexity,
                'target_users': target_users,
                'benefits': benefits,
                'confidence': confidence,
                'reasoning': f"Rule-based extraction: {category} category, {priority} priority"
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based feature extraction failed: {str(e)}")
            # Ultimate fallback
            return {
                'feedback_id': feedback_id,
                'category': 'Functionality',
                'priority': 'Medium',
                'impact_score': 0.5,
                'complexity': 'Medium',
                'target_users': 'General users',
                'benefits': [],
                'confidence': 0.5,
                'reasoning': 'Default extraction due to error'
            }
    
    def _determine_category(self, text_lower: str) -> str:
        """Determine feature category based on text content"""
        category_keywords = {
            'UI/UX': ['interface', 'design', 'layout', 'ui', 'ux', 'visual', 'display'],
            'Performance': ['speed', 'performance', 'fast', 'slow', 'optimize', 'efficient'],
            'Security': ['security', 'secure', 'authentication', 'privacy', 'protect'],
            'Integration': ['integrate', 'connect', 'sync', 'api', 'third-party'],
            'Functionality': ['feature', 'function', 'capability', 'ability', 'option'],
            'Accessibility': ['accessibility', 'accessible', 'disabled', 'screen reader'],
            'Mobile': ['mobile', 'app', 'phone', 'tablet', 'ios', 'android'],
            'Analytics': ['analytics', 'report', 'statistics', 'data', 'metrics']
        }
        
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            category_scores[category] = score
        
        best_category = max(category_scores, key=category_scores.get)
        return best_category if category_scores[best_category] > 0 else 'Functionality'
    
    def _determine_priority(self, text_lower: str) -> str:
        """Determine feature priority based on text content"""
        for level, keywords in self.impact_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                if level == 'high':
                    return 'Critical'
                elif level == 'medium':
                    return 'High'
                else:
                    return 'Medium'
        
        return 'Low'
    
    def _calculate_impact_score(self, text_lower: str) -> float:
        """Calculate feature impact score"""
        impact_score = 0.5  # Base score
        
        # Add points for impact indicators
        for keyword in self.impact_indicators['high']:
            if keyword in text_lower:
                impact_score += 0.2
        
        for keyword in self.impact_indicators['medium']:
            if keyword in text_lower:
                impact_score += 0.1
        
        # Add points for user benefit indicators
        benefit_keywords = ['improve', 'better', 'easier', 'faster', 'save', 'reduce']
        impact_score += sum(0.05 for keyword in benefit_keywords if keyword in text_lower)
        
        return min(impact_score, 1.0)
    
    def _determine_complexity(self, text: str) -> str:
        """Determine implementation complexity"""
        complexity_indicators = {
            'High': ['integration', 'api', 'database', 'architecture', 'system'],
            'Medium': ['feature', 'functionality', 'module', 'component'],
            'Low': ['button', 'option', 'setting', 'display', 'simple']
        }
        
        text_lower = text.lower()
        complexity_scores = {}
        
        for level, keywords in complexity_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            complexity_scores[level] = score
        
        if complexity_scores.get('High', 0) > 0:
            return 'High'
        elif complexity_scores.get('Medium', 0) > complexity_scores.get('Low', 0):
            return 'Medium'
        else:
            return 'Low'
    
    def _extract_target_users(self, text_lower: str) -> str:
        """Extract target user segment"""
        user_segments = {
            'Power users': ['advanced', 'power', 'expert', 'professional'],
            'New users': ['beginner', 'new', 'first-time', 'novice'],
            'Business users': ['business', 'enterprise', 'corporate', 'work'],
            'General users': ['user', 'customer', 'people', 'everyone'],
            'Developers': ['developer', 'api', 'integration', 'technical']
        }
        
        for segment, keywords in user_segments.items():
            if any(keyword in text_lower for keyword in keywords):
                return segment
        
        return 'General users'
    
    def _extract_benefits(self, text_lower: str) -> List[str]:
        """Extract expected benefits from feature request"""
        benefit_patterns = [
            r'will (help|improve|enable|allow|provide|save)',
            r'(better|faster|easier|more efficient)',
            r'(reduce|decrease|minimize)',
            r'(increase|enhance|boost)',
            r'(save|automate|streamline)'
        ]
        
        benefits = []
        sentences = re.split(r'[.!?]+', text_lower)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                for pattern in benefit_patterns:
                    if re.search(pattern, sentence):
                        # Extract the benefit phrase
                        benefit = sentence.capitalize()
                        if benefit not in benefits:
                            benefits.append(benefit[:100])  # Limit length
                        break
        
        return benefits[:3]  # Limit to 3 benefits
    
    def extract_batch(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract feature information from a batch of feature requests
        
        Args:
            feature_df (pd.DataFrame): DataFrame with feature requests
            
        Returns:
            pd.DataFrame: DataFrame with feature extraction results
        """
        try:
            results = []
            
            for idx, row in feature_df.iterrows():
                feedback_text = row['content']
                feedback_id = row['id']
                
                extraction = self.extract_feature_info(feedback_text, feedback_id)
                
                result = {
                    'id': feedback_id,
                    'content': feedback_text,
                    'feature_category': extraction['category'],
                    'feature_priority': extraction['priority'],
                    'feature_impact_score': extraction['impact_score'],
                    'implementation_complexity': extraction['complexity'],
                    'target_users': extraction['target_users'],
                    'expected_benefits': ';'.join(extraction['benefits']),
                    'extraction_confidence': extraction['confidence'],
                    'extraction_reasoning': extraction['reasoning'],
                    'source_type': row.get('source_type', 'unknown'),
                    'timestamp': row.get('timestamp', '')
                }
                
                results.append(result)
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Extracted features from {idx + 1}/{len(feature_df)} requests")
            
            extracted_df = pd.DataFrame(results)
            self.logger.info(f"Successfully extracted features from {len(extracted_df)} requests")
            return extracted_df
            
        except Exception as e:
            self.logger.error(f"Error in batch feature extraction: {str(e)}")
            raise
    
    def get_feature_statistics(self, extracted_df: pd.DataFrame) -> Dict:
        """
        Get statistics about extracted features
        
        Args:
            extracted_df (pd.DataFrame): Extracted feature data
            
        Returns:
            Dict: Feature statistics
        """
        try:
            stats = {}
            
            # Category distribution
            category_counts = extracted_df['feature_category'].value_counts()
            stats['category_distribution'] = category_counts.to_dict()
            
            # Priority distribution
            priority_counts = extracted_df['feature_priority'].value_counts()
            stats['priority_distribution'] = priority_counts.to_dict()
            
            # Complexity distribution
            complexity_counts = extracted_df['implementation_complexity'].value_counts()
            stats['complexity_distribution'] = complexity_counts.to_dict()
            
            # Average impact score
            avg_impact = extracted_df['feature_impact_score'].mean()
            stats['average_impact_score'] = avg_impact
            
            # High impact features
            high_impact_count = len(extracted_df[extracted_df['feature_impact_score'] >= 0.8])
            stats['high_impact_features'] = high_impact_count
            
            # Target user segments
            user_counts = extracted_df['target_users'].value_counts()
            stats['target_user_segments'] = user_counts.to_dict()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating feature statistics: {str(e)}")
            return {}
