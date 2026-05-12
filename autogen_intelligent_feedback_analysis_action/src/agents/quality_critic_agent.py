"""
AutoGen-based Quality Critic Agent for reviewing and validating tickets
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

class QualityCriticAgent:
    """
    AutoGen agent for reviewing and validating the quality of tickets and classifications
    """
    
    def __init__(self, min_quality_score: float = 0.7):
        """
        Initialize the Quality Critic Agent
        
        Args:
            min_quality_score (float): Minimum acceptable quality score
        """
        self.min_quality_score = min_quality_score
        self.logger = logging.getLogger(__name__)
        
        # Initialize AutoGen agents
        self._setup_autogen_agents()
        
        # Quality levels
        self.quality_levels = ['Excellent', 'Good', 'Acceptable', 'Needs Improvement', 'Poor']
        
        # Quality criteria weights
        self.quality_weights = {
            'completeness': 0.25,
            'accuracy': 0.30,
            'clarity': 0.20,
            'relevance': 0.15,
            'actionability': 0.10
        }
    
    def _setup_autogen_agents(self):
        """Setup AutoGen agents for quality review"""
        try:
            config_list = load_config_list(base_dir=os.path.join(os.path.dirname(__file__), '..', '..'))
            if not autogen_is_ready(config_list):
                raise RuntimeError("AutoGen dependencies or configuration are unavailable")
            
            # Create quality critic assistant agent
            self.quality_critic = AssistantAgent(
                name="quality_critic",
                llm_config={
                    "config_list": config_list,
                    "temperature": 0.1,
                    "timeout": 120,
                },
                system_message="""You are an expert quality assurance specialist. 
                Your task is to review and validate the quality of tickets and classifications.
                
                For each item, assess:
                1. Completeness (all required fields present)
                2. Accuracy (correct classification and analysis)
                3. Clarity (clear and understandable)
                4. Relevance (relevant to the feedback)
                5. Actionability (can be acted upon)
                
                Provide a JSON response with:
                {
                    "overall_score": 0.85,
                    "quality_level": "Good",
                    "completeness_score": 0.9,
                    "accuracy_score": 0.8,
                    "clarity_score": 0.85,
                    "relevance_score": 0.9,
                    "actionability_score": 0.8,
                    "issues": ["Missing reproduction steps"],
                    "suggestions": ["Add specific error messages"],
                    "needs_manual_review": false,
                    "reasoning": "Good quality but missing some details"
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
            
            self.logger.info("Quality critic AutoGen agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error setting up quality critic AutoGen agents: {str(e)}")
            # Fallback to rule-based review
            self.quality_critic = None
            self.user_proxy = None
    
    def review_ticket_quality(self, ticket_data: Dict) -> Dict:
        """
        Review the quality of a ticket
        
        Args:
            ticket_data (Dict): Ticket data to review
            
        Returns:
            Dict: Quality review result
        """
        try:
            # Try AutoGen review first
            if self.quality_critic and self.user_proxy:
                return self._review_with_autogen(ticket_data)
            else:
                return self._review_with_rules(ticket_data)
        except Exception as e:
            self.logger.error(f"Error reviewing ticket quality: {str(e)}")
            # Fallback to rule-based review
            return self._review_with_rules(ticket_data)
    
    def _review_with_autogen(self, ticket_data: Dict) -> Dict:
        """
        Review ticket quality using AutoGen agents
        
        Args:
            ticket_data (Dict): Ticket data
            
        Returns:
            Dict: Quality review result
        """
        try:
            # Prepare input for AutoGen
            input_text = f"""
            Ticket ID: {ticket_data.get('ticket_id', '')}
            Title: {ticket_data.get('title', '')}
            Description: {ticket_data.get('description', '')[:500]}...
            Type: {ticket_data.get('type', '')}
            Priority: {ticket_data.get('priority', '')}
            """
            
            message = f"Please review the quality of this ticket: {input_text}"
            
            # Start the review conversation
            chat_result = self.user_proxy.initiate_chat(
                self.quality_critic,
                message=message,
                max_turns=2
            )
            
            # Extract the last response
            last_message = chat_result.chat_history[-1]['content']
            
            # Parse JSON response
            import json
            review_result = json.loads(last_message)
            
            # Validate and enhance review
            review_result = self._validate_and_enhance_review(review_result, ticket_data)
            
            return review_result
            
        except Exception as e:
            self.logger.error(f"AutoGen quality review failed: {str(e)}")
            # Fallback to rule-based
            return self._review_with_rules(ticket_data)
    
    def _review_with_rules(self, ticket_data: Dict) -> Dict:
        """
        Review ticket quality using rule-based approach
        
        Args:
            ticket_data (Dict): Ticket data
            
        Returns:
            Dict: Quality review result
        """
        try:
            # Initialize scores
            scores = {
                'completeness': self._assess_completeness(ticket_data),
                'accuracy': self._assess_accuracy(ticket_data),
                'clarity': self._assess_clarity(ticket_data),
                'relevance': self._assess_relevance(ticket_data),
                'actionability': self._assess_actionability(ticket_data)
            }
            
            # Calculate overall score
            overall_score = sum(scores[criterion] * weight 
                              for criterion, weight in self.quality_weights.items())
            
            # Determine quality level
            quality_level = self._get_quality_level(overall_score)
            
            # Identify issues and suggestions
            issues, suggestions = self._identify_issues_and_suggestions(ticket_data, scores)
            
            # Determine if manual review is needed
            needs_manual_review = overall_score < self.min_quality_score or len(issues) > 2
            
            return {
                'ticket_id': ticket_data.get('ticket_id', ''),
                'overall_score': overall_score,
                'quality_level': quality_level,
                'completeness_score': scores['completeness'],
                'accuracy_score': scores['accuracy'],
                'clarity_score': scores['clarity'],
                'relevance_score': scores['relevance'],
                'actionability_score': scores['actionability'],
                'issues': issues,
                'suggestions': suggestions,
                'needs_manual_review': needs_manual_review,
                'reasoning': f"Rule-based review: {quality_level} quality with score {overall_score:.3f}"
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based quality review failed: {str(e)}")
            # Ultimate fallback
            return {
                'ticket_id': ticket_data.get('ticket_id', ''),
                'overall_score': 0.5,
                'quality_level': 'Needs Improvement',
                'completeness_score': 0.5,
                'accuracy_score': 0.5,
                'clarity_score': 0.5,
                'relevance_score': 0.5,
                'actionability_score': 0.5,
                'issues': ['Review failed due to error'],
                'suggestions': ['Manual review required'],
                'needs_manual_review': True,
                'reasoning': 'Default review due to error'
            }
    
    def _validate_and_enhance_review(self, review_result: Dict, ticket_data: Dict) -> Dict:
        """Validate and enhance review result"""
        # Ensure required fields
        if 'ticket_id' not in review_result:
            review_result['ticket_id'] = ticket_data.get('ticket_id', '')
        
        # Validate score ranges
        for score_field in ['overall_score', 'completeness_score', 'accuracy_score', 
                           'clarity_score', 'relevance_score', 'actionability_score']:
            if score_field in review_result:
                score = review_result[score_field]
                review_result[score_field] = max(0.0, min(1.0, score))
        
        # Validate quality level
        if review_result.get('quality_level') not in self.quality_levels:
            # Determine level based on score
            overall_score = review_result.get('overall_score', 0.5)
            review_result['quality_level'] = self._get_quality_level(overall_score)
        
        # Ensure issues and suggestions are lists
        if 'issues' in review_result and isinstance(review_result['issues'], str):
            review_result['issues'] = [review_result['issues']]
        
        if 'suggestions' in review_result and isinstance(review_result['suggestions'], str):
            review_result['suggestions'] = [review_result['suggestions']]
        
        return review_result
    
    def _assess_completeness(self, ticket_data: Dict) -> float:
        """Assess ticket completeness"""
        required_fields = ['ticket_id', 'title', 'description', 'type', 'priority', 'status']
        optional_fields = ['assignee', 'labels', 'estimated_effort']
        
        score = 0.0
        
        # Check required fields
        present_required = sum(1 for field in required_fields 
                              if field in ticket_data and ticket_data[field])
        score += (present_required / len(required_fields)) * 0.7
        
        # Check optional fields
        present_optional = sum(1 for field in optional_fields 
                              if field in ticket_data and ticket_data[field])
        score += (present_optional / len(optional_fields)) * 0.3
        
        return score
    
    def _assess_accuracy(self, ticket_data: Dict) -> float:
        """Assess ticket accuracy"""
        score = 0.8  # Base score
        
        # Check for valid ticket type
        valid_types = ['Bug', 'Feature Request', 'Improvement', 'Investigation', 'Documentation']
        if ticket_data.get('type') not in valid_types:
            score -= 0.2
        
        # Check for valid priority
        valid_priorities = ['Critical', 'High', 'Medium', 'Low']
        if ticket_data.get('priority') not in valid_priorities:
            score -= 0.2
        
        # Check for valid status
        valid_statuses = ['Open', 'In Progress', 'Pending', 'Resolved', 'Closed']
        if ticket_data.get('status') not in valid_statuses:
            score -= 0.1
        
        return max(0.0, score)
    
    def _assess_clarity(self, ticket_data: Dict) -> float:
        """Assess ticket clarity"""
        score = 0.0
        
        # Title clarity
        title = ticket_data.get('title', '')
        if len(title) > 10 and len(title) < 200:
            score += 0.3
        elif len(title) >= 200:
            score += 0.1
        
        # Description clarity
        description = ticket_data.get('description', '')
        if len(description) > 50:
            score += 0.4
            # Check for structure
            if any(marker in description.lower() for marker in ['**', 'steps:', 'expected:']):
                score += 0.2
        
        # Check for proper formatting
        if any(marker in str(ticket_data.values()) for marker in ['**', '*', '-', '1.']):
            score += 0.1
        
        return score
    
    def _assess_relevance(self, ticket_data: Dict) -> float:
        """Assess ticket relevance"""
        score = 0.8  # Base score
        
        # Check if title and description are related
        title = ticket_data.get('title', '').lower()
        description = ticket_data.get('description', '').lower()
        
        # Simple relevance check - do they share keywords?
        title_words = set(re.findall(r'\b\w+\b', title))
        desc_words = set(re.findall(r'\b\w+\b', description))
        
        if title_words and desc_words:
            overlap = len(title_words.intersection(desc_words))
            if overlap > 0:
                score += min(0.2, overlap * 0.05)
        
        return min(1.0, score)
    
    def _assess_actionability(self, ticket_data: Dict) -> float:
        """Assess ticket actionability"""
        score = 0.0
        
        # Check for assignee
        if ticket_data.get('assignee'):
            score += 0.3
        
        # Check for estimated effort
        if ticket_data.get('estimated_effort'):
            score += 0.2
        
        # Check for reproduction steps (for bugs) or expected outcome (for features)
        ticket_type = ticket_data.get('type', '').lower()
        if 'bug' in ticket_type and ticket_data.get('reproduction_steps'):
            score += 0.3
        elif 'feature' in ticket_type and ticket_data.get('expected_outcome'):
            score += 0.3
        
        # Check for labels
        labels = ticket_data.get('labels', [])
        if isinstance(labels, list) and len(labels) > 0:
            score += 0.2
        
        return score
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level based on score"""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Acceptable'
        elif score >= 0.6:
            return 'Needs Improvement'
        else:
            return 'Poor'
    
    def _identify_issues_and_suggestions(self, ticket_data: Dict, scores: Dict) -> Tuple[List[str], List[str]]:
        """Identify issues and suggestions for improvement"""
        issues = []
        suggestions = []
        
        # Check completeness
        if scores['completeness'] < 0.7:
            issues.append("Missing required fields")
            suggestions.append("Ensure all required fields are completed")
        
        # Check clarity
        if scores['clarity'] < 0.6:
            issues.append("Unclear title or description")
            suggestions.append("Improve clarity with better formatting and structure")
        
        # Check actionability
        if scores['actionability'] < 0.5:
            issues.append("Lacks actionable information")
            suggestions.append("Add assignee, effort estimate, and clear steps")
        
        # Check accuracy
        if scores['accuracy'] < 0.7:
            issues.append("Invalid field values")
            suggestions.append("Verify ticket type, priority, and status values")
        
        # Specific checks
        if not ticket_data.get('assignee'):
            issues.append("No assignee specified")
            suggestions.append("Assign to appropriate team member")
        
        if not ticket_data.get('estimated_effort'):
            issues.append("No effort estimate")
            suggestions.append("Provide effort estimate in story points")
        
        ticket_type = ticket_data.get('type', '').lower()
        if 'bug' in ticket_type and not ticket_data.get('reproduction_steps'):
            issues.append("Missing reproduction steps")
            suggestions.append("Add clear steps to reproduce the bug")
        
        return issues, suggestions
    
    def review_batch_tickets(self, tickets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Review quality of a batch of tickets
        
        Args:
            tickets_df (pd.DataFrame): Tickets data
            
        Returns:
            pd.DataFrame: Quality review results
        """
        try:
            reviews = []
            
            for idx, row in tickets_df.iterrows():
                ticket_data = row.to_dict()
                
                # Review ticket quality
                review = self.review_ticket_quality(ticket_data)
                reviews.append(review)
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    self.logger.info(f"Reviewed {idx + 1}/{len(tickets_df)} tickets")
            
            reviews_df = pd.DataFrame(reviews)
            self.logger.info(f"Successfully reviewed {len(reviews_df)} tickets")
            return reviews_df
            
        except Exception as e:
            self.logger.error(f"Error in batch quality review: {str(e)}")
            raise
    
    def get_quality_metrics(self, reviews_df: pd.DataFrame) -> Dict:
        """
        Get quality metrics from reviews
        
        Args:
            reviews_df (pd.DataFrame): Quality review data
            
        Returns:
            Dict: Quality metrics
        """
        try:
            metrics = {}
            
            # Quality level distribution
            quality_counts = reviews_df['quality_level'].value_counts()
            metrics['quality_distribution'] = quality_counts.to_dict()
            
            # Average scores
            metrics['average_overall_score'] = reviews_df['overall_score'].mean()
            metrics['average_completeness_score'] = reviews_df['completeness_score'].mean()
            metrics['average_accuracy_score'] = reviews_df['accuracy_score'].mean()
            metrics['average_clarity_score'] = reviews_df['clarity_score'].mean()
            metrics['average_relevance_score'] = reviews_df['relevance_score'].mean()
            metrics['average_actionability_score'] = reviews_df['actionability_score'].mean()
            
            # Tickets needing manual review
            needs_review_count = len(reviews_df[reviews_df['needs_manual_review'] == True])
            metrics['needs_manual_review_count'] = needs_review_count
            metrics['needs_manual_review_percentage'] = (needs_review_count / len(reviews_df)) * 100
            
            # Common issues
            all_issues = []
            for issues_list in reviews_df['issues']:
                if isinstance(issues_list, list):
                    all_issues.extend(issues_list)
            
            from collections import Counter
            issue_counts = Counter(all_issues)
            metrics['common_issues'] = dict(issue_counts.most_common(5))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating quality metrics: {str(e)}")
            return {}
    
    def save_quality_reviews(self, reviews_df: pd.DataFrame, output_path: str) -> bool:
        """
        Save quality reviews to CSV file
        
        Args:
            reviews_df (pd.DataFrame): Quality review data
            output_path (str): Output file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            reviews_df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(reviews_df)} quality reviews to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving quality reviews: {str(e)}")
            return False
